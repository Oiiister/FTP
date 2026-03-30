import os
import json
import dashscope
from tenacity import retry, stop_after_attempt, wait_exponential


class QwenExtractor:
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未检测到 DASHSCOPE_API_KEY，请检查 .env 是否已正确加载")

        dashscope.api_key = api_key
        self.model = "qwen-max"
        self.long_text_threshold = 5000
        self.chunk_size = 3000
        self.chunk_overlap = 300
        # 深度优化的 System Prompt
        self.system_prompt = """
        你是一个精通故障树分析（FTA）的专家。你的任务是从技术文本中抽取出严谨的故障三元组。

        ### 1. 核心任务目标
        - **逻辑层次化（核心）**：严禁将嵌套逻辑扁平化。如果存在多组并发原因（与门）通过“或”逻辑指向同一结果，必须发明【中间合成节点】。
        - **全面性**：必须体现故障传递链 (Basic -> Intermediate -> Top)。
        - **逻辑准确性**：严格区分 `resultsIn`（或门）和 `jointly_resultsIn`（与门）。

        ### 2. 逻辑分层与“中间节点”发明规范
        当遇到复杂逻辑如 “(A和B同时发生) 或者 (C和D同时发生) 导致 E” 时：
        - **错误做法**：直接将 A,B,C,D 全部通过 jointly_resultsIn 指向 E（这会导致逻辑变成 A∧B∧C∧D→E）。
        - **正确做法**：
            1. 发明节点“A与B组合触发”作为 IntermediateEvent。
            2. 建立三元组：(A, jointly_resultsIn, A与B组合触发), (B, jointly_resultsIn, A与B组合触发)。
            3. 建立三元组：(C, jointly_resultsIn, C与D组合触发), (D, jointly_resultsIn, C与D组合触发)。
            4. 汇总：(A与B组合触发, resultsIn, E), (C与D组合触发, resultsIn, E)。

        ### 3. 参数定义规范
        - **subject_name / object_name**: 故障描述词（如：阀门内漏、信号丢包）。
        - **subject_type / object_type**:
            - `BasicEvent`: 故障的最底层根源（通常是硬件损坏、人为操作错误、环境因素）。
            - `IntermediateEvent`: 故障链的中间环。它是由某种故障引起的，且会引发更严重的故障。
            - `TopEvent`: 最终观察到的、最严重的系统级故障现象。
        - **relation**:
            - `resultsIn`: 导致。用于单一诱因（或门）。触发词：导致、引起、造成、若...则...。
            - `jointly_resultsIn`: 共同导致。用于多个条件【同时满足】才发病的情况（与门）。触发词：且、同时、共同、...以及...才会。
            - `relatedTo`: 关联。用于描述两者有统计学相关性但因果不明的情况。
        - **confidence**: 动态打分（0.0-1.0）。
            - 描述确定（如“经查证是由于...”）: 0.98
            - 描述常规（如“会导致...”）: 0.90
            - 描述模糊（如“可能关联...”、“疑似...”）: 0.60-0.75
        - **source**: 必须忠实记录原文中描述该逻辑关系的原始文本片段，严禁概括或简化。

        ### 4. 嵌套逻辑案例分析 (Few-Shot)
        【输入文本】: 来源【技术文档01】。若[电源模块A]与[控制板B]同时失效，或[电源模块C]与[控制板D]同时失效，均会导致[系统宕机]。
        【期望JSON】:
        {
          "triplets": [
            {"subject_name": "电源模块A", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "组合故障路径1", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "若[电源模块A]与[控制板B]同时失效"},
            {"subject_name": "控制板B", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "组合故障路径1", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "若[电源模块A]与[控制板B]同时失效"},
            {"subject_name": "电源模块C", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "组合故障路径2", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "或[电源模块C]与[控制板D]同时失效"},
            {"subject_name": "控制板D", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "组合故障路径2", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "或[电源模块C]与[控制板D]同时失效"},
            {"subject_name": "组合故障路径1", "subject_type": "IntermediateEvent", "relation": "resultsIn", "object_name": "系统宕机", "object_type": "TopEvent", "confidence": 0.98, "source": "均会导致[系统宕机]"},
            {"subject_name": "组合故障路径2", "subject_type": "IntermediateEvent", "relation": "resultsIn", "object_name": "系统宕机", "object_type": "TopEvent", "confidence": 0.98, "source": "均会导致[系统宕机]"}
          ]
        }

        ### 5. 强制约束
        - 必须输出纯 JSON 格式。
        - 严禁对所有三元组使用统一的 confidence。
        - 严禁将所有 object_type 设为 TopEvent，必须体现故障传递过程。
        
        ### 6. 迭代追因协议（必须执行）
        你必须把故障抽取视为“逐层回溯”的迭代过程，而不是一次性平铺提取。
            
        ### 7. IntermediateEvent 多层约束
        - IntermediateEvent 可以有多层，严禁默认只有一层。
        - 若出现“由A导致B，B进一步导致C”，必须至少输出两条三元组：A->B、B->C。
        - 若某节点既有上游原因又有下游结果，则该节点必须是 IntermediateEvent。
        - 严禁将“仍可继续分解”的节点直接标为 BasicEvent。

        ### 8. 输出前自检（必须满足）
        1. 是否存在多层链条（例如 TopEvent <- IntermediateEvent <- IntermediateEvent <- BasicEvent）？
        2. 是否有 IntermediateEvent 只有入边或只有出边？若有，需补充链条或调整类型。
        3. 若文本明确存在中间传递过程，是否完整体现传递链，而非只保留首尾节点？

        ### 9. 执行步骤
        1. 先识别 TopEvent（最终故障现象）。
        2. 提取 TopEvent 的直接原因（这一层通常是 IntermediateEvent 或 BasicEvent）。
        3. 对每个直接原因继续追问：该事件是否还能被更上游原因解释？
        4. 只要还能继续解释，该节点必须标为 IntermediateEvent，并继续向上展开至少一层。
        5. 只有当文本中无法找到更上游原因时，才可标记为 BasicEvent。
        6. 每个 IntermediateEvent 必须尽量具备“入边+出边”（既有被谁导致，也有导致谁）。
        """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _extract_once(self, text: str, source_reference: str) -> str:
        # 在 User Prompt 中明确强调来源标记，防止模型瞎编 source
        user_prompt = f"""请提取以下文本中的三元组。来源标记请统一使用：'{source_reference}'。

文本内容：
{text}"""

        response = dashscope.Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            result_format='message',
            response_format={"type": "json_object"}
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"API调用失败: {response.code} - {response.message}")

    def _split_text(self, text: str) -> list[str]:
        """按段落优先切分长文本，保留少量重叠以减少跨段信息丢失。"""
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p for p in text.split("\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n{para}" if current else para
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                current = f"{overlap}\n{para}"
            else:
                # 单段过长时硬切分
                for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                    piece = para[i:i + self.chunk_size]
                    if piece.strip():
                        chunks.append(piece)
                current = ""

        if current.strip():
            chunks.append(current)

        return chunks

    def _safe_parse_triplets(self, raw_json: str) -> list[dict]:
        """尽量从模型返回中解析 triplets 列表，失败时返回空列表。"""
        try:
            data = json.loads(raw_json)
            triplets = data.get("triplets", [])
            if isinstance(triplets, list):
                return [t for t in triplets if isinstance(t, dict)]
            return []
        except Exception:
            return []

    def _deduplicate_triplets(self, triplets: list[dict]) -> list[dict]:
        """基于核心语义字段去重，避免分块结果重复。"""
        seen = set()
        result = []
        for t in triplets:
            key = (
                t.get("subject_name", "").strip(),
                t.get("subject_type", "").strip(),
                t.get("relation", "").strip(),
                t.get("object_name", "").strip(),
                t.get("object_type", "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(t)
        return result

    def _merge_triplets_globally(self, text: str, source_reference: str, triplets: list[dict]) -> str:
        """第二阶段全局合并：跨分块重建链路、补全中间层并统一命名。"""
        merge_prompt = (
            "你将看到同一文档不同片段抽取出的故障三元组候选。"
            "请进行全局融合，重点执行："
            "1) 跨上下文拼接完整因果链；"
            "2) 补全可证据支持的中间层 IntermediateEvent；"
            "3) 去重并统一同义命名（但不要误合并上下游事件）；"
            "4) 保留 OR/AND 逻辑正确性。"
            "只输出 JSON，格式为 {\"triplets\": [...]}。"
        )

        user_prompt = f"""来源标记统一使用：'{source_reference}'。

【原始文本（用于全局理解）】
{text[:6000]}

【分块候选三元组】
{json.dumps(triplets, ensure_ascii=False, indent=2)}
"""

        response = dashscope.Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": merge_prompt},
                {"role": "user", "content": user_prompt}
            ],
            result_format='message',
            response_format={"type": "json_object"}
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        raise Exception(f"全局融合失败: {response.code} - {response.message}")

    def extract(self, text: str, source_reference: str) -> str:
        """短文本直接抽取；长文本采用“分块抽取 + 全局融合”两阶段策略。"""
        if len(text) <= self.long_text_threshold:
            return self._extract_once(text, source_reference)

        chunks = self._split_text(text)
        all_triplets = []

        for idx, chunk in enumerate(chunks, 1):
            chunk_source = f"{source_reference} | chunk {idx}/{len(chunks)}"
            raw = self._extract_once(chunk, chunk_source)
            parsed = self._safe_parse_triplets(raw)
            all_triplets.extend(parsed)

        all_triplets = self._deduplicate_triplets(all_triplets)

        # 若分块阶段无结果，返回空结构，避免后续异常。
        if not all_triplets:
            return json.dumps({"triplets": []}, ensure_ascii=False)

        try:
            merged_raw = self._merge_triplets_globally(text, source_reference, all_triplets)
            merged_triplets = self._safe_parse_triplets(merged_raw)
        except Exception as e:
            print(f"[Extractor] 全局融合失败（{type(e).__name__}: {e}），降级使用分块去重结果")
            merged_triplets = []

        # 全局融合失败时兜底返回分块去重结果。
        if not merged_triplets:
            return json.dumps({"triplets": all_triplets}, ensure_ascii=False)

        return json.dumps({"triplets": self._deduplicate_triplets(merged_triplets)}, ensure_ascii=False)