import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class DsExtractor:
    def __init__(
        self,
        model: str = "Pro/deepseek-ai/DeepSeek-V3.2",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key_env: str = "SILICONFLOW_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 180,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"未检测到 {api_key_env}，请检查 .env 是否已正确加载")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
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
        user_prompt = f"""请提取以下文本中的三元组。来源标记请统一使用：'{source_reference}'。

文本内容：
{text}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("抽取模型返回空响应")
        return content

    def extract(self, text: str, source_reference: str) -> str:
        """直接对完整文本执行一次抽取。"""
        return self._extract_once(text, source_reference)