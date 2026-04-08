import os
import json
import re
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import time

try:
    import jieba as jieba_module
except Exception:
    jieba_module = None


class DeepSeekExtractor:
    def __init__(self, semantic_merge_threshold: float = 0.86):
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("未检测到 DEEPSEEK_API_KEY，请检查 .env 是否已正确加载")
        if self._is_placeholder_key(api_key):
            raise ValueError("检测到 DEEPSEEK_API_KEY 为占位符，请替换为真实 DeepSeek API Key")

        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = "deepseek-chat"  # DeepSeek 平台通用模型标识
        self.temperature = 1.0  # 温度参数：控制随机性（0.0-2.0），1.0 为标准值
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.long_text_threshold = 5000
        # 父块最小长度：尽量保证每个父块包含完整故障上下文
        self.parent_chunk_min = 600
        # 父块最大长度：控制单次送入 LLM 的上下文规模
        self.parent_chunk_max = 1000
        # 父块重叠长度：避免跨块边界丢失关键参数或因果关系
        self.parent_chunk_overlap = 120
        # section 标题仅在较短时才拼接进每个父块，避免过长标题挤占正文上下文
        self.parent_section_prefix_max_tokens = 24
        # 子块最小长度：过滤过短、检索价值低的句子
        self.child_min_length = 15
        # 章节标题识别：匹配如“2.1.1 控制单元硬件故障数据”
        self.section_pattern = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s+(.+?)\s*$")
        # 故障码识别：匹配如 F01000/F01002
        self.fault_code_pattern = re.compile(r"\bF\d{5}\b", flags=re.IGNORECASE)
        # 组件实体识别：用于补充父子元数据中的 components 字段
        self.component_pattern = re.compile(
            r"([A-Za-z0-9\u4e00-\u9fa5]{2,30}"
            r"(?:控制单元|功率单元|电源模块|控制板|模块|单元|组件|电机|驱动器|控制器|传感器|编码器|变频器|逆变器))"
        )
        self.jieba = self._load_jieba()
        self.semantic_merge_threshold = max(0.0, min(1.0, float(semantic_merge_threshold)))
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

    def _is_placeholder_key(self, api_key: str) -> bool:
        lowered = api_key.lower()
        placeholders = [
            "your_api_key_here",
            "your_deepseek_api_key_here",
            "dummy",
            "test",
            "example"
        ]
        return any(token in lowered for token in placeholders)

    def _load_jieba(self):
        return jieba_module

    def _estimate_token_count(self, text: str) -> int:
        normalized = (text or "").strip()
        if not normalized:
            return 0
        if self.jieba is not None:
            return len([tok for tok in self.jieba.lcut(normalized) if tok and tok.strip()])
        return len([tok for tok in re.split(r"\s+", normalized) if tok.strip()]) or len(normalized)

    def _build_parent_chunk_text(self, section_title: str, parent_text: str) -> str:
        normalized_parent = (parent_text or "").strip()
        normalized_title = (section_title or "").strip()
        if not normalized_parent:
            return ""
        if not normalized_title or normalized_title == "文档正文":
            return normalized_parent
        if self._estimate_token_count(normalized_title) > self.parent_section_prefix_max_tokens:
            return normalized_parent
        if normalized_parent.startswith(normalized_title):
            return normalized_parent
        return f"{normalized_title}\n{normalized_parent}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _extract_once(self, text: str, source_reference: str) -> str:
        user_prompt = f"""请提取以下文本中的三元组。来源标记请统一使用：'{source_reference}'。

文本内容：
{text}"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API 调用失败：{response.status_code} - {response.text}")

    def _extract_sections(self, text: str) -> list[dict]:
        lines = text.splitlines()
        sections = []
        current_title = "文档正文"
        current_lines = []

        for line in lines:
            matched = self.section_pattern.match(line.strip())
            if matched:
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append({"title": current_title, "text": section_text})
                current_title = f"{matched.group(1)} {matched.group(2)}"
                current_lines = [line.strip()]
                continue
            current_lines.append(line)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({"title": current_title, "text": section_text})

        if sections:
            return self._merge_heading_only_sections(sections)
        cleaned = text.strip()
        return [{"title": "文档正文", "text": cleaned}] if cleaned else []

    def _split_section_number(self, section_title: str) -> list[str]:
        matched = self.section_pattern.match((section_title or "").strip())
        if not matched:
            return []
        return [item for item in matched.group(1).split(".") if item]

    def _is_main_section(self, section_title: str) -> bool:
        parts = self._split_section_number(section_title)
        return bool(parts) and len(parts) <= 2

    def _merge_heading_only_sections(self, sections: list[dict]) -> list[dict]:
        merged_sections = []
        idx = 0
        while idx < len(sections):
            current = sections[idx]
            current_title = (current.get("title") or "").strip()
            current_text = (current.get("text") or "").strip()
            if not current_text:
                idx += 1
                continue

            if current_text == current_title and idx + 1 < len(sections):
                if self._is_main_section(current_title):
                    joined_parts = [current_text]
                    look_ahead = idx + 1
                    while look_ahead < len(sections):
                        candidate = sections[look_ahead]
                        candidate_text = (candidate.get("text") or "").strip()
                        if candidate_text:
                            joined_parts.append(candidate_text)
                        look_ahead += 1
                    merged_sections.append({"title": current_title, "text": "\n".join(joined_parts)})
                    break

                next_section = sections[idx + 1]
                next_text = (next_section.get("text") or "").strip()
                if next_text:
                    merged_sections.append(
                        {
                            "title": current_title,
                            "text": f"{current_text}\n{next_text}".strip()
                        }
                    )
                    idx += 2
                    continue

            merged_sections.append(current)
            idx += 1
        return merged_sections

    def _collect_boundary_positions(self, text: str) -> list[int]:
        boundaries = set()
        cursor = 0
        for paragraph in text.split("\n"):
            cursor += len(paragraph)
            boundaries.add(cursor)
            cursor += 1
        for matched in re.finditer(r"[。！？；]\s*", text):
            boundaries.add(matched.end())
        boundaries.add(len(text))
        return sorted(pos for pos in boundaries if pos > 0)

    def _pick_end(self, boundaries: list[int], start: int, min_end: int, max_end: int) -> int:
        candidates = [pos for pos in boundaries if min_end <= pos <= max_end]
        if candidates:
            return max(candidates)
        larger = [pos for pos in boundaries if pos > max_end]
        if larger:
            return min(larger)
        return min(max_end, len(boundaries) and boundaries[-1] or max_end)

    def _split_parent_chunks_from_section(self, section_text: str) -> list[str]:
        normalized = section_text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.parent_chunk_max:
            return [normalized]

        boundaries = self._collect_boundary_positions(normalized)
        chunks = []
        start = 0
        text_len = len(normalized)

        while start < text_len:
            if text_len - start <= self.parent_chunk_max:
                tail = normalized[start:].strip()
                if tail:
                    chunks.append(tail)
                break

            min_end = min(text_len, start + self.parent_chunk_min)
            max_end = min(text_len, start + self.parent_chunk_max)
            end = self._pick_end(boundaries, start, min_end, max_end)
            if end <= start:
                end = max_end

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)

            next_start = max(0, end - self.parent_chunk_overlap)
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _extract_fault_codes(self, text: str) -> list[str]:
        matched = self.fault_code_pattern.findall(text)
        seen = set()
        result = []
        for code in matched:
            normalized = code.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _extract_components(self, text: str) -> list[str]:
        matched = self.component_pattern.findall(text)
        seen = set()
        result = []
        for item in matched:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _split_sentences(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sentences = []
        if self.jieba is not None:
            buffer = []
            for token in self.jieba.cut(text, cut_all=False):
                if not token:
                    continue
                buffer.append(token)
                if any(mark in token for mark in ["。", "！", "？", "；", "\n"]):
                    sentence = "".join(buffer).strip()
                    if sentence:
                        sentences.append(sentence)
                    buffer = []
            if buffer:
                sentence = "".join(buffer).strip()
                if sentence:
                    sentences.append(sentence)
        else:
            rough = re.split(r"(?<=[。！？；])\s+|[\r\n]+", text)
            sentences = [s.strip() for s in rough if s.strip()]

        cleaned = []
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if len(normalized) >= self.child_min_length:
                cleaned.append(normalized)
        return cleaned

    def build_parent_child_index(self, text: str) -> dict:
        sections = self._extract_sections(text)
        parents = []
        children = []

        for section in sections:
            section_title = section["title"]
            parent_texts = self._split_parent_chunks_from_section(section["text"])
            for idx, raw_parent_text in enumerate(parent_texts, 1):
                parent_text = self._build_parent_chunk_text(section_title, raw_parent_text)
                parent_seed = f"{section_title}::{idx}::{(parent_text or '').strip()}"
                parent_id = f"parent_{hashlib.md5(parent_seed.encode('utf-8')).hexdigest()[:12]}"
                fault_codes = self._extract_fault_codes(parent_text)
                components = self._extract_components(parent_text)
                parent_meta = {
                    "parent_id": parent_id,
                    "text": parent_text,
                    "source_section": section_title,
                    "chunk_index_in_section": idx,
                    "fault_codes": fault_codes,
                    "components": components
                }
                parents.append(parent_meta)

                for sentence in self._split_sentences(parent_text):
                    child_meta = {
                        "text": sentence,
                        "metadata": {
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "is_child": True,
                            "chunk_type": "sentence_child",
                            "source_section": section_title,
                            "fault_codes": fault_codes,
                            "components": components
                        }
                    }
                    children.append(child_meta)

        return {"parents": parents, "children": children}

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

    def _select_child_evidence(self, source_text: str, child_sentences: list[str]) -> str:
        evidence = (source_text or "").strip()
        if not evidence:
            return child_sentences[0] if child_sentences else ""
        best = ""
        best_score = 0.0
        for sentence in child_sentences:
            candidate = (sentence or "").strip()
            if not candidate:
                continue
            if evidence in candidate or candidate in evidence:
                return candidate
            score = SequenceMatcher(None, evidence, candidate).ratio()
            if score > best_score:
                best = candidate
                best_score = score
        if best_score >= 0.45:
            return best
        return evidence

    def _attach_parent_metadata(self, triplets: list[dict], parent_meta: dict, child_chunks: list[dict]) -> list[dict]:
        if not triplets:
            return []
        parent_id = parent_meta.get("parent_id", "")
        source_section = parent_meta.get("source_section", "文档正文")
        fault_codes = list(parent_meta.get("fault_codes") or [])
        components = list(parent_meta.get("components") or [])
        child_texts = [item.get("text", "") for item in child_chunks if isinstance(item, dict)]
        enriched = []
        for triplet in triplets:
            normalized = dict(triplet)
            source_text = normalized.get("source", "")
            metadata = dict(normalized.get("metadata") or {})
            metadata.update({
                "parent_id": parent_id,
                "source_section": source_section,
                "fault_codes": fault_codes,
                "components": components,
                "evidence_text": self._select_child_evidence(source_text, child_texts)
            })
            normalized["metadata"] = metadata
            enriched.append(normalized)
        return enriched

    def _group_children_by_parent(self, child_chunks: list[dict]) -> dict[str, list[dict]]:
        grouped = defaultdict(list)
        for child in child_chunks:
            metadata = child.get("metadata", {}) if isinstance(child, dict) else {}
            parent_id = metadata.get("parent_id", "")
            if parent_id:
                grouped[parent_id].append(child)
        return grouped

    def _build_parent_lookup(self, parents: list[dict]) -> dict[str, dict]:
        lookup = {}
        for parent in parents:
            parent_id = parent.get("parent_id", "")
            if parent_id:
                lookup[parent_id] = parent
        return lookup

    def _ensure_triplet_metadata(self, triplets: list[dict], index_data: dict) -> list[dict]:
        if not triplets:
            return []
        parents = index_data.get("parents", [])
        children = index_data.get("children", [])
        if not parents:
            return triplets
        parent_lookup = self._build_parent_lookup(parents)
        parent_children = self._group_children_by_parent(children)
        fallback_parent = parents[0]
        ensured = []
        for triplet in triplets:
            metadata = dict(triplet.get("metadata") or {})
            parent_id = (metadata.get("parent_id") or "").strip()
            if parent_id and parent_id in parent_lookup:
                parent = parent_lookup[parent_id]
                ensured.extend(self._attach_parent_metadata([triplet], parent, parent_children.get(parent_id, [])))
                continue
            source = (triplet.get("source") or "").strip()
            subject = (triplet.get("subject_name") or "").strip()
            obj = (triplet.get("object_name") or "").strip()
            best_parent = fallback_parent
            best_score = -1.0
            for parent in parents:
                score = 0.0
                parent_text = parent.get("text", "")
                if source and source in parent_text:
                    score += 0.65
                if subject and subject in parent_text:
                    score += 0.2
                if obj and obj in parent_text:
                    score += 0.2
                if score > best_score:
                    best_parent = parent
                    best_score = score
            selected_parent_id = best_parent.get("parent_id", "")
            ensured.extend(
                self._attach_parent_metadata([triplet], best_parent, parent_children.get(selected_parent_id, []))
            )
        return ensured

    def _enforce_joint_logic(self, triplets: list[dict]) -> list[dict]:
        if not triplets:
            return []
        grouped = defaultdict(list)
        for t in triplets:
            if (t.get("relation") or "").strip() != "jointly_resultsIn":
                continue
            key = ((t.get("object_name") or "").strip(), (t.get("source") or "").strip())
            grouped[key].append(t)

        rewritten = []
        consumed = set()
        for t in triplets:
            key = (
                (t.get("subject_name") or "").strip(),
                (t.get("subject_type") or "").strip(),
                (t.get("relation") or "").strip(),
                (t.get("object_name") or "").strip(),
                (t.get("object_type") or "").strip(),
                (t.get("source") or "").strip(),
            )
            if key in consumed:
                continue
            relation = (t.get("relation") or "").strip()
            if relation != "jointly_resultsIn":
                rewritten.append(t)
                continue

            group_key = ((t.get("object_name") or "").strip(), (t.get("source") or "").strip())
            group = grouped.get(group_key, [])
            subjects = []
            for item in group:
                subject_name = (item.get("subject_name") or "").strip()
                if subject_name and subject_name not in subjects:
                    subjects.append(subject_name)

            if len(group) <= 2 or len(subjects) <= 2:
                for item in group:
                    item_key = (
                        (item.get("subject_name") or "").strip(),
                        (item.get("subject_type") or "").strip(),
                        (item.get("relation") or "").strip(),
                        (item.get("object_name") or "").strip(),
                        (item.get("object_type") or "").strip(),
                        (item.get("source") or "").strip(),
                    )
                    if item_key not in consumed:
                        rewritten.append(item)
                        consumed.add(item_key)
                continue

            intermediate_name = f"{'与'.join(subjects[:2])}组合触发"
            max_conf = max(float(item.get("confidence", 0.9) or 0.9) for item in group)
            for item in group:
                item_key = (
                    (item.get("subject_name") or "").strip(),
                    (item.get("subject_type") or "").strip(),
                    (item.get("relation") or "").strip(),
                    (item.get("object_name") or "").strip(),
                    (item.get("object_type") or "").strip(),
                    (item.get("source") or "").strip(),
                )
                if item_key in consumed:
                    continue
                consumed.add(item_key)
                rewritten.append({
                    "subject_name": item.get("subject_name"),
                    "subject_type": item.get("subject_type") or "BasicEvent",
                    "relation": "jointly_resultsIn",
                    "object_name": intermediate_name,
                    "object_type": "IntermediateEvent",
                    "confidence": item.get("confidence", max_conf),
                    "source": item.get("source"),
                    "metadata": dict(item.get("metadata") or {})
                })
            rewritten.append({
                "subject_name": intermediate_name,
                "subject_type": "IntermediateEvent",
                "relation": "resultsIn",
                "object_name": t.get("object_name"),
                "object_type": t.get("object_type") or "IntermediateEvent",
                "confidence": max_conf,
                "source": (t.get("source") or "").strip() or f"{intermediate_name}导致{t.get('object_name', '')}",
                "metadata": dict(group[0].get("metadata") or {})
            })
        return self._deduplicate_triplets(rewritten)

    def _repair_event_types(self, triplets: list[dict]) -> list[dict]:
        if not triplets:
            return []
        inbound = defaultdict(int)
        outbound = defaultdict(int)
        for t in triplets:
            subject = (t.get("subject_name") or "").strip()
            obj = (t.get("object_name") or "").strip()
            if subject:
                outbound[subject] += 1
            if obj:
                inbound[obj] += 1
        top_nodes = {name for name in inbound if outbound[name] == 0}
        repaired = []
        for t in triplets:
            normalized = dict(t)
            subject = (normalized.get("subject_name") or "").strip()
            obj = (normalized.get("object_name") or "").strip()
            if subject and inbound[subject] > 0 and outbound[subject] > 0:
                normalized["subject_type"] = "IntermediateEvent"
            if obj and inbound[obj] > 0 and outbound[obj] > 0:
                normalized["object_type"] = "IntermediateEvent"
            if obj in top_nodes:
                normalized["object_type"] = "TopEvent"
            repaired.append(normalized)
        return repaired

    def _node_id(self, node_name: str, node_type: str) -> str:
        base = f"{(node_type or '').strip()}::{(node_name or '').strip()}"
        return f"node_{hashlib.md5(base.encode('utf-8')).hexdigest()[:12]}"

    def _build_lightrag_graph(self, triplets: list[dict], source_reference: str) -> dict:
        node_map = {}
        edges = []
        for t in triplets:
            subject_name = (t.get("subject_name") or "").strip()
            object_name = (t.get("object_name") or "").strip()
            relation = (t.get("relation") or "").strip()
            subject_type = (t.get("subject_type") or "").strip() or "IntermediateEvent"
            object_type = (t.get("object_type") or "").strip() or "IntermediateEvent"
            metadata = dict(t.get("metadata") or {})
            if not subject_name or not object_name or not relation:
                continue

            for node_name, node_type in [(subject_name, subject_type), (object_name, object_type)]:
                node_key = (node_name, node_type)
                if node_key not in node_map:
                    node_map[node_key] = {
                        "id": self._node_id(node_name, node_type),
                        "name": node_name,
                        "event_type": node_type,
                        "metadata": {
                            "parent_ids": [],
                            "source_sections": [],
                            "fault_codes": [],
                            "components": []
                        }
                    }
                node_meta = node_map[node_key]["metadata"]
                parent_id = metadata.get("parent_id")
                if parent_id and parent_id not in node_meta["parent_ids"]:
                    node_meta["parent_ids"].append(parent_id)
                source_section = metadata.get("source_section")
                if source_section and source_section not in node_meta["source_sections"]:
                    node_meta["source_sections"].append(source_section)
                for code in metadata.get("fault_codes", []) or []:
                    if code not in node_meta["fault_codes"]:
                        node_meta["fault_codes"].append(code)
                for component in metadata.get("components", []) or []:
                    if component not in node_meta["components"]:
                        node_meta["components"].append(component)

            edges.append({
                "source": self._node_id(subject_name, subject_type),
                "target": self._node_id(object_name, object_type),
                "relation": relation,
                "confidence": t.get("confidence"),
                "source_text": t.get("source", ""),
                "metadata": metadata
            })

        return {
            "graph_format": "LightRAG",
            "source_reference": source_reference,
            "nodes": list(node_map.values()),
            "edges": edges
        }

    def _build_output_payload(self, triplets: list[dict], source_reference: str, index_data: dict) -> dict:
        normalized_triplets = self._deduplicate_triplets(triplets)
        return {
            "triplets": normalized_triplets,
            "parent_child_index": index_data,
            "lightrag_graph": self._build_lightrag_graph(normalized_triplets, source_reference)
        }

    def _deduplicate_triplets(self, triplets: list[dict]) -> list[dict]:
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

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"\s+", "", (name or "").strip().lower())

    def _name_match(self, candidate: str, target: str) -> bool:
        c = self._normalize_name(candidate)
        t = self._normalize_name(target)
        if not c or not t:
            return False
        return self._semantic_name_similarity(candidate, target) >= self.semantic_merge_threshold

    def _normalize_core_name(self, name: str) -> str:
        normalized = self._normalize_name(name)
        if not normalized:
            return ""
        core = re.sub(
            r"(电气|驱动|系统|组件|模块|单元|设备|内部|相关|类|级)?(故障|异常|问题|错误|触发|告警|失效|失常|失败|波动)$",
            "",
            normalized
        )
        return core or normalized

    def _simplify_name_for_match(self, name: str) -> str:
        normalized = self._normalize_name(name)
        if not normalized:
            return ""
        simplified = re.sub(r"(电气|驱动|内部|外部|相关|类型|类|级|模块化)", "", normalized)
        return simplified or normalized

    def _tokenize_for_similarity(self, name: str) -> list[str]:
        core = self._normalize_core_name(name)
        if not core:
            return []
        if self.jieba:
            tokens = [tok.strip() for tok in self.jieba.lcut(core) if tok and tok.strip()]
            return [t for t in tokens if t not in {"的", "了", "和", "与", "及"}]
        if len(core) <= 2:
            return list(core)
        return [core[idx:idx + 2] for idx in range(len(core) - 1)]

    def _semantic_name_similarity(self, left: str, right: str) -> float:
        l_norm = self._normalize_name(left)
        r_norm = self._normalize_name(right)
        if not l_norm or not r_norm:
            return 0.0
        if l_norm == r_norm:
            return 1.0
        if l_norm in r_norm or r_norm in l_norm:
            short_len = min(len(l_norm), len(r_norm))
            if short_len >= 3:
                return 0.96
        l_simple = self._simplify_name_for_match(left)
        r_simple = self._simplify_name_for_match(right)
        if l_simple == r_simple:
            return 0.95
        if l_simple and r_simple and (l_simple in r_simple or r_simple in l_simple):
            short_len = min(len(l_simple), len(r_simple))
            if short_len >= 3:
                return 0.93

        l_core = self._normalize_core_name(left)
        r_core = self._normalize_core_name(right)
        seq_score = SequenceMatcher(None, l_core or l_norm, r_core or r_norm).ratio()
        full_seq_score = SequenceMatcher(None, l_norm, r_norm).ratio()
        seq_score = max(seq_score, full_seq_score)

        left_tokens = set(self._tokenize_for_similarity(left))
        right_tokens = set(self._tokenize_for_similarity(right))
        if left_tokens and right_tokens:
            overlap = len(left_tokens & right_tokens)
            union = len(left_tokens | right_tokens)
            token_score = overlap / union if union else 0.0
        else:
            token_score = 0.0
        return 0.7 * seq_score + 0.3 * token_score

    def _select_better_name(self, current: str, candidate: str) -> str:
        if not current:
            return candidate
        current_core = self._normalize_core_name(current)
        candidate_core = self._normalize_core_name(candidate)
        if len(candidate_core) > len(current_core):
            return candidate
        if len(candidate_core) == len(current_core) and len(candidate) > len(current):
            return candidate
        return current

    def _find_semantic_canonical(self, canonical_names: list[str], name: str, threshold: float) -> str:
        if not canonical_names:
            return ""
        best_name = ""
        best_score = 0.0
        for current in canonical_names:
            score = self._semantic_name_similarity(current, name)
            if score > best_score:
                best_score = score
                best_name = current
        if best_score >= threshold:
            return best_name
        return ""

    def _merge_semantically_similar_triplets(self, triplets: list[dict], threshold: float) -> list[dict]:
        if not triplets:
            return []

        bounded_threshold = max(0.0, min(1.0, float(threshold)))
        canonical_by_type = {
            "BasicEvent": [],
            "IntermediateEvent": [],
            "TopEvent": [],
        }
        alias_map = {}

        def resolve_name(raw_name: str, event_type: str) -> str:
            name = (raw_name or "").strip()
            node_type = (event_type or "").strip()
            if not name or node_type not in canonical_by_type:
                return name
            alias_key = (node_type, name)
            if alias_key in alias_map:
                return alias_map[alias_key]
            canonical_list = canonical_by_type[node_type]
            matched = self._find_semantic_canonical(canonical_list, name, bounded_threshold)
            if not matched:
                canonical_list.append(name)
                alias_map[alias_key] = name
                return name
            better = self._select_better_name(matched, name)
            if better != matched:
                canonical_list[canonical_list.index(matched)] = better
                for key, value in list(alias_map.items()):
                    if key[0] == node_type and value == matched:
                        alias_map[key] = better
                alias_map[alias_key] = better
                return better
            alias_map[alias_key] = matched
            return matched

        merged = []
        for triplet in triplets:
            subject_type = (triplet.get("subject_type") or "").strip()
            object_type = (triplet.get("object_type") or "").strip()
            merged_triplet = dict(triplet)
            merged_triplet["subject_name"] = resolve_name(triplet.get("subject_name"), subject_type)
            merged_triplet["object_name"] = resolve_name(triplet.get("object_name"), object_type)
            merged.append(merged_triplet)

        return self._deduplicate_triplets(merged)

    def _canonicalize_top_event_name(self, triplets: list[dict], top_event: str) -> list[dict]:
        canonical = (top_event or "").strip()
        if not canonical:
            return triplets
        adjusted = []
        for triplet in triplets:
            normalized = dict(triplet)
            subject_name = (normalized.get("subject_name") or "").strip()
            object_name = (normalized.get("object_name") or "").strip()
            if subject_name and self._name_match(subject_name, canonical):
                normalized["subject_name"] = canonical
                normalized["subject_type"] = "TopEvent"
            if object_name and self._name_match(object_name, canonical):
                normalized["object_name"] = canonical
                normalized["object_type"] = "TopEvent"
            adjusted.append(normalized)
        return self._deduplicate_triplets(adjusted)

    def _collect_top_event_chain(self, triplets: list[dict], top_event: str) -> list[dict]:
        normalized_top = self._normalize_name(top_event)
        if not normalized_top:
            return triplets

        causal_forward = {"resultsIn", "jointly_resultsIn", "relatedTo"}
        seeds = {top_event.strip()}
        for t in triplets:
            subject = (t.get("subject_name") or "").strip()
            obj = (t.get("object_name") or "").strip()
            if self._name_match(subject, top_event) or self._name_match(obj, top_event):
                seeds.add(subject)
                seeds.add(obj)

        queue = [s for s in seeds if s]
        visited_nodes = {self._normalize_name(s) for s in queue}
        collected = []
        collected_keys = set()

        while queue:
            current = queue.pop(0)
            current_key = self._normalize_name(current)
            if not current_key:
                continue

            for t in triplets:
                relation = (t.get("relation") or "").strip()
                subject = (t.get("subject_name") or "").strip()
                obj = (t.get("object_name") or "").strip()
                key = (
                    subject,
                    (t.get("subject_type") or "").strip(),
                    relation,
                    obj,
                    (t.get("object_type") or "").strip(),
                )

                next_node = None
                if relation in causal_forward and self._normalize_name(obj) == current_key:
                    next_node = subject
                elif relation == "causedBy" and self._normalize_name(subject) == current_key:
                    next_node = obj

                if not next_node:
                    continue

                if key not in collected_keys:
                    collected_keys.add(key)
                    collected.append(t)

                next_key = self._normalize_name(next_node)
                if next_key and next_key not in visited_nodes:
                    visited_nodes.add(next_key)
                    queue.append(next_node)

        return self._canonicalize_top_event_name(self._deduplicate_triplets(collected), top_event)

    def _merge_triplets_globally(self, text: str, source_reference: str, triplets: list[dict], top_event: Optional[str] = None) -> str:
        """第二阶段全局合并：跨分块重建链路、补全中间层并统一命名。"""
        normalized_top = (top_event or "").strip()
        if normalized_top:
            merge_prompt = (
                "你将看到同一文档不同片段抽取出的故障三元组候选。"
                f"请仅围绕顶事件“{normalized_top}”进行全局融合，重点执行："
                "1) 仅保留可回溯到该顶事件的因果链；"
                "2) 跨上下文拼接完整链路并补全可证据支持的 IntermediateEvent；"
                "3) 去重并统一同义命名（不要误合并上下游事件）；"
                "4) 保留 OR/AND 逻辑正确性。"
                "只输出 JSON，格式为 {\"triplets\": [...]}。"
            )
            text_window = text[:2500]
        else:
            merge_prompt = (
                "你将看到同一文档不同片段抽取出的故障三元组候选。"
                "请进行全局融合，重点执行："
                "1) 跨上下文拼接完整因果链；"
                "2) 补全可证据支持的中间层 IntermediateEvent；"
                "3) 去重并统一同义命名（但不要误合并上下游事件）；"
                "4) 保留 OR/AND 逻辑正确性。"
                "只输出 JSON，格式为 {\"triplets\": [...]}。"
            )
            text_window = text[:6000]

        user_prompt = f"""来源标记统一使用：'{source_reference}'。

【原始文本（用于全局理解）】
{text_window}

【分块候选三元组】
{json.dumps(triplets, ensure_ascii=False, indent=2)}
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": merge_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        raise Exception(f"全局融合失败：{response.status_code} - {response.text}")

    def _append_debug_event(self, debug_log_path: Optional[str], event: str, payload: dict) -> None:
        target = (debug_log_path or "").strip()
        if not target:
            return
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            record = {"ts": time.time(), "event": event, "payload": payload}
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except Exception:
            return

    def _filter_index_by_parent_ids(self, index_data: dict, candidate_parent_ids: list[str]) -> tuple[list[dict], list[dict]]:
        normalized_ids = {str(item).strip() for item in (candidate_parent_ids or []) if str(item).strip()}
        parents = list(index_data.get("parents", []) or [])
        children = list(index_data.get("children", []) or [])
        if not normalized_ids:
            return [], []
        filtered_parents = [p for p in parents if str(p.get("parent_id", "") or "").strip() in normalized_ids]
        filtered_children = []
        for child in children:
            if not isinstance(child, dict):
                continue
            metadata = child.get("metadata", {}) or {}
            parent_id = str(metadata.get("parent_id", "") or "").strip()
            if parent_id in normalized_ids:
                filtered_children.append(child)
        return filtered_parents, filtered_children

    def _evidence_in_child_sentences(self, evidence_text: str, child_sentences: list[str]) -> bool:
        evidence = (evidence_text or "").strip()
        if not evidence:
            return False
        for sentence in child_sentences:
            candidate = (sentence or "").strip()
            if not candidate:
                continue
            if evidence == candidate or evidence in candidate or candidate in evidence:
                return True
            if SequenceMatcher(None, evidence, candidate).ratio() >= 0.82:
                return True
        return False

    def _triplet_has_candidate_evidence(self, triplet: dict, candidate_children_map: dict[str, list[str]]) -> bool:
        metadata = dict(triplet.get("metadata") or {})
        parent_id = str(metadata.get("parent_id", "") or "").strip()
        if not parent_id:
            return False
        child_sentences = candidate_children_map.get(parent_id, [])
        if not child_sentences:
            return False
        evidence_text = metadata.get("evidence_text", "") or triplet.get("source", "")
        return self._evidence_in_child_sentences(str(evidence_text), child_sentences)

    def _triplet_key(self, triplet: dict) -> tuple[str, str, str, str, str]:
        return (
            (triplet.get("subject_name") or "").strip(),
            (triplet.get("subject_type") or "").strip(),
            (triplet.get("relation") or "").strip(),
            (triplet.get("object_name") or "").strip(),
            (triplet.get("object_type") or "").strip(),
        )

    def refine_with_candidates(
        self,
        text: str,
        source_reference: str,
        index_data: dict,
        candidate_parent_ids: list[str],
        top_event: Optional[str] = None
    ) -> list[dict]:
        normalized_top = (top_event or "").strip()
        filtered_parents, filtered_children = self._filter_index_by_parent_ids(index_data or {}, candidate_parent_ids or [])
        if not filtered_parents:
            return []

        parent_lookup = self._build_parent_lookup(filtered_parents)
        parent_children = self._group_children_by_parent(filtered_children)
        refined_triplets: list[dict] = []

        for idx, parent in enumerate(filtered_parents, 1):
            parent_id = str(parent.get("parent_id", "") or "")
            section = parent.get("source_section", "文档正文")
            chunk_source = f"{source_reference} | refine_parent {idx}/{len(filtered_parents)} | {section} | {parent_id}"
            raw = self._extract_once(parent.get("text", ""), chunk_source)
            parsed = self._safe_parse_triplets(raw)
            parsed = self._attach_parent_metadata(
                parsed,
                parent_lookup.get(parent_id, parent),
                parent_children.get(parent_id, [])
            )
            refined_triplets.extend(parsed)

        refined_triplets = self._deduplicate_triplets(refined_triplets)
        refined_triplets = self._merge_semantically_similar_triplets(refined_triplets, self.semantic_merge_threshold)
        candidate_children_map = {}
        for parent_id, items in parent_children.items():
            candidate_children_map[parent_id] = [str(item.get("text", "") or "") for item in items if isinstance(item, dict)]
        refined_triplets = [t for t in refined_triplets if self._triplet_has_candidate_evidence(t, candidate_children_map)]
        refined_triplets = self._enforce_joint_logic(refined_triplets)
        refined_triplets = self._repair_event_types(refined_triplets)
        if normalized_top:
            scoped = self._collect_top_event_chain(refined_triplets, normalized_top)
            if scoped:
                refined_triplets = scoped
        return self._deduplicate_triplets(refined_triplets)

    def merge_with_refinement(
        self,
        base_triplets: list[dict],
        refined_triplets: list[dict],
        index_data: dict,
        candidate_parent_ids: list[str],
        policy: str = "balanced"
    ) -> tuple[list[dict], dict[str, Any]]:
        normalized_policy = (policy or "balanced").strip().lower()
        if normalized_policy not in {"strict", "balanced"}:
            normalized_policy = "balanced"

        candidate_set = {str(item).strip() for item in (candidate_parent_ids or []) if str(item).strip()}
        all_children = list(index_data.get("children", []) or [])
        candidate_children_map: dict[str, list[str]] = {}
        for child in all_children:
            if not isinstance(child, dict):
                continue
            metadata = child.get("metadata", {}) or {}
            parent_id = str(metadata.get("parent_id", "") or "").strip()
            if not parent_id or parent_id not in candidate_set:
                continue
            candidate_children_map.setdefault(parent_id, []).append(str(child.get("text", "") or ""))

        def is_graph_hit(triplet: dict) -> bool:
            parent_id = str(dict(triplet.get("metadata") or {}).get("parent_id", "") or "").strip()
            return bool(parent_id and parent_id in candidate_set)

        def has_evidence(triplet: dict) -> bool:
            return self._triplet_has_candidate_evidence(triplet, candidate_children_map)

        normalized_base = self._ensure_triplet_metadata(base_triplets, index_data)
        normalized_refined = self._ensure_triplet_metadata(refined_triplets, index_data)
        refined_strong = [t for t in normalized_refined if is_graph_hit(t) and has_evidence(t)]

        if normalized_policy == "strict":
            strict_candidates = [t for t in normalized_base if is_graph_hit(t) and has_evidence(t)]
            merged_candidates = strict_candidates + refined_strong
        else:
            balanced_base = []
            for triplet in normalized_base:
                confidence = float(triplet.get("confidence", 0.0) or 0.0)
                if is_graph_hit(triplet) or confidence >= 0.9:
                    balanced_base.append(triplet)
            merged_candidates = balanced_base + refined_strong

        by_key = {}
        for item in merged_candidates:
            key = self._triplet_key(item)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = item
                continue
            current_conf = float(existing.get("confidence", 0.0) or 0.0)
            new_conf = float(item.get("confidence", 0.0) or 0.0)
            if new_conf > current_conf:
                by_key[key] = item

        merged = list(by_key.values())
        merged = self._merge_semantically_similar_triplets(merged, self.semantic_merge_threshold)
        merged = self._enforce_joint_logic(merged)
        merged = self._repair_event_types(merged)
        merged = self._deduplicate_triplets(merged)

        stats = {
            "policy": normalized_policy,
            "candidate_parent_count": len(candidate_set),
            "base_triplet_count": len(base_triplets or []),
            "refined_triplet_count": len(refined_triplets or []),
            "refined_strong_count": len(refined_strong),
            "merged_triplet_count": len(merged),
        }
        return merged, stats

    def extract(
        self,
        text: str,
        source_reference: str,
        top_event: Optional[str] = None,
        debug_log_path: Optional[str] = None,
    ) -> str:
        normalized_top = (top_event or "").strip()
        debug_target = (debug_log_path or "").strip() or None
        if debug_target:
            try:
                os.makedirs(os.path.dirname(debug_target) or ".", exist_ok=True)
                with open(debug_target, "w", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": time.time(),
                                "event": "start",
                                "payload": {
                                    "source_reference": source_reference,
                                    "top_event": normalized_top or None,
                                    "semantic_merge_threshold": self.semantic_merge_threshold,
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()
            except Exception:
                debug_target = None
        index_data = self.build_parent_child_index(text)
        parent_chunks = index_data.get("parents", [])
        child_chunks = index_data.get("children", [])
        parent_lookup = self._build_parent_lookup(parent_chunks)
        parent_children = self._group_children_by_parent(child_chunks)
        section_count = len({p.get("source_section", "文档正文") for p in parent_chunks})
        print(
            f"[Index] source={source_reference} sections={section_count} "
            f"parents={len(parent_chunks)} children={len(child_chunks)}"
        )
        self._append_debug_event(
            debug_target,
            "index",
            {
                "source_reference": source_reference,
                "sections": section_count,
                "parents": len(parent_chunks),
                "children": len(child_chunks),
            },
        )
        if not parent_chunks:
            return json.dumps(self._build_output_payload([], source_reference, index_data), ensure_ascii=False)

        if len(parent_chunks) == 1 and len(text) <= self.long_text_threshold:
            print(f"[Index] 单父块直抽模式：parent_id={parent_chunks[0].get('parent_id', '')}")
            raw = self._extract_once(parent_chunks[0]["text"], source_reference)
            parsed = self._deduplicate_triplets(self._safe_parse_triplets(raw))
            parsed = self._merge_semantically_similar_triplets(parsed, self.semantic_merge_threshold)
            single_parent = parent_chunks[0]
            parsed = self._attach_parent_metadata(
                parsed,
                single_parent,
                parent_children.get(single_parent.get("parent_id", ""), [])
            )
            parsed = self._enforce_joint_logic(parsed)
            parsed = self._repair_event_types(parsed)
            print(f"[Extract] 单父块抽取得到候选三元组：{len(parsed)}")
            self._append_debug_event(
                debug_target,
                "pre_merge_triplets",
                {"mode": "single_parent_direct", "count": len(parsed), "triplets": parsed},
            )
            if not normalized_top:
                self._append_debug_event(
                    debug_target,
                    "final_triplets",
                    {
                        "mode": "single_parent_direct",
                        "count": len(parsed),
                        "used_global_merge": False,
                        "triplets": parsed,
                    },
                )
                return json.dumps(self._build_output_payload(parsed, source_reference, index_data), ensure_ascii=False)
            scoped = self._collect_top_event_chain(parsed, normalized_top)
            print(
                f"[TopEvent] 顶事件过滤：top_event={normalized_top} "
                f"before={len(parsed)} after={len(scoped) if scoped else len(parsed)}"
            )
            self._append_debug_event(
                debug_target,
                "top_event_filter",
                {
                    "top_event": normalized_top,
                    "before": len(parsed),
                    "after": len(scoped) if scoped else len(parsed),
                },
            )
            if not scoped:
                scoped = parsed
            scoped = self._ensure_triplet_metadata(scoped, index_data)
            scoped = self._enforce_joint_logic(scoped)
            scoped = self._repair_event_types(scoped)
            self._append_debug_event(
                debug_target,
                "final_triplets",
                {
                    "mode": "single_parent_direct",
                    "count": len(scoped),
                    "used_global_merge": False,
                    "triplets": scoped,
                },
            )
            return json.dumps(self._build_output_payload(scoped, source_reference, index_data), ensure_ascii=False)

        all_triplets = []
        for idx, parent in enumerate(parent_chunks, 1):
            section = parent.get("source_section", "文档正文")
            parent_id = parent.get("parent_id", "")
            chunk_source = f"{source_reference} | parent {idx}/{len(parent_chunks)} | {section} | {parent_id}"
            raw = self._extract_once(parent["text"], chunk_source)
            parsed = self._safe_parse_triplets(raw)
            parsed = self._attach_parent_metadata(
                parsed,
                parent_lookup.get(parent_id, parent),
                parent_children.get(parent_id, [])
            )
            all_triplets.extend(parsed)
            print(
                f"[Extract] parent {idx}/{len(parent_chunks)} "
                f"id={parent_id} section={section} chunk_triplets={len(parsed)}"
            )
            self._append_debug_event(
                debug_target,
                "chunk_triplets",
                {
                    "parent_index": idx,
                    "parent_total": len(parent_chunks),
                    "parent_id": parent_id,
                    "section": section,
                    "chunk_source": chunk_source,
                    "count": len(parsed),
                    "triplets": parsed,
                },
            )

        all_triplets = self._deduplicate_triplets(all_triplets)
        all_triplets = self._merge_semantically_similar_triplets(all_triplets, self.semantic_merge_threshold)
        all_triplets = self._enforce_joint_logic(all_triplets)
        all_triplets = self._repair_event_types(all_triplets)
        print(f"[Extract] 分块去重后候选三元组：{len(all_triplets)}")
        self._append_debug_event(
            debug_target,
            "pre_merge_triplets",
            {"mode": "chunk_dedup_semantic", "count": len(all_triplets), "triplets": all_triplets},
        )

        if not all_triplets:
            return json.dumps(self._build_output_payload([], source_reference, index_data), ensure_ascii=False)

        scoped_triplets = all_triplets
        if normalized_top:
            filtered_triplets = self._collect_top_event_chain(all_triplets, normalized_top)
            if filtered_triplets:
                scoped_triplets = filtered_triplets
            print(
                f"[TopEvent] 顶事件过滤：top_event={normalized_top} "
                f"before={len(all_triplets)} after={len(scoped_triplets)}"
            )
            self._append_debug_event(
                debug_target,
                "top_event_filter",
                {"top_event": normalized_top, "before": len(all_triplets), "after": len(scoped_triplets)},
            )

        try:
            print(f"[Merge] 全局融合输入三元组：{len(scoped_triplets)}")
            self._append_debug_event(
                debug_target,
                "global_merge_input",
                {"count": len(scoped_triplets), "triplets": scoped_triplets},
            )
            merged_raw = self._merge_triplets_globally(
                text,
                source_reference,
                scoped_triplets,
                top_event=normalized_top
            )
            merged_triplets = self._safe_parse_triplets(merged_raw)
            merged_triplets = self._merge_semantically_similar_triplets(merged_triplets, self.semantic_merge_threshold)
            if normalized_top:
                merged_triplets = self._canonicalize_top_event_name(merged_triplets, normalized_top)
            merged_triplets = self._ensure_triplet_metadata(merged_triplets, index_data)
            merged_triplets = self._enforce_joint_logic(merged_triplets)
            merged_triplets = self._repair_event_types(merged_triplets)
            print(f"[Merge] 全局融合输出三元组：{len(merged_triplets)}")
            self._append_debug_event(
                debug_target,
                "post_merge_triplets",
                {"count": len(merged_triplets), "triplets": merged_triplets},
            )
        except Exception as e:
            print(f"[Extractor] 全局融合失败（{type(e).__name__}: {e}），降级使用分块去重结果")
            merged_triplets = []
            self._append_debug_event(
                debug_target,
                "global_merge_error",
                {"error_type": type(e).__name__, "error": str(e)},
            )

        if not merged_triplets:
            scoped_triplets = self._ensure_triplet_metadata(scoped_triplets, index_data)
            scoped_triplets = self._enforce_joint_logic(scoped_triplets)
            scoped_triplets = self._repair_event_types(scoped_triplets)
            self._append_debug_event(
                debug_target,
                "final_triplets",
                {
                    "mode": "fallback_pre_merge",
                    "count": len(scoped_triplets),
                    "used_global_merge": False,
                    "triplets": scoped_triplets,
                },
            )
            return json.dumps(self._build_output_payload(scoped_triplets, source_reference, index_data), ensure_ascii=False)

        merged_triplets = self._ensure_triplet_metadata(merged_triplets, index_data)
        merged_triplets = self._enforce_joint_logic(merged_triplets)
        merged_triplets = self._repair_event_types(merged_triplets)
        self._append_debug_event(
            debug_target,
            "final_triplets",
            {"mode": "global_merge", "count": len(merged_triplets), "used_global_merge": True, "triplets": merged_triplets},
        )
        return json.dumps(self._build_output_payload(merged_triplets, source_reference, index_data), ensure_ascii=False)
