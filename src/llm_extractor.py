import os
import dashscope
from tenacity import retry, stop_after_attempt, wait_exponential

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


class QwenExtractor:
    def __init__(self):
        self.model = "qwen-max"
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
        """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract(self, text: str, source_reference: str) -> str:
        # 在 User Prompt 中明确强调来源标记，防止模型瞎编 source
        user_prompt = f"请提取以下文本中的三元组。来源标记请统一使用：'{source_reference}'。\n文本内容：\n{text}"

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