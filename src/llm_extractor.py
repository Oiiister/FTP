import os
import dashscope
from tenacity import retry, stop_after_attempt, wait_exponential

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


class QwenExtractor:
    def __init__(self):
        self.model = "qwen-max"
        # 深度优化的 System Prompt
        self.system_prompt = """
        你是一个精通故障树分析（FTA）的专家。你的任务是从技术文本中抽取出严谨的故障三元组。

        ### 1. 核心任务目标
        - **全面性**：不要只提取最终结果。如果文本描述了“原因A -> 过程B -> 现象C”，必须提取出两条记录。
        - **逻辑准确性**：必须严格区分“独立原因（OR逻辑）”和“共同原因（AND逻辑）”。

        ### 2. 参数定义规范
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
        - **source**: 必须直接使用用户提供的【来源标记】。

        ### 3. 复杂案例分析 (Few-Shot)
        【输入文本】: 来源标记为【手册V2】。当液压油污染且滤芯堵塞时，会共同引发油压异常。油压异常若持续存在，会导致执行机构动作缓慢，最终引发系统停机。
        【期望JSON】:
        {
          "triplets": [
            {"subject_name": "液压油污染", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "油压异常", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "手册V2"},
            {"subject_name": "滤芯堵塞", "subject_type": "BasicEvent", "relation": "jointly_resultsIn", "object_name": "油压异常", "object_type": "IntermediateEvent", "confidence": 1.0, "source": "手册V2"},
            {"subject_name": "油压异常", "subject_type": "IntermediateEvent", "relation": "resultsIn", "object_name": "执行机构动作缓慢", "object_type": "IntermediateEvent", "confidence": 0.95, "source": "手册V2"},
            {"subject_name": "执行机构动作缓慢", "subject_type": "IntermediateEvent", "relation": "resultsIn", "object_name": "系统停机", "object_type": "TopEvent", "confidence": 0.98, "source": "手册V2"}
          ]
        }

        ### 4. 强制约束
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