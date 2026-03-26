import os
import json
import dashscope
from tenacity import retry, stop_after_attempt, wait_exponential


class QwenEvaluator:
    def __init__(self):
        self.model = "qwen-max"  # 建议使用逻辑能力最强的模型
        self.system_prompt = """
        你是一个精通故障树分析（FTA）的逻辑审计专家。你的任务是对比【原始文本】和【提取出的三元组】，找出逻辑矛盾或缺失。

        ### 审计核心维度：
        1. **与门逻辑 (AND Gate)**：只有在原文明确表达“共同满足、缺一不可、同时发生才导致结果”时，才能使用 `jointly_resultsIn`。且同一个 Object 必须对应至少两个不同的 Subject。
        2. **逻辑断层 (Causal Gap)**：检查是否存在巨大的跳跃。例如“基础事件”直接导致“顶层事件”而没有中间过程。
        3. **中间节点完整性**：IntermediateEvent 是否既有来源又有去向？
        4. **幻觉检查**：三元组中的实体是否在原文中真实存在。
        5. **命名一致性 (Naming Consistency)**：同一个事件是否被写成多个名称，例如“空压机启动失败/空压机无法启动”“进气阀发生卡滞/进气阀卡滞”“最小压力阀出现故障/最小压力阀故障”。若出现这种情况，应判定为缺陷。

        ### OR/AND 审计边界：
        - 如果多个原因分别都能独立导致同一个结果，这是 OR 逻辑，应接受多条 `resultsIn`，不得误判为必须使用 `jointly_resultsIn`。
        - 如果原文只是枚举多个可能故障来源、多个系统分支、多个并列原因，这通常是 OR 逻辑，不是 AND 逻辑。
        - 只有当原文明确说明“必须同时满足”“共同造成”“缺少任何一个都不成立”时，才能要求 `jointly_resultsIn`。
        - 不要因为多个原因最终指向同一个 TopEvent，就自动认定它们是 AND。
        - 对故障树顶事件来说，多个中间事件或基础事件并列汇入同一顶事件，默认优先按 OR 理解，除非原文明确写出联合触发条件。

        ### 命名一致性审计要求：
                - 只有同时满足以下条件，才可判定为“同一事件命名冲突”：
                    1) 事件层级一致（BasicEvent/IntermediateEvent/TopEvent 相同）；
                    2) 语义可互换（只是措辞差异，如“出现故障/故障”“发生卡滞/卡滞”）；
                    3) 原文存在可支持的同义证据（如又称、即、也叫，或明显同义表达）。
                - 以下情况【禁止】判定为同一事件：
                    1) 存在明确因果链条（A 导致 B），A 和 B 不同事件；
                    2) 粒度不同（部件级故障 vs 系统级故障）；
                    3) 上下游关系（如“进气阀卡滞”与“进气系统堵塞”）；
                    4) 保护动作与本体故障（如“保护动作”与“设备故障”）混为同一实体。
                - 若仅“可能是同义”但证据不足，标记为“弱疑似”，可给建议但不要强制合并，不应显著扣分。
                - 一旦发现“强证据命名冲突”，必须在 `missing_logic_details` 中给出冲突名称对、判定依据与推荐统一名称。

        ### 打分原则：
        - 初始按高分审计。
        - 每发现一类明显问题就扣分，尤其是命名不一致、逻辑断层、与门错误、中间节点缺失。
        - 仅“强证据命名冲突”可明显扣分；“弱疑似命名冲突”只做轻微提醒。
        - 如果存在多组强证据命名冲突，`score` 不应高于 0.75。
        - 如果既有命名冲突又有明显逻辑断层，`score` 应明显低于 0.75。

        ### 输出格式要求：
        必须输出纯 JSON 格式，包含以下字段：
        - "score": 0.0-1.0 之间的浮点数（0.8以下将触发重抽）。
                - "missing_logic_details": 具体问题列表。每一项建议使用对象格式：
                    {"type": "问题类型", "severity": "strong|weak", "description": "问题描述", "evidence": "判定依据", "suggestion": "修正建议"}
                - "advice": 给提取器的修正建议。仅当 `severity=strong` 时给出“应统一为 XXX”的强制建议。
        """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def evaluate(self, text: str, triplets: list) -> str:
        # 将当前的三元组转为字符串方便模型阅读
        triplets_str = json.dumps(triplets, ensure_ascii=False, indent=2)

        user_prompt = f"""
        请审计以下内容。

        审计时请重点检查：
        1. 是否把同一个事件写成了多个名字。
        2. 先判断是否满足“同一事件判定条件”，不满足则禁止合并。
        3. 若存在命名冲突，请明确写出“名称A 与 名称B 应统一为 名称C”，并给出原文证据。
        4. 若仅疑似同义且证据不足，标记为 weak，不要给强制合并建议。
        5. 不要把多个独立原因分支误判为 AND；若它们是并列独立致因，应明确说明这是 OR 逻辑。

        【原始文本】：
        {text}

        【当前三元组】：
        {triplets_str}
        """

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
            raise Exception(f"审计模型调用失败: {response.message}")