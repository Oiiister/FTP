#!/usr/bin/env python3
"""
LangGraph 故障树三元组抽取系统测试脚本

测试 LangGraphExtractor 的功能，包括：
1. 解析文本文件
2. 运行 LangGraph 工作流
3. 输出三元组结果
"""

import json
import os
import sys
from typing import Dict, Any

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from langgraph.graph import StateGraph, END
    from src.parser import parse_response
    from src.llm_extractor import QwenExtractor
    from src.llm_evaluator import QwenEvaluator
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需依赖: pip install langgraph")
    sys.exit(1)


class AgentState:
    """LangGraph 状态类"""
    def __init__(self):
        self.file_content = ""
        self.current_triplets = []
        self.evaluation_feedback = []
        self.evaluation_score = 0.0
        self.iteration_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "file_content": self.file_content,
            "current_triplets": self.current_triplets,
            "evaluation_feedback": self.evaluation_feedback,
            "evaluation_score": self.evaluation_score,
            "iteration_count": self.iteration_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """从字典创建实例"""
        state = cls()
        state.file_content = data.get("file_content", "")
        state.current_triplets = data.get("current_triplets", [])
        state.evaluation_feedback = data.get("evaluation_feedback", [])
        state.evaluation_score = data.get("evaluation_score", 0.0)
        state.iteration_count = data.get("iteration_count", 0)
        return state


def extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """抽取器节点 - 从文本中提取三元组"""
    iteration_count = state.get("iteration_count", 0)
    print(f"\n=== 第 {iteration_count + 1} 轮抽取 ===")
    
    # 实例化抽取器
    extractor = QwenExtractor()

    # 构造动态 Prompt
    feedback_text = ""
    if iteration_count > 0:
        # 将评估反馈加入下一轮的输入
        evaluation_feedback = state.get("evaluation_feedback", [])
        feedback_text = f"\n\n### 修正要求：\n" + "\n".join(evaluation_feedback)
        print(f"反馈信息: {feedback_text}")

    # 调用抽取逻辑
    try:
        file_content = state.get("file_content", "")
        json_result = extractor.extract(
            text=file_content + feedback_text,
            source_reference="test.txt"
        )
        
        # 解析 JSON 响应
        triplets = parse_response(json_result)
        print(f"抽取到 {len(triplets)} 个三元组")
        
        return {
            "current_triplets": triplets,
            "iteration_count": iteration_count + 1
        }
        
    except Exception as e:
        print(f"抽取失败: {e}")
        return {
            "current_triplets": [],
            "iteration_count": iteration_count + 1
        }


def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """评估器节点 - 评估三元组的逻辑质量"""
    iteration_count = state.get("iteration_count", 0)
    print(f"\n=== 第 {iteration_count} 轮评估 ===")
    
    # 实例化评估器
    evaluator = QwenEvaluator()

    # 如果没有提取到三元组，直接判 0 分
    current_triplets = state.get("current_triplets", [])
    if not current_triplets:
        print("未提取到任何三元组")
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": ["未提取到任何三元组，请重新检查文本。"]
        }

    try:
        # 调用 AI 进行逻辑审计
        file_content = state.get("file_content", "")
        json_result = evaluator.evaluate(
            text=file_content,
            triplets=current_triplets
        )

        # 解析审计报告
        report = json.loads(json_result)
        
        score = report.get("score", 0.0)
        feedback = report.get("missing_logic_details", [])
        advice = report.get("advice", "")
        
        print(f"评估得分: {score}")
        print(f"反馈意见: {feedback}")
        
        return {
            "evaluation_score": score,
            "evaluation_feedback": feedback,
            "feedback_to_extractor": advice
        }

    except Exception as e:
        print(f"评估失败: {e}")
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": [f"审计节点运行异常: {str(e)}"]
        }


def decide_to_end(state: Dict[str, Any]) -> str:
    """决定是否结束循环"""
    current_triplets = state.get("current_triplets", [])
    iteration_count = state.get("iteration_count", 0)
    evaluation_score = state.get("evaluation_score", 0.0)
    
    # 如果模型报错导致没有三元组，但迭代次数小于3，继续重试
    if not current_triplets and iteration_count < 3:
        return "continue"

    # 核心判断逻辑：得分达到0.8或达到最大迭代次数
    if evaluation_score >= 0.8 or iteration_count >= 3:
        return "end"

    return "continue"


def create_workflow():
    """创建 LangGraph 工作流"""
    # 实例化图
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("evaluator", evaluator_node)

    # 设置起点
    workflow.set_entry_point("extractor")

    # 普通边：抽取器完成后总是去评估器
    workflow.add_edge("extractor", "evaluator")

    # 条件边：根据评估结果决定是否继续
    workflow.add_conditional_edges(
        "evaluator",
        decide_to_end,
        {
            "continue": "extractor",  # 继续抽取
            "end": END                # 结束流程
        }
    )

    # 编译工作流
    return workflow.compile()


def read_text_file(file_path: str) -> str:
    """读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return ""
    except Exception as e:
        print(f"读取文件失败: {e}")
        return ""


def save_results(triplets: list, output_file: str):
    """保存三元组结果到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"triplets": triplets}, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def main():
    """主函数"""
    print("=== LangGraph 故障树三元组抽取系统 ===\n")
    
    # 输入文件路径
    input_file = input("请输入文本文件路径 (默认: test.txt): ").strip()
    if not input_file:
        input_file = "test.txt"
    
    # 读取文件内容
    file_content = read_text_file(input_file)
    if not file_content:
        print("文件内容为空，程序退出")
        return
    
    print(f"文件内容长度: {len(file_content)} 字符\n")
    
    # 创建初始状态
    initial_state = AgentState()
    initial_state.file_content = file_content
    
    # 创建并运行工作流
    app = create_workflow()
    
    print("开始运行 LangGraph 工作流...\n")
    
    try:
        # 运行工作流
        final_state = app.invoke(initial_state.to_dict())
        
        print("\n=== 抽取完成 ===")
        print(f"总迭代次数: {final_state['iteration_count']}")
        print(f"最终得分: {final_state['evaluation_score']}")
        print(f"抽取到 {len(final_state['current_triplets'])} 个三元组")
        
        # 显示三元组
        if final_state['current_triplets']:
            print("\n=== 抽取的三元组 ===")
            for i, triplet in enumerate(final_state['current_triplets'], 1):
                print(f"{i}. {triplet.get('subject_name', 'N/A')} -> {triplet.get('relation', 'N/A')} -> {triplet.get('object_name', 'N/A')}")
                print(f"   类型: {triplet.get('subject_type', 'N/A')} -> {triplet.get('object_type', 'N/A')}")
                print(f"   置信度: {triplet.get('confidence', 'N/A')}, 来源: {triplet.get('source', 'N/A')}\n")
        
        # 保存结果
        output_file = input("请输入结果保存路径 (默认: results.json): ").strip()
        if not output_file:
            output_file = "results.json"
        
        save_results(final_state['current_triplets'], output_file)
        
    except Exception as e:
        print(f"工作流运行失败: {e}")


if __name__ == "__main__":
    # 检查 API 密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请设置: export DASHSCOPE_API_KEY=your_api_key")
        
        # 尝试从环境文件读取
        env_file = ".env"
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith("DASHSCOPE_API_KEY"):
                            key = line.split('=', 1)[1].strip()
                            os.environ["DASHSCOPE_API_KEY"] = key
                            print("已从 .env 文件读取 API 密钥")
                            break
            except Exception:
                pass
    
    main()