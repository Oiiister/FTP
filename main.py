import os
import json
import argparse  # 引入参数解析模块
import torch
from dotenv import load_dotenv
from src.preprocessor import DataPreprocessor, LegacyPreprocessor
from src.llm_extractor import QwenExtractor
from src.schemas import TripletExtractionResult
from pydantic import ValidationError

load_dotenv()


def run_extraction_legacy(file_path, output_dir, preprocessor, extractor):
    """旧版提取逻辑（仅文本）"""
    filename = os.path.basename(file_path)
    print(f"正在处理文件: {filename}")
    try:
        text = preprocessor.process_file(file_path)
        json_result = extractor.extract(text, source_reference=filename)

        parsed_data = json.loads(json_result)
        validated_data = TripletExtractionResult(**parsed_data)

        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_triplets.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(validated_data.model_dump_json(indent=4))
        print(f"成功保存至: {output_file}\n")
    except Exception as e:
        print(f"处理 {filename} 失败: {e}\n")


def run_extraction_multimodal(file_path, output_dir, preprocessor, extractor):
    """多模态提取逻辑"""
    filename = os.path.basename(file_path)
    print(f"正在处理文件: {filename}")
    
    try:
        # 使用增强版预处理器
        result = preprocessor.process_file(file_path)
        
        if not result['success']:
            print(f"文件预处理失败: {result['error']}\n")
            return
        
        # 提取文本内容
        text = result['text_content']
        
        # 如果有多模态编码信息，添加到提示词中
        embedding_available = result['embedding'] is not None and isinstance(result['embedding'], torch.Tensor)
        
        if embedding_available:
            # 在提示词中添加多模态信息
            enhanced_prompt = f"""
文件类型: {result['file_type']}
向量编码维度: {result['embedding_dim']}

原始内容:
{text}

请基于上述内容进行故障知识图谱抽取。
"""
            json_result = extractor.extract(enhanced_prompt, source_reference=filename)
        else:
            json_result = extractor.extract(text, source_reference=filename)

        parsed_data = json.loads(json_result)
        validated_data = TripletExtractionResult(**parsed_data)

        # 保存结果，包含多模态信息
        output_data = validated_data.model_dump()
        output_data['multimodal_info'] = {
            'file_type': result['file_type'],
            'embedding_dim': result['embedding_dim'],
            'embedding_available': embedding_available
        }

        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_triplets.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"成功保存至: {output_file}")
        print(f"文件类型: {result['file_type']}, 向量编码: {'可用' if result['embedding'] else '不可用'}\n")
        
    except Exception as e:
        print(f"处理 {filename} 失败: {e}\n")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="故障知识图谱抽取工具")
    parser.add_argument("--files", nargs='+', help="指定要处理的一个或多个文件名（在 data/input 目录下）")
    parser.add_argument("--mode", choices=['legacy', 'multimodal'], default='multimodal', 
                       help="处理模式: legacy(仅文本) 或 multimodal(多模态编码)")
    parser.add_argument("--no-encoding", action="store_true", help="禁用多模态编码（仅提取文本）")
    args = parser.parse_args()

    input_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    # 根据模式选择预处理器
    if args.mode == 'multimodal' and not args.no_encoding:
        preprocessor = DataPreprocessor(use_multimodal_encoding=True)
        extraction_func = run_extraction_multimodal
        print("使用多模态编码模式")
    else:
        preprocessor = LegacyPreprocessor()
        extraction_func = run_extraction_legacy
        print("使用传统文本模式")

    extractor = QwenExtractor()

    # 逻辑：如果用户指定了文件，则处理指定文件；否则遍历整个目录
    files_to_process = args.files if args.files else [f for f in os.listdir(input_dir) if
                                                      os.path.isfile(os.path.join(input_dir, f))]

    for filename in files_to_process:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            extraction_func(file_path, output_dir, preprocessor, extractor)
        else:
            print(f"警告: 文件 {filename} 在 {input_dir} 中未找到，已跳过。")


if __name__ == "__main__":
    main()