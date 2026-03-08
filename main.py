import os
import json
import argparse  # 引入参数解析模块
from dotenv import load_dotenv
from src.preprocessor import DataPreprocessor
from src.llm_extractor import QwenExtractor
from src.schemas import TripletExtractionResult
from pydantic import ValidationError

load_dotenv()


def run_extraction(file_path, output_dir, preprocessor, extractor):
    """提取单个文件的逻辑"""
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


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="故障知识图谱抽取工具")
    parser.add_argument("--files", nargs='+', help="指定要处理的一个或多个文件名（在 data/input 目录下）")
    args = parser.parse_args()

    input_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = DataPreprocessor()
    extractor = QwenExtractor()

    # 逻辑：如果用户指定了文件，则处理指定文件；否则遍历整个目录
    files_to_process = args.files if args.files else [f for f in os.listdir(input_dir) if
                                                      os.path.isfile(os.path.join(input_dir, f))]

    for filename in files_to_process:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            run_extraction(file_path, output_dir, preprocessor, extractor)
        else:
            print(f"警告: 文件 {filename} 在 {input_dir} 中未找到，已跳过。")


if __name__ == "__main__":
    main()