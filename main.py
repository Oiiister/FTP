import os
import json
import argparse
from dotenv import load_dotenv

# 加载环境变量 (需要有一个包含 DASHSCOPE_API_KEY 的 .env 文件)
load_dotenv()

from src.preprocessor import DataPreprocessor
from src.llm_extractor import QwenExtractor
from src.schemas import TripletExtractionResult
from pydantic import ValidationError


def main(input_path: str = "data/input"):
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError("未检测到 DASHSCOPE_API_KEY，请检查项目根目录下的 .env 文件或当前环境变量配置")

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = DataPreprocessor()
    extractor = QwenExtractor()

    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        files_to_process = [
            os.path.join(input_path, filename)
            for filename in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, filename))
        ]
    else:
        raise FileNotFoundError(f"输入路径不存在或不可用: {input_path}")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        print(f"正在处理文件: {filename}")
        try:
            # 1. 预处理提取文本
            text = preprocessor.process_file(file_path)

            # 2. 调用模型进行抽取
            json_result = extractor.extract(text, source_reference=filename)

            # 3. 校验并解析 JSON
            parsed_data = json.loads(json_result)
            validated_data = TripletExtractionResult(**parsed_data)

            # 4. 保存为标准 JSON 文件
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_triplets.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(validated_data.model_dump_json(indent=4))

            print(f"成功保存提取结果至: {output_file}\n")

        except ValidationError as e:
            print(f"文件 {filename} 抽取的数据不符合Schema规范:\n{e}\n")
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从指定文件或目录中提取三元组信息")
    parser.add_argument(
        "-i",
        "--input-path",
        default="data/input",
        help="待处理输入路径，可传单个文件路径或目录路径",
    )
    args = parser.parse_args()
    main(input_path=args.input_path)