import os
import json
import argparse
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.preprocessor import DataPreprocessor
from src.llm_extractor import DsExtractor


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")


def get_versioned_output_path(output_dir: str, input_filename: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    prefix = f"{base_name}_triplets_v"
    suffix = ".json"
    latest_version = 0

    for existing_name in os.listdir(output_dir):
        if not existing_name.startswith(prefix) or not existing_name.endswith(suffix):
            continue

        version_text = existing_name[len(prefix):-len(suffix)]
        if version_text.isdigit():
            latest_version = max(latest_version, int(version_text))

    next_version = latest_version + 1
    return os.path.join(output_dir, f"{base_name}_triplets_v{next_version}.json")


def main(input_path: str = "data/input"):
    if not os.getenv("SILICONFLOW_API_KEY"):
        raise EnvironmentError("未检测到 SILICONFLOW_API_KEY，请检查项目根目录下的 .env 文件或当前环境变量配置")

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = DataPreprocessor()
    extractor = DsExtractor()

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
            print("步骤1：预处理，提取文本内容")
            step_start = time.perf_counter()
            text = preprocessor.process_file(file_path)
            print(f"文本提取完成，耗时：{round(time.perf_counter() - step_start)}s")

            print("步骤2：LLM抽取，调用：DeepSeek V3.2抽取")
            step_start = time.perf_counter()
            json_result = extractor.extract(text, source_reference=filename)
            print(f"抽取完成，耗时：{round(time.perf_counter() - step_start)}s")

            parsed_data = json.loads(json_result)

            print("步骤3：保存抽取结果")

            output_file = get_versioned_output_path(output_dir, filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=4)

            print(f"成功保存提取结果至: {output_file}\n")

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