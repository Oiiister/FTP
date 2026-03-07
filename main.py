import os
import json
from dotenv import load_dotenv
from src.preprocessor import DataPreprocessor
from src.llm_extractor import QwenExtractor
from src.schemas import TripletExtractionResult
from pydantic import ValidationError

# 加载环境变量 (需要有一个包含 DASHSCOPE_API_KEY 的 .env 文件)
load_dotenv()


def main():
    input_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = DataPreprocessor()
    extractor = QwenExtractor()

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
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
    main()