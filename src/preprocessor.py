import os
import pdfplumber
import pytesseract
from PIL import Image

class DataPreprocessor:
    @staticmethod
    def process_file(file_path: str) -> str:
        """根据文件扩展名选择不同的解析方式，转换为统一文本"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        elif ext in ['.png', '.jpg', '.jpeg']:
            # 需要提前安装 tesseract-ocr
            return pytesseract.image_to_string(Image.open(file_path), lang='chi_sim')
        else:
            raise ValueError(f"暂不支持的文件格式: {ext}")