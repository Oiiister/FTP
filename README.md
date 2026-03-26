# FTP — 故障树三元组提取工具

基于大语言模型（阿里云 Qwen）的故障因果三元组自动提取工具，支持 TXT、PDF、图片等格式的技术文档输入，输出结构化的故障树三元组 JSON 文件。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9+ |
| Tesseract OCR | 图片文件识别所需（PDF/TXT 可不安装） |
| DashScope API Key | 阿里云百炼平台申请 |

---

## 安装依赖

```bash
# 1. 创建并激活虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 2. 安装 Python 依赖
pip install -r requirements.txt
```

**依赖清单：**

| 包名 | 用途 |
|---|---|
| `dashscope` | 调用阿里云 Qwen 大模型 API |
| `pydantic` | 输出结构验证 |
| `python-dotenv` | 加载 `.env` 环境变量 |
| `pdfplumber` | PDF 文本提取 |
| `pytesseract` | 图片 OCR 识别 |
| `Pillow` | 图片处理 |
| `tenacity` | 自动重试机制 |
| `langgraph` | 多 Agent 流程支持 |

### Tesseract OCR（处理图片时需要）

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

---

## 环境配置

在项目根目录创建 `.env` 文件，填入 API Key：

```
DASHSCOPE_API_KEY=your_api_key_here
```

> 在 [阿里云百炼平台](https://bailian.console.aliyun.com/) 注册并获取 API Key。

---

## 使用方法

### 处理单个文件

```bash
python main.py --input-path data/input/your_file.pdf
```

### 处理整个目录（批量）

```bash
python main.py --input-path data/input/
```

不指定路径时，默认处理 `data/input/` 目录下的所有文件：

```bash
python main.py
```

### 输出结果

提取结果保存在 `data/output/` 目录下，文件名格式为 `{原文件名}_triplets.json`，例如：

```
data/output/test2_triplets.json
```

---

## 输出格式说明

每个文件输出为 JSON 格式，包含三元组列表：

```json
{
  "triplets": [
    {
      "subject_name": "RAM芯片连接不良",
      "subject_type": "BasicEvent",
      "relation": "jointly_resultsIn",
      "object_name": "写RAM失败",
      "object_type": "IntermediateEvent",
      "confidence": 0.98,
      "source": "原始依据文本片段"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `subject_name` / `object_name` | 事件名称 |
| `subject_type` / `object_type` | 事件类型：`BasicEvent` / `IntermediateEvent` / `TopEvent` |
| `relation` | 关系类型：`resultsIn` / `causedBy` / `relatedTo` / `jointly_resultsIn` |
| `confidence` | 置信度（0~1） |
| `source` | 原文依据 |

---

## 项目结构

```
FTP/
├── main.py                  # 主入口
├── requirements.txt         # 依赖列表
├── .env                     # API Key（不纳入版本控制）
├── data/
│   ├── input/               # 输入文档（TXT / PDF / 图片）
│   └── output/              # 提取结果 JSON
└── src/
    ├── llm_extractor.py     # LLM 提取逻辑（分块 + 全局合并）
    ├── preprocessor.py      # 文档预处理（文本 / PDF / 图片）
    ├── schemas.py           # 输出结构 Schema（Pydantic）
    └── parser.py            # 辅助解析
```
