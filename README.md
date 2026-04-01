# FTP — 故障树三元组提取工具

基于硅基流动平台调用 DeepSeek 大模型的故障因果三元组自动提取工具，使用 OpenAI Python SDK 访问兼容接口，支持 TXT、PDF、图片等格式的技术文档输入，输出结构化的故障树三元组 JSON 文件。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9+ |
| Tesseract OCR | 图片文件识别所需（PDF/TXT 可不安装） |
| SiliconFlow API Key | 硅基流动平台申请 |

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
| `openai` | 通过 OpenAI SDK 调用硅基流动兼容接口 |
| `pydantic` | 输出结构定义与可选校验 |
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
SILICONFLOW_API_KEY=your_api_key_here
```

> 在硅基流动平台申请并获取 API Key。

---

## 使用方法

### 处理单个文件

```bash
python main.py --input-path data/input/your_file.pdf
例如：python main.py --input-path /Users/oyster/PycharmProjects/FTP/data/input/test1.txt
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

提取结果统一保存在 `/Users/oyster/PycharmProjects/FTP/data/output/` 目录下。

为避免覆盖历史结果：

- 不同输入文件会使用不同文件名前缀
- 同一输入文件的多次抽取会自动追加版本号

文件名格式为 `{原文件名}_triplets_v{版本号}.json`，例如：

```
/Users/oyster/PycharmProjects/FTP/data/output/test2_triplets_v1.json
/Users/oyster/PycharmProjects/FTP/data/output/test2_triplets_v2.json
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
  ├── llm_extractor.py     # 基于 OpenAI SDK 的 LLM 提取逻辑（直接整文抽取）
    ├── preprocessor.py      # 文档预处理（文本 / PDF / 图片）
  ├── schemas.py           # 输出结构定义（Pydantic）
    └── parser.py            # 辅助解析
```

CFY注：
1.保留了关于双Agent的框架：
llm_evaluator.py,LangGraphExtractor.ipynb,test_langgraph_extractor.py

2.注意：preprocessor.py是文本预处理部分，后续需要融合多模态。将多模态处理功能融合进def process_file(file_path: str) -> str:函数，防止破坏项目结构。