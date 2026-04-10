"""
故障树三元组抽取系统 (Fault Tree Triple Extraction System)

本模块提供从文本/PDF/图片中提取故障树三元组的功能，
支持使用LLM进行抽取和评估。
"""

from .schemas import Triplet, TripletExtractionResult
from .preprocessor import DataPreprocessor
from .llm_extractor import DeepSeekExtractor
from .llm_evaluator import QwenEvaluator
from .parser import parse_response, parse_response_with_validation, parse_triplets

try:
    from .lightrag_engine import LightRAGEngine, create_lightrag_engine
except ModuleNotFoundError:
    LightRAGEngine = None
    create_lightrag_engine = None

__all__ = [
    "Triplet",
    "TripletExtractionResult",
    "DataPreprocessor",
    "DeepSeekExtractor",
    "LightRAGEngine",
    "create_lightrag_engine",
    "QwenEvaluator",
    "parse_response",
    "parse_response_with_validation",
    "parse_triplets",
]

__version__ = "0.1.0"
