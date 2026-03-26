"""
JSON 响应解析模块

提供用于解析大模型返回的 JSON 响应的工具函数。
"""

import json
import logging
from typing import List, Dict, Any

try:
    from .schemas import Triplet, TripletExtractionResult
    HAS_SCHEMAS = True
except ImportError:
    # 如果 schemas 模块不可用，提供基本的验证功能
    HAS_SCHEMAS = False
    Triplet = None
    TripletExtractionResult = None

logger = logging.getLogger(__name__)


def parse_response(json_response: str) -> List[Dict[str, Any]]:
    """
    解析大模型返回的 JSON 响应，提取三元组数据
    
    Args:
        json_response: 大模型返回的 JSON 字符串
        
    Returns:
        List[Dict]: 解析后的三元组字典列表
        
    Raises:
        json.JSONDecodeError: JSON 解析失败时抛出
        ValueError: 数据格式不符合预期时抛出
    """
    try:
        # 解析 JSON 字符串
        data = json.loads(json_response)
        
        # 检查是否包含三元组数据
        if "triplets" not in data:
            logger.warning("JSON 响应中未找到 'triplets' 字段")
            return []
        
        triplets_data = data["triplets"]
        
        # 验证三元组数据格式
        if not isinstance(triplets_data, list):
            raise ValueError("'triplets' 字段应该是一个列表")
        
        # 验证每个三元组的必需字段
        valid_triplets = []
        for i, triplet in enumerate(triplets_data):
            if not isinstance(triplet, dict):
                logger.warning(f"跳过第 {i} 个三元组：不是字典格式")
                continue
            
            # 检查必需字段
            required_fields = ["subject_name", "subject_type", "relation", 
                             "object_name", "object_type", "confidence", "source"]
            
            missing_fields = [field for field in required_fields if field not in triplet]
            if missing_fields:
                logger.warning(f"跳过第 {i} 个三元组：缺少字段 {missing_fields}")
                continue
            
            # 验证字段类型和值
            try:
                if HAS_SCHEMAS:
                    # 使用 Pydantic 模型验证数据格式
                    validated_triplet = Triplet(**triplet)
                    valid_triplets.append(validated_triplet.dict())
                else:
                    # 基本验证：检查字段类型
                    if not isinstance(triplet["subject_name"], str):
                        raise ValueError("subject_name 必须是字符串")
                    if not isinstance(triplet["object_name"], str):
                        raise ValueError("object_name 必须是字符串")
                    if not isinstance(triplet["confidence"], (int, float)):
                        raise ValueError("confidence 必须是数字")
                    if not 0 <= triplet["confidence"] <= 1:
                        raise ValueError("confidence 必须在 0-1 范围内")
                    
                    # 添加到有效列表
                    valid_triplets.append(triplet)
                    
            except Exception as e:
                logger.warning(f"跳过第 {i} 个三元组：验证失败 - {str(e)}")
                continue
        
        logger.info(f"成功解析 {len(valid_triplets)} 个有效三元组")
        return valid_triplets
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败：{str(e)}")
        raise
    except Exception as e:
        logger.error(f"解析响应时发生错误：{str(e)}")
        raise


def parse_response_with_validation(json_response: str):
    """
    解析并验证大模型返回的 JSON 响应，返回完整的提取结果
    
    Args:
        json_response: 大模型返回的 JSON 字符串
        
    Returns:
        包含验证后的三元组列表的结果对象
        
    Raises:
        json.JSONDecodeError: JSON 解析失败时抛出
        ValueError: 数据格式不符合预期时抛出
    """
    try:
        data = json.loads(json_response)
        
        if "triplets" not in data:
            raise ValueError("JSON 响应中缺少 'triplets' 字段")
        
        if HAS_SCHEMAS:
            # 使用 Pydantic 模型进行完整验证
            result = TripletExtractionResult(**data)
            logger.info(f"成功验证 {len(result.triplets)} 个三元组")
            return result
        else:
            # 基本验证版本
            triplets = parse_response(json_response)
            logger.info(f"成功验证 {len(triplets)} 个三元组")
            return {"triplets": triplets}
        
    except Exception as e:
        logger.error(f"完整验证解析失败：{str(e)}")
        raise


# 提供向后兼容的别名
parse_triplets = parse_response