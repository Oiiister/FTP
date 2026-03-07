from pydantic import BaseModel, Field, validator
from typing import List

class Triplet(BaseModel):
    subject_name: str = Field(..., description="故障节点的具体名称，如 '气路压力不足'") # [cite: 4]
    subject_type: str = Field(..., description="故障节点的分类标签，如 'BasicEvent'") # [cite: 5]
    relation: str = Field(..., description="头实体与尾实体之间的关联关系，如 'resultsIn'") # [cite: 6]
    object_name: str = Field(..., description="故障节点的具体名称，如 '回收故障'") # [cite: 7]
    object_type: str = Field(..., description="故障节点的分类标签，如 'IntermediateEvent'") # [cite: 8]
    confidence: float = Field(..., description="该三元组的可信程度，取值范围 0~1") # [cite: 9]
    source: str = Field(..., description="该三元组的来源依据") # [cite: 10]

    @validator('subject_type', 'object_type')
    def check_entity_type(cls, v):
        allowed = {"BasicEvent", "IntermediateEvent", "TopEvent"} # [cite: 28]
        if v not in allowed:
            raise ValueError(f"实体类型必须是 {allowed} 之一")
        return v

    @validator('relation')
    def check_relation(cls, v):
        allowed = {"resultsIn", "causedBy", "relatedTo", "jointly_resultsIn"} # [cite: 30]
        if v not in allowed:
            raise ValueError(f"关系类型必须是 {allowed} 之一")
        return v

class TripletExtractionResult(BaseModel):
    triplets: List[Triplet]