"""
Evaluation metrics schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PrecisionRecallPoint(BaseModel):
    recall: float
    precision: float


class ModelMetrics(BaseModel):
    model_name: str
    precision_at_10: float
    recall: float
    map: float
    ndcg_at_10: float
    avg_latency_ms: float


class EvaluationQueryItem(BaseModel):
    query_id: str
    query_text: str
    category: str
    num_relevant_docs: int
    sample_relevant_isbns: List[str]


class EvaluationSummaryResponse(BaseModel):
    num_queries: int
    models_evaluated: List[str]
    metrics_comparison: List[ModelMetrics]
    precision_recall_curves: Dict[str, List[PrecisionRecallPoint]]
    protocol_description: str
    limitations_notes: str
