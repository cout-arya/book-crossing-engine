"""
Evaluation endpoints.
Serves model comparison tables, precision-recall graphs, and benchmark queries.
"""

import json
from fastapi import APIRouter, HTTPException
from backend.app.config import EVALUATION_RESULTS_PATH

router = APIRouter()


def _load_eval_results():
    if not EVALUATION_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run scripts/run_evaluation.py first."
        )
    try:
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {e}")


@router.get("/evaluation/summary")
def get_evaluation_summary():
    return _load_eval_results()


@router.get("/evaluation/comparison")
def get_evaluation_comparison():
    data = _load_eval_results()
    return {
        "num_queries": data.get("num_queries", 0),
        "cutoff_k": data.get("cutoff_k", 10),
        "comparison": data.get("metrics_comparison", [])
    }


@router.get("/evaluation/precision-recall")
def get_precision_recall_curves():
    data = _load_eval_results()
    return {
        "precision_recall_curves": data.get("precision_recall_curves", {})
    }


@router.get("/evaluation/queries")
def get_evaluation_queries():
    data = _load_eval_results()
    return {
        "queries": data.get("benchmark_queries", [])
    }
