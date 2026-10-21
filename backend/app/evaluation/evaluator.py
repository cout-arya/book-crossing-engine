"""
Evaluation Engine.
Runs benchmark queries against all 4 retrieval models,
computes Precision@10, Recall@10, MAP, and nDCG@10,
and generates 11-point Precision-Recall curves.
Includes model warm-up for consistent, fair latency measurements.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

from backend.app.config import EVALUATION_RESULTS_PATH
from backend.app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    compute_11_point_interpolated_pr
)
from backend.app.evaluation.benchmark_queries import (
    build_benchmark_ground_truth
)

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, df_books: pd.DataFrame, retrievers: Dict[str, Any]):
        self.df_books = df_books
        self.retrievers = retrievers
        self.benchmark_queries = build_benchmark_ground_truth(df_books)

    def run_evaluation(self, cutoff_k: int = 10, max_retrieval_depth: int = 50) -> Dict[str, Any]:
        """Runs the benchmark across all models and computes metrics."""
        logger.info(f"Starting evaluation across {len(self.benchmark_queries)} benchmark queries...")
        
        comparison_results = []
        pr_curves = {}

        for model_key, retriever in self.retrievers.items():
            model_display_name = retriever.model_name
            is_avail = retriever.is_available
            logger.info(f"Evaluating {model_display_name} (Available: {is_avail})...")

            if not is_avail:
                comparison_results.append({
                    "model_name": model_display_name,
                    "precision_at_10": 0.0,
                    "recall": 0.0,
                    "map": 0.0,
                    "ndcg_at_10": 0.0,
                    "avg_latency_ms": 0.0,
                    "candidate_coverage_pct": 0.0,
                    "is_available": False,
                    "status_note": "Model artifacts not generated or model unavailable."
                })
                pr_curves[model_display_name] = [
                    {"recall": round(r * 0.1, 1), "precision": 0.0} for r in range(11)
                ]
                continue

            # Model Warm-up (1 query) so disk loading does not inflate latency
            try:
                retriever.retrieve("warmup query", limit=1)
            except Exception:
                pass

            precisions_at_k = []
            recalls_at_k = []
            average_precisions = []
            ndcgs_at_k = []
            latencies = []
            coverage_ratios = []

            all_retrieved_lists = []
            all_relevant_sets = []

            # Determine indexed ISBN set for candidate coverage reporting
            indexed_isbns = set(getattr(retriever, "doc_isbns", []))
            if not indexed_isbns and hasattr(retriever, "retrievers"):
                # Hybrid RRF: union of constituent retrievers
                indexed_isbns = set()
                for sub in retriever.retrievers.values():
                    indexed_isbns.update(getattr(sub, "doc_isbns", []))

            for q in self.benchmark_queries:
                q_text = q["query_text"]
                relevant_set = set(q["relevant_isbns"])
                rel_grades = q["relevance_grades"]

                # Candidate coverage for this query
                if indexed_isbns and len(relevant_set) > 0:
                    covered = sum(1 for isbn in relevant_set if isbn in indexed_isbns)
                    coverage_ratios.append(covered / float(len(relevant_set)))
                else:
                    coverage_ratios.append(1.0)

                # Pure retrieval timing (preprocessing + query encoding + score computation + ranking)
                t0 = time.perf_counter()
                results = retriever.retrieve(q_text, limit=max_retrieval_depth)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)

                retrieved_isbns = [isbn for isbn, _ in results]
                all_retrieved_lists.append(retrieved_isbns)
                all_relevant_sets.append(relevant_set)

                # Compute metrics
                p_k = precision_at_k(retrieved_isbns, relevant_set, k=cutoff_k)
                r_k = recall_at_k(retrieved_isbns, relevant_set, k=cutoff_k)
                ap = average_precision(retrieved_isbns, relevant_set, cutoff=max_retrieval_depth)
                ndcg = ndcg_at_k(retrieved_isbns, rel_grades, k=cutoff_k)

                precisions_at_k.append(p_k)
                recalls_at_k.append(r_k)
                average_precisions.append(ap)
                ndcgs_at_k.append(ndcg)

            # Macro-averaged metrics
            n_queries = len(self.benchmark_queries)
            mean_p = round(float(sum(precisions_at_k) / n_queries), 4)
            mean_r = round(float(sum(recalls_at_k) / n_queries), 4)
            mean_map = round(float(sum(average_precisions) / n_queries), 4)
            mean_ndcg = round(float(sum(ndcgs_at_k) / n_queries), 4)
            mean_lat = round(float(sum(latencies) / n_queries), 2)
            mean_cov = round(float(sum(coverage_ratios) / n_queries) * 100.0, 1)

            comparison_results.append({
                "model_name": model_display_name,
                "precision_at_10": mean_p,
                "recall": mean_r,
                "map": mean_map,
                "ndcg_at_10": mean_ndcg,
                "avg_latency_ms": mean_lat,
                "candidate_coverage_pct": mean_cov,
                "is_available": True,
                "status_note": f"Evaluated across all {n_queries} benchmark queries."
            })

            # Compute 11-point PR curve
            pr_curve = compute_11_point_interpolated_pr(
                all_retrieved_lists,
                all_relevant_sets,
                cutoff=max_retrieval_depth
            )
            pr_curves[model_display_name] = pr_curve

        evaluation_data = {
            "num_queries": len(self.benchmark_queries),
            "cutoff_k": cutoff_k,
            "models_evaluated": list(self.retrievers.keys()),
            "metrics_comparison": comparison_results,
            "precision_recall_curves": pr_curves,
            "benchmark_queries": [
                {
                    "query_id": q["query_id"],
                    "query_text": q["query_text"],
                    "category": q["category"],
                    "num_relevant_docs": q["num_relevant_docs"],
                    "sample_relevant_isbns": q["relevant_isbns"][:5]
                }
                for q in self.benchmark_queries
            ],
            "protocol_description": (
                "Evaluation was performed on 17 fixed benchmark queries spanning Exact Titles, "
                "Author searches, and Thematic/Natural Language keywords. Ground truth relevance "
                "was derived from catalog metadata and explicit user ratings (ratings >= 7.0). "
                "Implicit interactions (rating=0) were excluded from positive relevance. "
                "Models were pre-warmed so latency strictly measures query preprocessing, encoding, and ranking."
            ),
            "limitations_notes": (
                "Book-Crossing lacks human search-intent labels; relevance is established through title/author metadata "
                "and explicit positive ratings (>= 7.0). The dense semantic model covers the top 10,000 catalog books, "
                "while TF-IDF, BM25, and Hybrid RRF cover the full 271,045 catalog records."
            )
        }

        # Persist results
        EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {EVALUATION_RESULTS_PATH}")
        return evaluation_data
