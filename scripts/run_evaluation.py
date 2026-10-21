"""
Evaluation execution script for Project 15: Book Search using Book-Crossing.
Runs all 4 models against the benchmark ground truth, displays the comparison table,
and outputs precision-recall evaluation points.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.dataset_service import DatasetService
from backend.app.services.search_service import SearchService
from backend.app.evaluation.evaluator import Evaluator


def main():
    print("=" * 75)
    print("INFORMATION RETRIEVAL SYSTEM EVALUATION BENCHMARK")
    print("=" * 75)

    print("[*] Loading dataset and retrieval models...")
    df_books = DatasetService.load_processed_books()
    
    search_service = SearchService.get_instance()
    search_service.initialize()

    retrievers = {
        "tfidf": search_service.tfidf_retriever,
        "bm25": search_service.bm25_retriever,
        "dense": search_service.dense_retriever,
        "hybrid": search_service.hybrid_retriever
    }

    evaluator = Evaluator(df_books, retrievers)
    results = evaluator.run_evaluation(cutoff_k=10, max_retrieval_depth=50)

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 75)
    header = f"{'Model':<22} | {'P@10':<8} | {'Recall@10':<10} | {'MAP':<8} | {'nDCG@10':<8} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    for m in results["metrics_comparison"]:
        print(
            f"{m['model_name']:<22} | "
            f"{m['precision_at_10']:<8.4f} | "
            f"{m['recall']:<10.4f} | "
            f"{m['map']:<8.4f} | "
            f"{m['ndcg_at_10']:<8.4f} | "
            f"{m['avg_latency_ms']:<12.2f}"
        )

    print("-" * len(header))

    print("\n" + "-" * 75)
    print("11-POINT INTERPOLATED PRECISION-RECALL CURVE DATA")
    print("-" * 75)
    for model_name, curve in results["precision_recall_curves"].items():
        print(f"\nModel: {model_name}")
        rec_str = "Recall:    " + " ".join([f"{pt['recall']:>5.1f}" for pt in curve])
        prec_str = "Precision: " + " ".join([f"{pt['precision']:>5.3f}" for pt in curve])
        print(rec_str)
        print(prec_str)

    print("\n" + "=" * 75)
    print(f"[+] Evaluation results saved to {evaluator.benchmark_queries[0] if False else 'backend/artifacts/evaluation_results.json'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
