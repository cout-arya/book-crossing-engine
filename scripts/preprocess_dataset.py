"""
Dataset preprocessing script for Project 15: Book Search using Book-Crossing.
Cleans raw CSV data, deduplicates records, aggregates rating metrics,
and saves the cleaned catalog and statistics.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.dataset_service import DatasetService


def main():
    print("=" * 70)
    print("BOOK-CROSSING DATASET CLEANING & PREPROCESSING")
    print("=" * 70)
    service = DatasetService()
    try:
        df_cleaned, stats = service.clean_and_process(save_artifacts=True)
    except Exception as e:
        print(f"[-] Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Cleaning and Processing Statistics ---")
    print(f"Books loaded:             {stats['books_loaded']:,}")
    print(f"Books retained:           {stats['books_retained']:,}")
    print(f"Duplicates removed:       {stats['duplicates_removed']:,}")
    print(f"Unusable books removed:   {stats['books_removed']:,}")
    print(f"Ratings loaded:           {stats['ratings_loaded']:,}")
    print(f"Users loaded:             {stats['users_loaded']:,}")
    print(f"Explicit ratings (>0):    {stats['explicit_ratings_count']:,}")
    print(f"Implicit ratings (=0):    {stats['implicit_ratings_count']:,}")
    print("\nMissing values before clean:")
    for col, count in stats['missing_values_before_clean'].items():
        print(f"  - {col}: {count:,}")
    print("\nPublication Year Distribution:")
    for era, count in stats['year_distribution'].items():
        print(f"  - {era}: {count:,}")
    print("\nTop 5 Authors by Title Count:")
    for author, count in list(stats['top_authors'].items())[:5]:
        print(f"  - {author}: {count:,}")
    print("=" * 70)
    print("[+] Preprocessing completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
