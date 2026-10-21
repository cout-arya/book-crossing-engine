"""
Dataset loading, cleaning, deduplication, and statistics service.
Handles Book-Crossing CSVs, preserves ISBN strings, generates ratings summaries,
and exports processed catalogs.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from backend.app.config import (
    RAW_BOOKS_PATH,
    RAW_RATINGS_PATH,
    RAW_USERS_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_BOOKS_PARQUET,
    PROCESSED_BOOKS_CSV,
    PROCESSED_STATS_JSON
)

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self):
        self.stats: Dict[str, Any] = {}

    @staticmethod
    def load_raw_books(path: Path = RAW_BOOKS_PATH) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Books file not found at {path}")
        
        # Read with Latin-1 encoding, semicolon separator, and string dtypes
        df = pd.read_csv(
            path,
            sep=';',
            encoding='latin-1',
            dtype=str,
            on_bad_lines='skip',
            low_memory=False
        )
        return df

    @staticmethod
    def load_raw_ratings(path: Path = RAW_RATINGS_PATH) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning(f"Ratings file not found at {path}")
            return None
        
        df = pd.read_csv(
            path,
            sep=';',
            encoding='latin-1',
            dtype={"User-ID": str, "ISBN": str, "Book-Rating": str},
            on_bad_lines='skip',
            low_memory=False
        )
        return df

    @staticmethod
    def load_raw_users(path: Path = RAW_USERS_PATH) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning(f"Users file not found at {path}")
            return None
        
        df = pd.read_csv(
            path,
            sep=';',
            encoding='latin-1',
            dtype=str,
            on_bad_lines='skip',
            low_memory=False
        )
        return df

    def clean_and_process(
        self,
        save_artifacts: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans raw Book-Crossing files, deduplicates by ISBN,
        aggregates rating information, and produces rich statistics.
        """
        logger.info("Loading raw dataset files...")
        df_books_raw = self.load_raw_books()
        df_ratings_raw = self.load_raw_ratings()
        df_users_raw = self.load_raw_users()

        raw_books_count = len(df_books_raw)
        raw_ratings_count = len(df_ratings_raw) if df_ratings_raw is not None else 0
        raw_users_count = len(df_users_raw) if df_users_raw is not None else 0

        # 1. Normalize Column Names for Books
        col_map = {
            "ISBN": "isbn",
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
            "Image-URL-S": "image_url_s",
            "Image-URL-M": "image_url_m",
            "Image-URL-L": "image_url_l"
        }
        df_books = df_books_raw.rename(columns={k: v for k, v in col_map.items() if k in df_books_raw.columns})

        # Ensure all core columns exist
        for col in ["isbn", "title", "author", "year", "publisher", "image_url_m", "image_url_l"]:
            if col not in df_books.columns:
                df_books[col] = ""

        # Count missing values before cleaning
        missing_counts_before = {
            "title": int(df_books["title"].isna().sum()),
            "author": int(df_books["author"].isna().sum()),
            "year": int(df_books["year"].isna().sum()),
            "publisher": int(df_books["publisher"].isna().sum())
        }

        # 2. Clean ISBNs (preserve leading zeros, trim whitespace)
        df_books["isbn"] = df_books["isbn"].astype(str).str.strip().str.upper()

        # 3. Filter unusable records: missing or empty title
        df_books["title"] = df_books["title"].fillna("").astype(str).str.strip()
        usable_mask = df_books["title"].str.len() > 0
        df_books = df_books[usable_mask].copy()

        # 4. Normalize metadata
        df_books["author"] = df_books["author"].fillna("Unknown").astype(str).str.strip()
        df_books.loc[df_books["author"] == "", "author"] = "Unknown"

        df_books["publisher"] = df_books["publisher"].fillna("Unknown").astype(str).str.strip()
        df_books.loc[df_books["publisher"] == "", "publisher"] = "Unknown"

        # Year cleaning
        def parse_year(val):
            try:
                y = int(str(val).strip())
                if 1800 <= y <= 2026:
                    return y
                return 0
            except Exception:
                return 0

        df_books["year_int"] = df_books["year"].apply(parse_year)
        df_books["year"] = df_books["year_int"].astype(str)

        # 5. Deduplicate books by ISBN
        duplicates_removed = int(df_books.duplicated(subset=["isbn"]).sum())
        df_books = df_books.drop_duplicates(subset=["isbn"]).reset_index(drop=True)

        # 6. Process Ratings
        ratings_stats = {}
        if df_ratings_raw is not None and not df_ratings_raw.empty:
            df_ratings = df_ratings_raw.rename(columns={
                "User-ID": "user_id",
                "ISBN": "isbn",
                "Book-Rating": "rating"
            })
            df_ratings["isbn"] = df_ratings["isbn"].astype(str).str.strip().str.upper()
            df_ratings["rating"] = pd.to_numeric(df_ratings["rating"], errors="coerce").fillna(0).astype(int)

            # Calculate distributions
            implicit_count = int((df_ratings["rating"] == 0).sum())
            explicit_mask = df_ratings["rating"] > 0
            explicit_count = int(explicit_mask.sum())

            rating_dist = df_ratings["rating"].value_counts().to_dict()
            rating_dist_formatted = {str(k): int(v) for k, v in sorted(rating_dist.items())}

            ratings_stats = {
                "total_ratings": raw_ratings_count,
                "implicit_ratings_count": implicit_count,
                "explicit_ratings_count": explicit_count,
                "rating_distribution": rating_dist_formatted
            }

            # Aggregate ratings per book
            book_ratings_agg = df_ratings.groupby("isbn").agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", lambda s: float(np.round(s[s > 0].mean() if (s > 0).any() else 0.0, 2)))
            ).reset_index()

            # Merge with books
            df_books = df_books.merge(book_ratings_agg, on="isbn", how="left")
            df_books["rating_count"] = df_books["rating_count"].fillna(0).astype(int)
            df_books["avg_rating"] = df_books["avg_rating"].fillna(0.0).astype(float)
        else:
            df_books["rating_count"] = 0
            df_books["avg_rating"] = 0.0
            ratings_stats = {
                "total_ratings": 0,
                "implicit_ratings_count": 0,
                "explicit_ratings_count": 0,
                "rating_distribution": {}
            }

        # 7. Year Distribution
        years_nonzero = df_books[df_books["year_int"] > 0]["year_int"]
        year_bins = {
            "Pre-1970": int((years_nonzero < 1970).sum()),
            "1970-1979": int(((years_nonzero >= 1970) & (years_nonzero < 1980)).sum()),
            "1980-1989": int(((years_nonzero >= 1980) & (years_nonzero < 1990)).sum()),
            "1990-1999": int(((years_nonzero >= 1990) & (years_nonzero < 2000)).sum()),
            "2000-2009": int(((years_nonzero >= 2000) & (years_nonzero < 2010)).sum()),
            "2010-Present": int((years_nonzero >= 2010).sum()),
            "Unknown/0": int((df_books["year_int"] == 0).sum())
        }

        # 8. Top Authors & Publishers
        top_authors = df_books[df_books["author"] != "Unknown"]["author"].value_counts().head(10).to_dict()
        top_publishers = df_books[df_books["publisher"] != "Unknown"]["publisher"].value_counts().head(10).to_dict()

        retained_books_count = len(df_books)

        # Compile final stats
        self.stats = {
            "books_loaded": raw_books_count,
            "books_retained": retained_books_count,
            "books_removed": raw_books_count - retained_books_count,
            "duplicates_removed": duplicates_removed,
            "ratings_loaded": raw_ratings_count,
            "users_loaded": raw_users_count,
            "missing_values_before_clean": missing_counts_before,
            "year_distribution": year_bins,
            "top_authors": {str(k): int(v) for k, v in top_authors.items()},
            "top_publishers": {str(k): int(v) for k, v in top_publishers.items()},
            **ratings_stats
        }

        if save_artifacts:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Save parquet for high performance and csv for portability
            try:
                df_books.to_parquet(PROCESSED_BOOKS_PARQUET, index=False)
                logger.info(f"Saved cleaned books to {PROCESSED_BOOKS_PARQUET}")
            except Exception as e:
                logger.warning(f"Could not save parquet ({e}), saving CSV only.")
            
            df_books.to_csv(PROCESSED_BOOKS_CSV, index=False, encoding="utf-8")
            logger.info(f"Saved cleaned books CSV to {PROCESSED_BOOKS_CSV}")

            with open(PROCESSED_STATS_JSON, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Saved dataset stats to {PROCESSED_STATS_JSON}")

        return df_books, self.stats

    @staticmethod
    def load_processed_books() -> pd.DataFrame:
        """Loads cleaned catalog from parquet or csv."""
        if PROCESSED_BOOKS_PARQUET.exists():
            return pd.read_parquet(PROCESSED_BOOKS_PARQUET)
        elif PROCESSED_BOOKS_CSV.exists():
            return pd.read_csv(PROCESSED_BOOKS_CSV, dtype=str)
        else:
            raise FileNotFoundError("Processed books dataset not found. Run preprocessing first.")

    @staticmethod
    def load_stats() -> Dict[str, Any]:
        """Loads dataset statistics JSON."""
        if PROCESSED_STATS_JSON.exists():
            with open(PROCESSED_STATS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
