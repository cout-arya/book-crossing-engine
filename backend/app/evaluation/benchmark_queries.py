"""
Evaluation Benchmark Definition & Ground Truth Construction.
Builds a fixed, reproducible benchmark suite covering:
1. Exact Title Queries
2. Author Queries
3. Natural-Language / Thematic Subject Queries

Ground truth relevance is tied directly to catalog metadata and explicit high user ratings (rating >= 7).
Zero-valued ratings (implicit feedback) are correctly treated as neutral, not negative.
"""

from typing import List, Dict, Set, Any
import pandas as pd


BENCHMARK_SPECIFICATIONS = [
    # 1. Exact Title Queries
    {
        "id": "Q01",
        "query": "The Da Vinci Code",
        "category": "Exact Title",
        "title_match": "The Da Vinci Code",
        "min_rating": 7.0
    },
    {
        "id": "Q02",
        "query": "To Kill a Mockingbird",
        "category": "Exact Title",
        "title_match": "To Kill a Mockingbird",
        "min_rating": 7.0
    },
    {
        "id": "Q03",
        "query": "The Catcher in the Rye",
        "category": "Exact Title",
        "title_match": "The Catcher in the Rye",
        "min_rating": 7.0
    },
    {
        "id": "Q04",
        "query": "Pride and Prejudice",
        "category": "Exact Title",
        "title_match": "Pride and Prejudice",
        "min_rating": 7.0
    },
    {
        "id": "Q05",
        "query": "1984",
        "category": "Exact Title",
        "title_match": "1984",
        "min_rating": 7.0
    },
    {
        "id": "Q06",
        "query": "The Hobbit",
        "category": "Exact Title",
        "title_match": "The Hobbit",
        "min_rating": 7.0
    },
    {
        "id": "Q07",
        "query": "Fahrenheit 451",
        "category": "Exact Title",
        "title_match": "Fahrenheit 451",
        "min_rating": 7.0
    },

    # 2. Author Queries
    {
        "id": "Q08",
        "query": "Stephen King",
        "category": "Author Search",
        "author_match": "Stephen King",
        "min_rating": 7.0
    },
    {
        "id": "Q09",
        "query": "Agatha Christie",
        "category": "Author Search",
        "author_match": "Agatha Christie",
        "min_rating": 7.0
    },
    {
        "id": "Q10",
        "query": "John Grisham",
        "category": "Author Search",
        "author_match": "John Grisham",
        "min_rating": 7.0
    },
    {
        "id": "Q11",
        "query": "J. K. Rowling",
        "category": "Author Search",
        "author_match": "J. K. Rowling",
        "min_rating": 7.0
    },
    {
        "id": "Q12",
        "query": "Dan Brown",
        "category": "Author Search",
        "author_match": "Dan Brown",
        "min_rating": 7.0
    },

    # 3. Thematic / Natural Language Queries
    {
        "id": "Q13",
        "query": "murder mystery detective thriller",
        "category": "Thematic / Natural Language",
        "keywords": ["murder", "mystery", "detective", "thriller", "crime"],
        "min_rating": 7.0
    },
    {
        "id": "Q14",
        "query": "vampire horror supernatural dark",
        "category": "Thematic / Natural Language",
        "keywords": ["vampire", "horror", "dracula", "supernatural", "dark"],
        "min_rating": 7.0
    },
    {
        "id": "Q15",
        "query": "magic wizard fantasy adventure",
        "category": "Thematic / Natural Language",
        "keywords": ["magic", "wizard", "fantasy", "dragon", "sorcerer"],
        "min_rating": 7.0
    },
    {
        "id": "Q16",
        "query": "courtroom legal justice lawyer drama",
        "category": "Thematic / Natural Language",
        "keywords": ["courtroom", "legal", "lawyer", "justice", "jury", "trial"],
        "min_rating": 7.0
    },
    {
        "id": "Q17",
        "query": "space science fiction alien galaxy",
        "category": "Thematic / Natural Language",
        "keywords": ["space", "alien", "galaxy", "star", "planet", "sci-fi"],
        "min_rating": 7.0
    }
]


def build_benchmark_ground_truth(df_books: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Constructs ground truth relevance sets from actual catalog entries.
    Assigns:
    - Grade 3 (Highly Relevant): Exact match with explicit positive user rating (>= 7.0)
    - Grade 2 (Relevant): Strong match with explicit positive rating (>= 7.0) or exact title match
    - Grade 1 (Partially Relevant): Metadata match with moderate positive rating (>= 5.0)
    Books with 0 ratings or ratings < 5.0 are excluded from positive ground truth.
    """
    eval_queries = []

    for spec in BENCHMARK_SPECIFICATIONS:
        q_id = spec["id"]
        q_text = spec["query"]
        category = spec["category"]
        
        relevance_grades: Dict[str, float] = {}
        relevant_isbns: Set[str] = set()

        if "title_match" in spec:
            target_title = spec["title_match"].lower()
            mask = df_books["title"].str.lower().str.contains(target_title, regex=False)
            matched_books = df_books[mask]
            
            for _, row in matched_books.iterrows():
                isbn = str(row["isbn"])
                t_lower = str(row["title"]).lower()
                rating_count = int(row.get("rating_count", 0))
                avg_rating = float(row.get("avg_rating", 0.0))

                is_exact = (
                    t_lower == target_title or
                    t_lower.startswith(target_title + ":") or
                    t_lower.startswith(target_title + " (")
                )

                if is_exact and avg_rating >= 7.0:
                    grade = 3.0
                elif is_exact or (avg_rating >= 7.0 and rating_count >= 3):
                    grade = 2.0
                elif avg_rating >= 5.0:
                    grade = 1.0
                else:
                    continue  # unrated or low rating excluded

                relevance_grades[isbn] = grade
                relevant_isbns.add(isbn)

        elif "author_match" in spec:
            target_author = spec["author_match"].lower()
            mask = df_books["author"].str.lower().str.contains(target_author, regex=False)
            matched_books = df_books[mask]
            
            for _, row in matched_books.iterrows():
                isbn = str(row["isbn"])
                avg_rating = float(row.get("avg_rating", 0.0))
                rating_count = int(row.get("rating_count", 0))

                # Positive explicit feedback
                if avg_rating >= 7.5 or (avg_rating >= 7.0 and rating_count >= 5):
                    grade = 3.0
                elif avg_rating >= 6.5 or (avg_rating >= 5.0 and rating_count >= 2):
                    grade = 2.0
                elif avg_rating >= 5.0:
                    grade = 1.0
                else:
                    continue

                relevance_grades[isbn] = grade
                relevant_isbns.add(isbn)

        elif "keywords" in spec:
            keywords = spec["keywords"]
            pattern = "|".join(keywords)
            mask = df_books["title"].str.lower().str.contains(pattern, regex=True)
            matched_books = df_books[mask]
            
            for _, row in matched_books.iterrows():
                isbn = str(row["isbn"])
                avg_rating = float(row.get("avg_rating", 0.0))
                t_lower = str(row["title"]).lower()
                kw_hits = sum(1 for kw in keywords if kw in t_lower)

                # Must match keywords AND have positive explicit rating
                if kw_hits >= 2 and avg_rating >= 7.0:
                    grade = 3.0
                elif kw_hits >= 1 and avg_rating >= 7.0:
                    grade = 2.0
                elif kw_hits >= 1 and avg_rating >= 5.5:
                    grade = 1.0
                else:
                    continue

                relevance_grades[isbn] = grade
                relevant_isbns.add(isbn)

        eval_queries.append({
            "query_id": q_id,
            "query_text": q_text,
            "category": category,
            "relevant_isbns": list(relevant_isbns),
            "relevance_grades": relevance_grades,
            "num_relevant_docs": len(relevant_isbns)
        })

    return eval_queries
