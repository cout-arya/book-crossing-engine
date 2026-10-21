"""
Text Preprocessing module for Information Retrieval.
Provides cleaning, tokenization, stop-word removal, and stemming.
Distinct pipelines for lexical retrieval (TF-IDF, BM25, Inverted Index)
and semantic dense retrieval (Sentence Transformer).
"""

import re
import unicodedata
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords are available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# Common regex patterns
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")
CLEAN_TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

stemmer = PorterStemmer()


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (e.g. accented characters in author names)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")


def clean_text(text: Optional[str]) -> str:
    """Basic text cleaning: unicode normalization and lowercase."""
    if text is None:
        return ""
    text = str(text)
    text = normalize_unicode(text)
    text = text.lower()
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Extract alphanumeric word tokens."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out English stopwords while preserving meaningful terms."""
    return [token for token in tokens if token not in STOP_WORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter Stemmer to tokens."""
    return [stemmer.stem(token) for token in tokens]


def preprocess_lexical(
    text: Optional[str],
    remove_stops: bool = True,
    apply_stem: bool = True
) -> List[str]:
    """
    Standard lexical preprocessing pipeline:
    1. Clean & Lowercase
    2. Tokenize
    3. Remove Stopwords (optional)
    4. Stem tokens (optional)
    Returns list of processed tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if apply_stem:
        tokens = stem_tokens(tokens)
    return tokens


def preprocess_lexical_string(
    text: Optional[str],
    remove_stops: bool = True,
    apply_stem: bool = True
) -> str:
    """Returns space-joined string of preprocessed tokens (useful for TF-IDF vectorizer)."""
    tokens = preprocess_lexical(text, remove_stops=remove_stops, apply_stem=apply_stem)
    return " ".join(tokens)


def preprocess_dense(title: Optional[str], author: Optional[str] = None, publisher: Optional[str] = None, year: Optional[str] = None) -> str:
    """
    Prepare text for Sentence Transformer.
    DO NOT stem or strip stopwords aggressively, as dense models require
    natural linguistic context and grammatical subwords.
    """
    parts = []
    if title and str(title).strip():
        parts.append(f"Title: {str(title).strip()}")
    if author and str(author).strip() and str(author).strip().lower() != "unknown":
        parts.append(f"Author: {str(author).strip()}")
    if publisher and str(publisher).strip():
        parts.append(f"Publisher: {str(publisher).strip()}")
    if year and str(year).strip() and str(year).strip() != "0":
        parts.append(f"Year: {str(year).strip()}")
    
    combined = " | ".join(parts)
    # Normalize multiple whitespace
    return WHITESPACE_PATTERN.sub(" ", combined).strip()


def build_searchable_document(
    title: Optional[str],
    author: Optional[str] = None,
    publisher: Optional[str] = None,
    year: Optional[str] = None
) -> str:
    """
    Combines metadata fields into a comprehensive string for lexical retrieval,
    weighting title and author by repeating or arranging cleanly.
    """
    title_str = str(title).strip() if title else ""
    author_str = str(author).strip() if author else ""
    publisher_str = str(publisher).strip() if publisher else ""
    year_str = str(year).strip() if year and str(year).strip() != "0" else ""

    # Emphasize title and author slightly for lexical match
    doc_text = f"{title_str} {title_str} {author_str} {publisher_str} {year_str}"
    return doc_text.strip()
