"""
Unit tests for text preprocessing pipeline.
"""

from backend.app.preprocessing.preprocessor import (
    clean_text,
    tokenize,
    remove_stopwords,
    stem_tokens,
    preprocess_lexical,
    preprocess_lexical_string,
    preprocess_dense,
    build_searchable_document
)


def test_clean_text():
    assert clean_text("Harry Potter & The Deathly Hallows") == "harry potter & the deathly hallows"
    # Unicode accents
    assert clean_text("Café René") == "cafe rene"
    # None and empty
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_tokenize():
    tokens = tokenize("hello world 123, testing-tokenization!")
    assert tokens == ["hello", "world", "123", "testing", "tokenization"]
    assert tokenize("") == []
    assert tokenize("   !@#$%^   ") == []


def test_remove_stopwords():
    tokens = ["the", "quick", "brown", "fox", "and", "a", "dog"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "and" not in filtered
    assert "a" not in filtered
    assert "quick" in filtered
    assert "fox" in filtered


def test_stem_tokens():
    tokens = ["running", "cats", "retrieval", "books"]
    stemmed = stem_tokens(tokens)
    assert stemmed == ["run", "cat", "retriev", "book"]


def test_preprocess_lexical_pipeline():
    text = "The Da Vinci Code: Mysteries of the Holy Grail"
    tokens = preprocess_lexical(text, remove_stops=True, apply_stem=True)
    # Stopwords like 'the', 'of' are removed; 'mysteries' stemmed to 'mysteri'
    assert "da" in tokens
    assert "vinci" in tokens
    assert "code" in tokens
    assert "mysteri" in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_preprocess_dense():
    dense_text = preprocess_dense(
        title="1984",
        author="George Orwell",
        publisher="Signet",
        year="1990"
    )
    assert "Title: 1984" in dense_text
    assert "Author: George Orwell" in dense_text
    assert "Publisher: Signet" in dense_text
    assert "Year: 1990" in dense_text


def test_empty_and_edge_cases():
    assert preprocess_lexical(None) == []
    assert preprocess_lexical("    ") == []
    assert preprocess_lexical("!@#$%^&*()") == []
    assert preprocess_lexical_string("   ") == ""
    assert build_searchable_document(None, None, None, None) == ""
