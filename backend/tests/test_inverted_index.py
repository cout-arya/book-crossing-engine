"""
Unit tests for Inverted Index data structures and retrieval logic.
"""

from backend.app.indexing.inverted_index import InvertedIndex


def test_inverted_index_build(sample_inverted_index):
    stats = sample_inverted_index.get_statistics()
    assert stats["total_documents"] == 6
    assert stats["vocabulary_size"] > 0
    assert stats["total_postings_entries"] > 0
    assert stats["avg_document_length"] > 0


def test_inverted_index_lookup(sample_inverted_index):
    # 'rowling' is stemmed to 'rowl' by PorterStemmer, and appears in doc 0 and doc 1
    postings = sample_inverted_index.get_postings("rowl")
    assert len(postings) == 2
    matched_doc_ids = [doc_id for doc_id, tf in postings]
    assert 0 in matched_doc_ids
    assert 1 in matched_doc_ids

    # lookup_term takes unstemmed input and normalizes it automatically
    term_info = sample_inverted_index.lookup_term("rowling")
    assert term_info["normalized_term"] == "rowl"
    assert term_info["df"] == 2


def test_inverted_index_query_inspection(sample_inverted_index):
    inspection = sample_inverted_index.inspect_query("harry potter rowling")
    # 'harry' -> 'harri', 'potter' -> 'potter', 'rowling' -> 'rowl'
    assert "potter" in inspection["processed_tokens"]
    assert "rowl" in inspection["processed_tokens"]
    assert inspection["total_candidate_documents"] >= 2


def test_inverted_index_unknown_term(sample_inverted_index):
    postings = sample_inverted_index.get_postings("nonexistenttermxyz")
    assert postings == []
    df = sample_inverted_index.get_doc_freq("nonexistenttermxyz")
    assert df == 0


def test_inverted_index_serialization(sample_inverted_index, tmp_path):
    save_path = tmp_path / "test_index.pkl"
    sample_inverted_index.save(save_path)
    assert save_path.exists()

    loaded = InvertedIndex.load(save_path)
    assert loaded.num_docs == sample_inverted_index.num_docs
    assert len(loaded.postings) == len(sample_inverted_index.postings)
