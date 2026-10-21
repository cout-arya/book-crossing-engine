"""
Unit tests for dataset loading, CSV parsing, delimiter handling, and ISBN string preservation.
"""

import io
import pandas as pd
from scripts.download_dataset import validate_csv_file


def test_isbn_preservation_with_leading_zeros(tmp_path):
    # CSV with leading zeros in ISBN and trailing 'X'
    csv_content = (
        '"ISBN";"Book-Title";"Book-Author"\n'
        '"0195153448";"Classical Mythology";"Mark Morford"\n'
        '"034545104X";"Flesh Tones";"M. J. Rose"\n'
    )
    test_csv = tmp_path / "test_books.csv"
    test_csv.write_text(csv_content, encoding="latin-1")

    # Read as dtype=str
    df = pd.read_csv(test_csv, sep=';', encoding='latin-1', dtype=str)
    assert df["ISBN"].iloc[0] == "0195153448"
    assert df["ISBN"].iloc[1] == "034545104X"
    # Verify leading zero was not stripped
    assert df["ISBN"].iloc[0].startswith("0")


def test_validate_csv_file_valid(tmp_path):
    csv_content = (
        '"ISBN";"Book-Title";"Book-Author"\n'
        '"0123456789";"Test Title";"Test Author"\n'
    )
    test_csv = tmp_path / "valid.csv"
    test_csv.write_text(csv_content, encoding="latin-1")

    valid, rows, cols = validate_csv_file(test_csv, "books")
    assert valid is True
    assert rows == 1
    assert "ISBN" in cols


def test_validate_csv_file_missing_columns(tmp_path):
    csv_content = (
        '"ColA";"ColB"\n'
        '"Val1";"Val2"\n'
    )
    test_csv = tmp_path / "invalid.csv"
    test_csv.write_text(csv_content, encoding="latin-1")

    valid, rows, cols = validate_csv_file(test_csv, "books")
    assert valid is False
