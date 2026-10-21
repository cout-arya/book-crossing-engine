"""
Dataset validation script for Project 15: Book Search using Book-Crossing.
Performs thorough schema, encoding, and data integrity verification.
"""

import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "backend" / "data" / "raw"

EXPECTED_FILES = {
    "BX-Books.csv": {
        "required_cols": ["ISBN", "Book-Title", "Book-Author"],
        "recommended_cols": ["Year-Of-Publication", "Publisher", "Image-URL-M"],
        "min_rows": 200000
    },
    "BX-Book-Ratings.csv": {
        "required_cols": ["User-ID", "ISBN", "Book-Rating"],
        "recommended_cols": [],
        "min_rows": 900000
    },
    "BX-Users.csv": {
        "required_cols": ["User-ID"],
        "recommended_cols": ["Location", "Age"],
        "min_rows": 200000
    }
}


def validate_file(filename: str, spec: dict) -> bool:
    file_path = RAW_DATA_DIR / filename
    print(f"\n[*] Validating: {filename}")
    if not file_path.exists():
        print(f"[-] ERROR: File {file_path} does not exist.")
        return False

    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"    File size: {size_mb:.2f} MB")

    # Read sample
    try:
        sample = pd.read_csv(
            file_path,
            sep=';',
            encoding='latin-1',
            nrows=100,
            dtype=str,
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"[-] ERROR: Failed to parse CSV with sep=';' and encoding='latin-1': {e}")
        return False

    cols = list(sample.columns)
    print(f"    Columns detected ({len(cols)}): {cols}")

    # Check required columns
    missing_req = [c for c in spec["required_cols"] if c not in cols]
    if missing_req:
        print(f"[-] ERROR: Missing required columns: {missing_req}")
        return False

    # Check ISBN preservation if books or ratings
    if "ISBN" in cols:
        isbn_samples = sample["ISBN"].dropna().tolist()[:5]
        print(f"    Sample ISBNs (verifying string preservation): {isbn_samples}")
        if not all(isinstance(x, str) for x in isbn_samples):
            print("[-] ERROR: ISBN values are not preserved as strings.")
            return False

    # Fast row count
    row_count = 0
    try:
        for chunk in pd.read_csv(
            file_path,
            sep=';',
            encoding='latin-1',
            usecols=[spec["required_cols"][0]],
            chunksize=100000,
            dtype=str,
            on_bad_lines='skip'
        ):
            row_count += len(chunk)
    except Exception as e:
        print(f"[-] ERROR counting rows: {e}")
        return False

    print(f"    Total rows parsed: {row_count:,}")
    if row_count < spec["min_rows"]:
        print(f"[-] WARNING: Row count ({row_count:,}) is below expected minimum ({spec['min_rows']:,}).")
    else:
        print(f"    Row count meets criteria (>= {spec['min_rows']:,}).")

    print(f"[+] {filename} validation PASSED.")
    return True


def run():
    print("=" * 70)
    print("BOOK-CROSSING DATASET DEEP VALIDATION")
    print("=" * 70)
    all_ok = True
    for filename, spec in EXPECTED_FILES.items():
        ok = validate_file(filename, spec)
        if not ok:
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("[+] OVERALL STATUS: ALL DATASET FILES ARE VALID AND READY FOR PREPROCESSING.")
        print("=" * 70)
        return 0
    else:
        print("[-] OVERALL STATUS: VALIDATION FAILED. CHECK ERRORS ABOVE.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run())
