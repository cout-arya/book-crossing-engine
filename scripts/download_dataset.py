"""
Dataset acquisition script for Project 15: Book Search using Book-Crossing.
Downloads, validates integrity, extracts, and reports actual counts.
"""

import os
import sys
import shutil
import zipfile
import requests
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "backend" / "data" / "raw"
TMP_ZIP_PATH = RAW_DATA_DIR / "download_temp.zip"

EXPECTED_FILES = {
    "books": ["BX-Books.csv", "books.csv"],
    "ratings": ["BX-Book-Ratings.csv", "book_ratings.csv", "ratings.csv"],
    "users": ["BX-Users.csv", "users.csv"]
}

STANDARD_FILENAMES = {
    "books": "BX-Books.csv",
    "ratings": "BX-Book-Ratings.csv",
    "users": "BX-Users.csv"
}

SOURCES = [
    {
        "name": "Source A (freeCodeCamp Book-Crossing Archive)",
        "url": "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip",
        "type": "zip"
    },
    {
        "name": "Source B (GroupLens Book-Crossing mirror / archive)",
        "url": "https://files.grouplens.org/datasets/book-crossing/bx.zip",
        "type": "zip"
    },
    {
        "name": "Source C (Public GitHub Mirror - AKT-W)",
        "url": "https://raw.githubusercontent.com/AKT-W/Book-Crossing-Dataset/master/BX-Books.csv",
        "type": "direct_github"
    }
]

EXPECTED_COLUMNS = {
    "books": ["ISBN", "Book-Title", "Book-Author"],
    "ratings": ["User-ID", "ISBN", "Book-Rating"],
    "users": ["User-ID"]
}


def check_existing_files(target_dir: Path) -> dict:
    """Check if valid CSV files already exist in target_dir."""
    found = {}
    for key, candidates in EXPECTED_FILES.items():
        for cand in candidates:
            p = target_dir / cand
            if p.exists() and p.stat().st_size > 1024:
                found[key] = p
                break
    return found


def is_safe_path(base_dir: Path, path: Path) -> bool:
    """Prevent Zip Slip / directory traversal."""
    try:
        resolved_base = base_dir.resolve()
        resolved_path = path.resolve()
        return resolved_base in resolved_path.parents or resolved_base == resolved_path
    except Exception:
        return False


def validate_csv_file(file_path: Path, file_type: str) -> tuple[bool, int, list]:
    """Validate delimiter, encoding, readability, and required columns."""
    delimiters = [";", ",", "\t"]
    encodings = ["latin-1", "iso-8859-1", "cp1252", "utf-8"]
    
    for enc in encodings:
        for delim in delimiters:
            try:
                # Read first 100 rows to check columns and delimiter
                df_sample = pd.read_csv(
                    file_path,
                    sep=delim,
                    encoding=enc,
                    nrows=50,
                    dtype=str,
                    on_bad_lines='skip'
                )
                cols = [c.strip('" ') for c in df_sample.columns]
                req = EXPECTED_COLUMNS.get(file_type, [])
                matched = all(any(r.lower() == c.lower() for c in cols) for r in req)
                if matched:
                    # Count total rows
                    total_rows = 0
                    for chunk in pd.read_csv(
                        file_path,
                        sep=delim,
                        encoding=enc,
                        chunksize=50000,
                        dtype=str,
                        on_bad_lines='skip'
                    ):
                        total_rows += len(chunk)
                    return True, total_rows, cols
            except Exception:
                continue
    return False, 0, []


def extract_and_organize(zip_path: Path, target_dir: Path) -> dict:
    """Extract zip securely and flatten files into target_dir."""
    temp_extract_dir = target_dir / "_temp_extract"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Verifying ZIP archive integrity: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise ValueError(f"Corrupted file inside zip: {bad_file}")
        
        for member in zf.infolist():
            member_path = temp_extract_dir / member.filename
            if not is_safe_path(temp_extract_dir, member_path):
                raise SecurityError(f"Unsafe path detected in zip archive: {member.filename}")
        
        zf.extractall(temp_extract_dir)

    # Locate expected CSVs inside temp_extract_dir
    found_files = {}
    for root, _, files in os.walk(temp_extract_dir):
        for f in files:
            f_lower = f.lower()
            for key, patterns in EXPECTED_FILES.items():
                if key not in found_files and any(pat.lower() == f_lower for pat in patterns):
                    src_file = Path(root) / f
                    dest_file = target_dir / STANDARD_FILENAMES[key]
                    shutil.move(str(src_file), str(dest_file))
                    found_files[key] = dest_file

    # Clean up temp extract directory
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    return found_files


def download_file(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Download a file with streaming, redirect following, and validation."""
    print(f"Connecting to: {url} ...")
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "IR-BookSearch-Setup/1.0"})
        with session.get(url, stream=True, allow_redirects=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                print(f"[-] HTTP error {resp.status_code}: {resp.reason}")
                return False
            
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type and not url.endswith(".csv"):
                print(f"[-] Received HTML error page instead of archive: {content_type}")
                return False
            
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024 # 1 MB
            
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="")
                        else:
                            print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB", end="")
            print()
            return True
    except Exception as e:
        print(f"[-] Download error: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def run():
    print("=" * 70)
    print("BOOK-CROSSING DATASET SETUP & VERIFICATION")
    print("=" * 70)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Check local files
    existing = check_existing_files(RAW_DATA_DIR)
    if len(existing) == 3:
        print("[+] All 3 expected dataset files are already present locally in:")
        print(f"    {RAW_DATA_DIR}")
        all_valid = True
        for key, p in existing.items():
            valid, rows, cols = validate_csv_file(p, key)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"    - {p.name}: {size_mb:.2f} MB | {rows:,} rows | Valid: {valid}")
            if not valid:
                all_valid = False
        if all_valid:
            print("[+] Existing dataset validated successfully. No download needed.")
            return True
        else:
            print("[-] Existing files are corrupted or incomplete. Proceeding to download.")

    # 2. Iterate through configured sources
    download_success = False
    for source in SOURCES:
        print(f"\n[*] Trying {source['name']}...")
        if source["type"] == "zip":
            if download_file(source["url"], TMP_ZIP_PATH):
                try:
                    found = extract_and_organize(TMP_ZIP_PATH, RAW_DATA_DIR)
                    if TMP_ZIP_PATH.exists():
                        TMP_ZIP_PATH.unlink()
                    if "books" in found:
                        print("[+] Extraction completed successfully.")
                        download_success = True
                        break
                except Exception as e:
                    print(f"[-] Extraction failed: {e}")
                    if TMP_ZIP_PATH.exists():
                        TMP_ZIP_PATH.unlink()
        elif source["type"] == "direct_github":
            # Direct CSV downloads from repo
            repo_base = "https://raw.githubusercontent.com/AKT-W/Book-Crossing-Dataset/master/"
            success_count = 0
            for key, filename in STANDARD_FILENAMES.items():
                dest = RAW_DATA_DIR / filename
                if download_file(repo_base + filename, dest):
                    success_count += 1
            if success_count >= 1:
                download_success = True
                break

    if not download_success:
        print("\n" + "!" * 70)
        print("[-] AUTOMATIC DOWNLOAD FAILED FOR ALL CONFIGURED SOURCES.")
        print("!" * 70)
        print("Manual Setup Instructions:")
        print("1. Download the Book-Crossing dataset manually from:")
        print("   - https://cdn.freecodecamp.org/project-data/books/book-crossings.zip")
        print("   - OR https://grouplens.org/datasets/book-crossing/")
        print("2. Extract the archive.")
        print(f"3. Place the following CSV files directly into: {RAW_DATA_DIR}")
        print("   - BX-Books.csv")
        print("   - BX-Book-Ratings.csv")
        print("   - BX-Users.csv")
        print("4. Run: python scripts/validate_dataset.py")
        print("!" * 70)
        sys.exit(1)

    # 3. Post-download Validation
    print("\n" + "-" * 70)
    print("VALIDATING DOWNLOADED FILES")
    print("-" * 70)
    validation_passed = True
    for key, filename in STANDARD_FILENAMES.items():
        path = RAW_DATA_DIR / filename
        if not path.exists():
            print(f"[-] MISSING: {filename}")
            validation_passed = False
            continue
        valid, rows, cols = validate_csv_file(path, key)
        size_mb = path.stat().st_size / (1024 * 1024)
        status_sym = "[+]" if valid else "[-]"
        print(f"{status_sym} {filename:<20} | Size: {size_mb:>6.2f} MB | Rows: {rows:>9,} | Columns: {len(cols)}")
        if not valid:
            validation_passed = False

    if validation_passed:
        print("-" * 70)
        print("[+] SUCCESS: All Book-Crossing files downloaded, extracted and verified.")
        print("=" * 70)
        return True
    else:
        print("[-] One or more files failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    run()
