"""Download official Tesseract fast Indonesian/English language data into ignored data/."""

import hashlib
import json
import urllib.request

from src.config import PROJECT_ROOT


def main():
    target = PROJECT_ROOT / "data/ocr/tessdata"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for language in ("ind", "eng"):
        url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{language}.traineddata"
        data = urllib.request.urlopen(url, timeout=60).read()
        if len(data) < 100_000:
            raise ValueError("Invalid OCR language download")
        (target / f"{language}.traineddata").write_bytes(data)
        records.append(
            {"language": language, "source": url, "sha256": hashlib.sha256(data).hexdigest()}
        )
    (target.parent / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("OCR ind+eng language data ready in data/ocr/tessdata")


if __name__ == "__main__":
    main()
