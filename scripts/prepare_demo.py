"""Download public PTVI policies and extract their official Indonesian body pages.

PDFs stay under ignored data/. Source URLs, revisions and page maps are explicit.
No translation or content rewriting is performed.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from pypdf import PdfReader, PdfWriter

SOURCES = [
    (
        "hr",
        "people_policy_ptvi_pol_0014",
        5,
        7,
        "https://vale.com/documents/d/guest/-ptvi-pol-0014-people-policy",
        "2902cd8756a132d8fc044bf4219b24d359da991806fa3e91697d16f358fddbb4",
    ),
    (
        "hr",
        "diversity_inclusion_ptvi_pol_0036",
        5,
        7,
        "https://vale.com/documents/d/guest/ptvi-pol-0036-diversity-equity-and-inclusion",
        "feb4bbf72d6b57346e21ebccd5a5ef17925661aa56dfd37a20de46bda35b03fc",
    ),
    (
        "legal",
        "anti_corruption_ptvi_pol_0016",
        10,
        18,
        "https://vale.com/documents/d/guest/-ptvi-pol-0016-anti-corruption-policy",
        "1b556ab74bd037ff9278811c015113ee5280e581dc77fc81ee5d4996730a461a",
    ),
    (
        "legal",
        "human_rights_ptvi_pol_0005",
        17,
        32,
        "https://vale.com/documents/d/guest/ptvi-pol-0005-human-rights-policy-1",
        "4f4852f3a17d53366e30f86defb3f5395f2abbd471337db7fc9e4304c4a3a5b8",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals-dir", type=Path, help="Use previously downloaded originals")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    records = []
    for group, stem, first, last, url, digest in SOURCES:
        if args.originals_dir:
            raw = (args.originals_dir / group / f"{stem}.pdf").read_bytes()
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "RuangDokumen-Demo/0.1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read(20 * 1024 * 1024 + 1)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Source revision changed: {stem}. Verify pages before continuing.")
        target = args.output_dir / group / f"{stem}_id.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        import io

        reader, writer = PdfReader(io.BytesIO(raw)), PdfWriter()
        for page in reader.pages[first - 1 : last]:
            writer.add_page(page)
        writer.add_metadata(
            {
                "/Title": f"{stem} - Indonesian body",
                "/Subject": f"Extracted from {url}; original PDF pages {first}-{last}.",
            }
        )
        writer.write(target)
        reopened = PdfReader(target)
        assert len(reopened.pages) == last - first + 1
        assert all(page.extract_text().strip() for page in reopened.pages)
        records.append(
            {
                "file": str(target),
                "source_url": url,
                "original_sha256": digest,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "page_map": {str(i + 1): first + i for i in range(last - first + 1)},
            }
        )
        print(f"{target}: {len(reopened.pages)} pages")
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
