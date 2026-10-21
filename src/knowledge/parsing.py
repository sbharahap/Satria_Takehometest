import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


class DocumentError(ValueError):
    """Safe, user-facing ingestion failure."""


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def parse_pdf(data: bytes) -> list[PageText]:
    if not data:
        raise DocumentError("File kosong. Unggah PDF yang memiliki teks.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise DocumentError("File bukan PDF yang valid atau file rusak.")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise DocumentError("PDF dilindungi kata sandi. Unggah salinan yang tidak terenkripsi.")
        if len(reader.pages) > 300:
            raise DocumentError("PDF melebihi batas 300 halaman. Pecah menjadi beberapa dokumen.")
        pages = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"[^\S\n]+", " ", text.replace("\x00", " ")).strip()
            total += len(text)
            if total > 1_000_000:
                raise DocumentError("Teks PDF terlalu besar. Pecah menjadi beberapa dokumen.")
            if text:
                pages.append(PageText(number, text))
        if not pages or sum(c.isalnum() for p in pages for c in p.text) < 20:
            raise DocumentError(
                "PDF tidak memiliki teks yang dapat dibaca (kosong atau hasil scan). "
                "OCR belum tersedia; unggah PDF dengan teks yang dapat diseleksi."
            )
        return pages
    except DocumentError:
        raise
    except Exception:
        raise DocumentError(
            "PDF rusak atau teksnya tidak dapat diproses. Coba file lain."
        ) from None


def chunk_pages(pages: list[PageText], tokenizer, size: int, overlap: int) -> list[PageText]:
    """Use model token offsets, preserving the original text and page locator."""
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk overlap")
    chunks = []
    for page in pages:
        offsets = tokenizer(
            page.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        for start in range(0, len(offsets), size - overlap):
            window = offsets[start : start + size]
            text = page.text[window[0][0] : window[-1][1]].strip()
            if text:
                chunks.append(PageText(page.page, text))
            if start + size >= len(offsets):
                break
    return chunks
