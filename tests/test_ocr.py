"""Bounded local OCR failure handling and original-page provenance."""

import io
import subprocess
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, NumberObject

from src.knowledge.ocr import OCRConfig, OCRError, recognize_pages
from src.knowledge.parsing import DocumentError, parse_pdf


def scan_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(100, 100)
    image = DecodedStreamObject()
    image.set_data(bytes([255, 255, 255]))
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    from pypdf.generic import DictionaryObject

    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): writer._add_object(image)})}
    )
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_scan_opt_in_preserves_page_and_ocr_label(monkeypatch):
    data = scan_pdf()
    with pytest.raises(DocumentError, match="scan"):
        parse_pdf(data)
    recognize = Mock(return_value={1: "Karyawan memperoleh cuti tahunan dua belas hari kerja."})
    monkeypatch.setattr("src.knowledge.parsing.recognize_pages", recognize)
    pages = parse_pdf(data, ocr=OCRConfig())
    assert pages[0].page == 1 and pages[0].locator == "Halaman PDF 1 · OCR"
    recognize.assert_called_once_with(data, [1], OCRConfig())


def test_ocr_missing_tools_and_page_limit(monkeypatch):
    monkeypatch.setattr("src.knowledge.ocr.shutil.which", lambda command: None)
    with pytest.raises(OCRError, match="Poppler"):
        recognize_pages(b"pdf", [1], OCRConfig())
    with pytest.raises(OCRError, match="batas OCR"):
        recognize_pages(b"pdf", [1, 2], OCRConfig(max_pages=1))


def test_ocr_timeout_is_safe_user_message(monkeypatch):
    monkeypatch.setattr("src.knowledge.ocr.shutil.which", lambda command: "/bin/tool")
    monkeypatch.setattr(
        "src.knowledge.ocr.subprocess.run",
        Mock(side_effect=subprocess.TimeoutExpired("sensitive-path", 1)),
    )
    with pytest.raises(OCRError, match="halaman 3 melewati batas waktu") as error:
        recognize_pages(b"pdf", [3], OCRConfig())
    assert "sensitive-path" not in str(error.value)
