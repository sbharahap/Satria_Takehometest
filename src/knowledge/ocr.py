"""Local bounded OCR using Poppler and Tesseract. No document leaves the machine."""

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class OCRError(ValueError):
    pass


@dataclass(frozen=True)
class OCRConfig:
    language: str = "ind+eng"
    max_pages: int = 40
    page_timeout: int = 45
    total_timeout: int = 300
    tessdata_dir: Path | None = None


def recognize_pages(data: bytes, numbers: list[int], config: OCRConfig) -> dict[int, str]:
    if len(numbers) > config.max_pages:
        raise OCRError(f"PDF scan melebihi batas OCR {config.max_pages} halaman. Pecah dokumennya.")
    if not shutil.which("pdftoppm") or not shutil.which("tesseract"):
        raise OCRError(
            "OCR memerlukan Poppler dan Tesseract. Ikuti setup OCR di README atau gunakan Docker."
        )
    results = {}
    deadline = time.monotonic() + config.total_timeout
    with tempfile.TemporaryDirectory(prefix="document-ocr-") as directory:
        root = Path(directory)
        source = root / "source.pdf"
        source.write_bytes(data)
        for number in numbers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OCRError("OCR melewati batas waktu. Pecah PDF scan menjadi file lebih kecil.")
            timeout = min(config.page_timeout, remaining)
            prefix = root / "page"
            try:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-f",
                        str(number),
                        "-l",
                        str(number),
                        "-scale-to",
                        "2400",
                        "-singlefile",
                        "-png",
                        str(source),
                        str(prefix),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=timeout,
                )
                command = [
                    "tesseract",
                    str(prefix) + ".png",
                    "stdout",
                    "-l",
                    config.language,
                    "--psm",
                    "3",
                ]
                if config.tessdata_dir:
                    command += ["--tessdata-dir", str(config.tessdata_dir)]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OCRError(
                        "OCR melewati batas waktu. Pecah dokumen menjadi file lebih kecil."
                    )
                output = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    timeout=min(config.page_timeout, remaining),
                )
                results[number] = output.stdout.decode("utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                raise OCRError(
                    f"OCR halaman {number} melewati batas waktu. Coba PDF lebih kecil."
                ) from None
            except subprocess.CalledProcessError:
                raise OCRError(
                    f"OCR halaman {number} gagal. "
                    "Pastikan bahasa ind+eng terpasang dan file dapat dibaca."
                ) from None
    return results
