"""Small, secret-free readiness probes used by the UI and local CLI."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

import httpx

from src.config import PROJECT_ROOT, Settings


def ollama_status(settings: Settings) -> dict:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5, trust_env=False)
        response.raise_for_status()
        names = [item["name"] for item in response.json().get("models", [])]
        requested = settings.litellm_model.split("/", 1)[-1]
        return {
            "reachable": True,
            "models": names,
            "selected_model": requested,
            "selected_model_available": requested in names,
        }
    except (httpx.HTTPError, ValueError, KeyError):
        return {"reachable": False, "models": [], "selected_model_available": False}


def embedding_downloaded(settings: Settings) -> bool:
    model_dir = settings.embedding_cache_dir / (
        "models--" + settings.embedding_model.replace("/", "--")
    )
    return any(
        (snapshot / "config.json").is_file()
        and (
            (snapshot / "model.safetensors").is_file() or (snapshot / "pytorch_model.bin").is_file()
        )
        for snapshot in (model_dir / "snapshots").glob("*")
    )


def ocr_status(settings: Settings) -> dict:
    tools = {name: bool(shutil.which(name)) for name in ("pdftoppm", "tesseract")}
    languages = []
    if tools["tesseract"]:
        command = ["tesseract", "--list-langs"]
        directory = settings.ocr_tessdata_dir or PROJECT_ROOT / "data/ocr/tessdata"
        if directory.is_dir():
            command += ["--tessdata-dir", str(directory)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                languages = result.stdout.splitlines()[1:]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "enabled": settings.ocr_enabled,
        "tools": tools,
        "indonesian_and_english_ready": all(tools.values()) and {"ind", "eng"}.issubset(languages),
        "max_scan_pages": settings.ocr_max_pages,
    }


def environment_report(settings: Settings) -> dict:
    packages = {}
    for name in (
        "streamlit",
        "chromadb",
        "sentence-transformers",
        "pypdf",
        "python-docx",
        "litellm",
        "openai",
        "langgraph",
        "pydantic-settings",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    docker = {"installed": bool(shutil.which("docker")), "running": False}
    if docker["installed"]:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            docker["running"] = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "python": platform.python_version(),
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
        "platform": f"{platform.system()} {platform.machine()}",
        "packages": packages,
        "ollama": ollama_status(settings),
        "embedding_downloaded": embedding_downloaded(settings),
        "credentials_configured": {
            "openai": bool(settings.openai_api_key),
            "openrouter": bool(settings.openrouter_api_key),
            "litellm": bool(settings.litellm_api_key),
        },
        "docker": docker,
        "ocr": ocr_status(settings),
    }


def print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))
