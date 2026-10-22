import argparse
import json
import sys
from datetime import UTC, datetime

from pydantic import ValidationError

from src.bootstrap import initialize
from src.config import get_settings
from src.environment import environment_report, print_report
from src.knowledge.embeddings import LocalEmbeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup dan pemeriksaan Ruang Dokumen")
    parser.add_argument("command", choices=["init", "doctor", "download-embedding"])
    args = parser.parse_args()
    try:
        settings = get_settings()
        settings.ensure_directories()
        if args.command == "download-embedding":
            embeddings = LocalEmbeddings(settings, local_files_only=False)
            vector = embeddings.embed_query("Apa kebijakan pengembangan karyawan?")
            receipt = {
                "model": settings.embedding_model,
                "dimension": len(vector),
                "verified_at": datetime.now(UTC).isoformat(),
            }
            (settings.embedding_cache_dir / "verified.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            print_report(receipt)
            return 0
        services = initialize(settings)
        if args.command == "init":
            print_report(
                {"sqlite": "ready", "chroma": "ready", "agents": len(services.agents.list_agents())}
            )
            return 0
        report = environment_report(settings)
        with services.database.connect() as connection:
            report["sqlite"] = {
                "integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
                "foreign_keys": bool(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
                "schema_version": connection.execute("PRAGMA user_version").fetchone()[0],
            }
        report["chroma"] = {"ready": True, "chunks": services.knowledge.count()}
        report["ready_for_local_development"] = (
            report["python_supported"]
            and all(report["packages"].values())
            and report["embedding_downloaded"]
            and report["ollama"]["selected_model_available"]
            and report["sqlite"]["integrity"] == "ok"
            and report["sqlite"]["foreign_keys"]
            and (not settings.ocr_enabled or report["ocr"]["indonesian_and_english_ready"])
        )
        print_report(report)
        return 0 if report["ready_for_local_development"] else 1
    except ValidationError as error:
        # Do not print Pydantic's raw inputs: a misconfigured field could contain a secret.
        fields = [".".join(str(part) for part in item["loc"]) for item in error.errors()]
        print("Konfigurasi tidak valid. Periksa field: " + ", ".join(fields), file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Pemeriksaan gagal ({type(error).__name__}). Periksa layanan dan konfigurasi.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
