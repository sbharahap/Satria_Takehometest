from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from src.knowledge.parsing import DocumentError
from src.services.uploads import UploadQueue
from tests.test_rag import agent, rag  # noqa: F401


def test_upload_queue_is_serial_rejects_duplicate_and_checks_owner():
    started, release = Event(), Event()
    queue = UploadQueue()

    def upload(owner, name, content):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(filename=name, status="ready")

    try:
        job = queue.start("hr", [("policy.pdf", b"data")], SimpleNamespace(upload=upload))
        assert started.wait(5)
        assert queue.busy() and queue.get(job, "hr") == "processing"
        assert queue.get(job, "legal") is None
        with pytest.raises(DocumentError, match="sedang diproses"):
            queue.start("hr", [], SimpleNamespace(upload=upload))
        release.set()
        queue.executor.shutdown(wait=True)
        assert queue.get(job, "hr").ready == ("policy.pdf",)
        assert not queue.busy()
    finally:
        release.set()
        queue.executor.shutdown(wait=True)


def test_upload_failure_is_not_shown_as_success_or_exception_details():
    queue = UploadQueue()

    def upload(*args):
        raise RuntimeError("secret internal path or credential")

    job = queue.start("legal", [("broken.pdf", b"bad")], SimpleNamespace(upload=upload))
    queue.executor.shutdown(wait=True)
    receipt = queue.get(job, "legal")
    assert not receipt.ready and len(receipt.errors) == 1
    assert "secret" not in receipt.errors[0]


def test_upload_button_changes_to_processing_then_complete(rag, monkeypatch):  # noqa: F811
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.bootstrap
    import src.services.uploads
    from src.config import get_settings

    services, _, _ = rag
    owner = agent(services, "HR upload")
    started, release = Event(), Event()
    queue = UploadQueue()

    def upload(*args):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(filename="policy.pdf", status="ready")

    job = queue.start(owner.id, [("policy.pdf", b"data")], SimpleNamespace(upload=upload))
    assert started.wait(5)
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    monkeypatch.setattr(src.services.uploads, "UploadQueue", lambda: queue)
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
        app.session_state[f"upload_job_{owner.id}"] = job
        app.run()
        assert not app.exception
        button = app.button(key=f"upload_submit_{owner.id}")
        assert button.disabled and "Memproses" in button.label
        release.set()
        queue.executor.shutdown(wait=True)
        app.run()
        assert not app.exception
        assert app.button(key=f"upload_submit_{owner.id}").label == "✅ SELESAI"
    finally:
        release.set()
        queue.executor.shutdown(wait=True)
        st.cache_resource.clear()
        get_settings.cache_clear()
