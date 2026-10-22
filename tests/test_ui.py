from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import get_settings
from src.models import ProviderName
from tests.test_rag import agent, rag  # noqa: F401


def test_create_edit_delete_agent_in_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    try:
        app.run()
        assert not app.exception
        app.text_input(key="create_name").input("HR demo")
        next(button for button in app.button if button.label == "Simpan agent").click().run()
        assert not app.exception
        assert app.selectbox(key="active_agent").options == ["HR demo"]
        agent_id = app.selectbox(key="active_agent").value
        app.text_input(key=f"edit_{agent_id}_name").input("HR diperbarui")
        next(button for button in app.button if button.label == "Simpan perubahan").click().run()
        assert app.selectbox(key="active_agent").options == ["HR diperbarui"]
        app.checkbox(key=f"confirm_{agent_id}").check().run()
        next(button for button in app.button if button.label == "Hapus agent ini").click().run()
        assert not app.exception
        assert app.selectbox(key="active_agent").options == []
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


def test_provider_switch_sets_model_default_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    try:
        app.run()
        app.text_input(key="create_name").input("Legal provider demo")
        app.selectbox(key="create_provider").select(ProviderName.OPENAI).run()
        assert app.selectbox(key="create_model").value == get_settings().openai_model
        app.selectbox(key="create_provider").select(ProviderName.OPENROUTER).run()
        assert app.selectbox(key="create_model").value == get_settings().openrouter_model
        next(button for button in app.button if button.label == "Simpan agent").click().run()
        assert not app.exception
        agent_id = app.selectbox(key="active_agent").value
        app.selectbox(key=f"edit_{agent_id}_provider").select(ProviderName.LITELLM).run()
        assert app.selectbox(key=f"edit_{agent_id}_model").value == get_settings().litellm_model
        next(button for button in app.button if button.label == "Simpan perubahan").click().run()
        assert not app.exception
        from src.database.connection import Database

        with Database(get_settings().sqlite_path).connect() as connection:
            row = connection.execute(
                "SELECT provider,model FROM agents WHERE id=?", (agent_id,)
            ).fetchone()
        assert row["provider"] == "litellm" and row["model"] == get_settings().litellm_model
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


def test_reload_configuration_refreshes_key_status_without_displaying_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    try:
        app.run()
        assert any(c.value == "Key OpenRouter: belum diisi" for c in app.caption)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-never-show")
        app.button(key="reload_config").click().run()
        assert not app.exception
        assert any(c.value == "Key OpenRouter: terkonfigurasi" for c in app.caption)
        assert all("test-placeholder-never-show" not in c.value for c in app.caption)
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


def test_documents_and_chat_tabs_have_independent_agent_selection(rag, monkeypatch):  # noqa: F811
    import streamlit as st

    import src.bootstrap

    services, _, _ = rag
    first, second = agent(services, "HR Vale"), agent(services, "Legal Vale")
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
        app.run()
        assert not app.exception
        assert app.selectbox(key="documents_agent").value == first.id
        app.selectbox(key="documents_agent").select(second.id).run()
        assert not app.exception
        assert app.selectbox(key="documents_agent").value == second.id
        assert app.selectbox(key="active_agent").value == first.id
        app.selectbox(key="chat_agent").select(second.id).run()
        assert not app.exception
        assert app.selectbox(key="chat_agent").value == second.id
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


def test_saved_conversations_can_be_selected_again(rag, monkeypatch):  # noqa: F811
    import src.bootstrap

    services, _, _ = rag
    owner = agent(services, "HR history")
    old = services.chat.new_conversation(owner.id)
    services.chat.ask(owner.id, old, "Berapa jumlah cuti di dokumen?")
    recent = services.chat.new_conversation(owner.id)
    rows = services.chat.conversations(owner.id)
    assert {row["id"] for row in rows} == {old, recent}
    assert next(row for row in rows if row["id"] == old)["title"].startswith("Berapa jumlah")
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
        app.run()
        app.selectbox(key=f"history_{owner.id}").select(old).run()
        assert not app.exception
        assert app.session_state[f"conversation_{owner.id}"] == old
        assert any(m.value == "Berapa jumlah cuti di dokumen?" for m in app.markdown)
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()
