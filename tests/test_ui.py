from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import get_settings
from src.models import ProviderName


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
        assert app.text_input(key="create_model").value == get_settings().openai_model
        app.selectbox(key="create_provider").select(ProviderName.OPENROUTER).run()
        assert app.text_input(key="create_model").value == get_settings().openrouter_model
        next(button for button in app.button if button.label == "Simpan agent").click().run()
        assert not app.exception
        agent_id = app.selectbox(key="active_agent").value
        app.selectbox(key=f"edit_{agent_id}_provider").select(ProviderName.LITELLM).run()
        assert app.text_input(key=f"edit_{agent_id}_model").value == get_settings().litellm_model
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
