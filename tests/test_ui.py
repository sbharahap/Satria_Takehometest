from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import get_settings


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
