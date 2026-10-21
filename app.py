import logging
import sqlite3

import streamlit as st
from pydantic import ValidationError

from src.bootstrap import initialize
from src.config import get_settings
from src.environment import ollama_status
from src.models import AgentInput, ProviderName
from src.services.agents import AgentDeletionError, AgentNotFoundError

st.set_page_config(page_title="Ruang Dokumen", page_icon="📚", layout="wide")
logger = logging.getLogger(__name__)


@st.cache_resource
def resources():
    settings = get_settings()
    return settings, initialize(settings)


@st.cache_data(ttl=30)
def local_status(base_url: str, model: str):
    from src.config import Settings

    return ollama_status(Settings(ollama_base_url=base_url, litellm_model=model))


def show_notice(message: str) -> None:
    st.session_state["notice"] = message
    st.rerun()


try:
    settings, services = resources()
except Exception:
    st.title("Ruang Dokumen")
    st.error("Aplikasi belum dapat membuka penyimpanan. Periksa konfigurasi dan jalankan doctor.")
    st.stop()

providers = list(ProviderName)
provider_labels = {"litellm": "LiteLLM · lokal", "openai": "OpenAI", "openrouter": "OpenRouter"}
default_models = {
    "litellm": settings.litellm_model,
    "openai": settings.openai_model,
    "openrouter": settings.openrouter_model,
}
agents = services.agents.list_agents()
if agents and st.session_state.get("active_agent") not in {agent.id for agent in agents}:
    st.session_state["active_agent"] = agents[0].id
elif not agents:
    st.session_state.pop("active_agent", None)
with st.sidebar:
    st.markdown("### 📚 Ruang Dokumen")
    st.caption("Asisten dengan sumber pengetahuan terpisah")
    selected_id = st.selectbox(
        "Agent aktif",
        options=[agent.id for agent in agents],
        format_func=lambda value: next(a.name for a in agents if a.id == value),
        placeholder="Buat agent untuk memulai",
        key="active_agent",
        disabled=not agents,
    )
    st.divider()
    st.caption("Data agent disimpan secara lokal dan bertahan setelah restart.")
    with st.expander("Koneksi layanan"):
        if st.button("Periksa Ollama", key="check_ollama"):
            local_status.clear()
            status = local_status(settings.ollama_base_url, settings.litellm_model)
            if status["selected_model_available"]:
                st.success("Ollama tersambung; model lokal tersedia.")
            elif status["reachable"]:
                st.warning("Ollama tersambung; model pilihan belum tersedia.")
            else:
                st.warning("Ollama belum dapat dihubungi.")
        st.caption(
            "Key OpenAI: " + ("terkonfigurasi" if settings.openai_api_key else "belum diisi")
        )
        st.caption(
            "Key OpenRouter: "
            + ("terkonfigurasi" if settings.openrouter_api_key else "belum diisi")
        )

st.title(settings.app_name)
st.write("Kelola asisten dan sumber pengetahuannya dalam satu ruang kerja.")
if notice := st.session_state.pop("notice", None):
    st.success(notice)
manage, documents, chat = st.tabs(["Kelola agent", "Dokumen", "Percakapan"])


def agent_fields(key: str, current=None):
    name = st.text_input("Nama agent", value=current.name if current else "", key=f"{key}_name")
    description = st.text_input(
        "Deskripsi", value=current.description if current else "", key=f"{key}_description"
    )
    instructions = st.text_area(
        "Instruksi agent",
        height=130,
        value=current.instructions
        if current
        else (
            "Jawab dalam Bahasa Indonesia berdasarkan dokumen aktif milik agent ini. "
            "Sertakan sumber. Jika bukti tidak tersedia, nyatakan tidak tahu."
        ),
        key=f"{key}_instructions",
    )
    initial = current.provider.value if current else settings.default_chat_provider
    provider = st.selectbox(
        "Provider",
        options=providers,
        index=providers.index(ProviderName(initial)),
        format_func=lambda value: provider_labels[value.value],
        key=f"{key}_provider",
    )
    model = st.text_input(
        "Model",
        value=current.model if current else default_models[initial],
        key=f"{key}_model",
        help="Gunakan ID model yang tersedia pada provider pilihan.",
    )
    st.caption("Model bawaan: " + " · ".join(f"{p}: {m}" for p, m in default_models.items()))
    return name, description, instructions, provider, model


def validated_agent(values):
    name, description, instructions, provider, model = values
    try:
        return AgentInput(
            name=name,
            description=description,
            instructions=instructions,
            provider=provider,
            model=model,
        )
    except ValidationError:
        st.error("Isi nama, instruksi, dan ID model yang valid. Nama maksimal 100 karakter.")
        return None


with manage:
    st.subheader("Agent Anda")
    st.caption("Contoh: HR PT Vale Indonesia dan Legal & Compliance PT Vale Indonesia.")
    if agents:
        st.dataframe(
            [
                {
                    "Nama": a.name,
                    "Deskripsi": a.description,
                    "Provider": a.provider.value,
                    "Model": a.model,
                }
                for a in agents
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Belum ada agent. Buat asisten pertama Anda melalui formulir di bawah.")

    with (
        st.expander("Buat agent", expanded=not agents),
        st.form("create_agent", clear_on_submit=True),
    ):
        values = agent_fields("create")
        if st.form_submit_button("Simpan agent", type="primary"):
            if payload := validated_agent(values):
                try:
                    services.agents.create(payload)
                    show_notice("Agent berhasil dibuat.")
                except (OSError, RuntimeError, sqlite3.Error):
                    st.error("Agent belum dapat disimpan. Periksa penyimpanan aplikasi.")

    if selected_id:
        current = services.agents.get(selected_id)
        with st.expander(f"Edit {current.name}"), st.form(f"edit_{current.id}"):
            values = agent_fields(f"edit_{current.id}", current)
            if st.form_submit_button("Simpan perubahan"):
                if payload := validated_agent(values):
                    try:
                        services.agents.update(current.id, payload)
                        show_notice("Konfigurasi agent diperbarui.")
                    except AgentNotFoundError as error:
                        st.error(str(error))
        with st.expander("Hapus agent"):
            confirmed = st.checkbox(
                "Hapus agent beserta seluruh dokumen dan percakapannya", key=f"confirm_{current.id}"
            )
            if st.button("Hapus agent ini", disabled=not confirmed):
                try:
                    services.agents.delete(current.id)
                    show_notice("Agent dan data terkait telah dihapus.")
                except (AgentDeletionError, AgentNotFoundError) as error:
                    st.error(str(error))
    pending = [
        a for a in services.agents.list_agents(include_deleting=True) if a.status == "deleting"
    ]
    for agent in pending:
        st.warning(f"Penghapusan {agent.name} belum selesai. Agent sudah dinonaktifkan.")
        if st.button("Coba hapus kembali", key=f"retry_{agent.id}"):
            try:
                services.agents.delete(agent.id)
                show_notice("Pembersihan data agent selesai.")
            except (AgentDeletionError, AgentNotFoundError) as error:
                st.error(str(error))

with documents:
    st.subheader("Dokumen agent")
    if selected_id:
        st.caption(f"Agent: {services.agents.get(selected_id).name}")
    st.info("Pengunggahan dan pemrosesan PDF/DOCX belum tersedia pada versi fondasi ini.")

with chat:
    st.subheader("Percakapan")
    st.info("Percakapan bersumber dokumen akan tersedia setelah pemrosesan knowledge diaktifkan.")
