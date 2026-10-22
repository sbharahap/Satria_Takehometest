import logging
import sqlite3

import streamlit as st
from pydantic import ValidationError

from src.bootstrap import initialize
from src.config import get_settings
from src.environment import ollama_status
from src.knowledge.parsing import DocumentError, pdf_page_preview
from src.models import AgentInput, ProviderName
from src.providers.base import ProviderError
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
        if settings.openrouter_free_only:
            st.caption("OpenRouter: hanya model gratis yang diizinkan.")
        if settings.local_fallback_enabled:
            st.caption(f"Fallback OpenRouter → lokal: {settings.local_fallback_model}")
        if st.button("Muat ulang konfigurasi", key="reload_config"):
            get_settings.cache_clear()
            resources.clear()
            local_status.clear()
            st.rerun()

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
    previous_provider = st.session_state.get(f"{key}_provider_seen", provider.value)
    if previous_provider != provider.value:
        st.session_state[f"{key}_model"] = default_models[provider.value]
    st.session_state[f"{key}_provider_seen"] = provider.value
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

    if st.session_state.pop("_reset_create", False):
        for key in list(st.session_state):
            if key.startswith("create_"):
                del st.session_state[key]
    with st.expander("Buat agent", expanded=not agents):
        values = agent_fields("create")
        if st.button("Simpan agent", type="primary"):
            if payload := validated_agent(values):
                try:
                    services.agents.create(payload)
                    st.session_state["_reset_create"] = True
                    show_notice("Agent berhasil dibuat.")
                except (OSError, RuntimeError, sqlite3.Error):
                    st.error("Agent belum dapat disimpan. Periksa penyimpanan aplikasi.")

    if selected_id:
        current = services.agents.get(selected_id)
        with st.expander(f"Edit {current.name}"):
            values = agent_fields(f"edit_{current.id}", current)
            if st.button("Simpan perubahan"):
                if payload := validated_agent(values):
                    try:
                        services.agents.update(current.id, payload)
                        show_notice("Konfigurasi agent diperbarui.")
                    except AgentNotFoundError as error:
                        st.error(str(error))
                    except sqlite3.Error:
                        st.error("Konfigurasi belum dapat disimpan. Periksa penyimpanan aplikasi.")
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
        st.caption(
            "PDF atau DOCX dengan teks, maksimal "
            f"{settings.max_upload_mb} MB/file. OCR belum tersedia."
        )
        with st.form(f"upload_{selected_id}", clear_on_submit=True):
            uploads = st.file_uploader(
                "Pilih PDF atau DOCX",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                max_upload_size=settings.max_upload_mb,
            )
            submit_upload = st.form_submit_button("Proses dokumen", type="primary")
        if submit_upload:
            if not uploads:
                st.warning("Pilih minimal satu file PDF atau DOCX.")
            for upload in uploads:
                with st.status(f"Memproses {upload.name}…", expanded=True) as status:
                    try:
                        doc = services.documents.upload(selected_id, upload.name, upload.getvalue())
                        if doc.status == "ready":
                            status.update(
                                label=f"{doc.filename} siap · {doc.chunks_count} chunk",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            st.error(doc.error_message)
                            status.update(label=f"{doc.filename} gagal diproses", state="error")
                    except (DocumentError, AgentNotFoundError) as error:
                        st.error(str(error))
                        status.update(label="Upload gagal", state="error")
        uploaded_docs = services.documents.list_documents(selected_id)
        if not uploaded_docs:
            st.info("Belum ada dokumen. Unggah PDF atau DOCX untuk menyiapkan knowledge agent ini.")
        for doc in uploaded_docs:
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.write(doc.filename)
                    st.caption(
                        f"Status: {doc.status} · {doc.chunks_count} chunk · revisi {doc.revision}"
                    )
                    if doc.error_message:
                        st.error(doc.error_message)
                with st.expander("Tinjau dan koreksi hasil ekstraksi"):
                    pages = services.documents.extraction(selected_id, doc.id)
                    if pages:
                        st.caption(
                            "Bandingkan dengan dokumen asli, lalu koreksi kesalahan ekstraksi. "
                            "Simpan akan mengganti knowledge dengan teks koreksi "
                            "dan mengindeks ulang. "
                            "PDF asli tetap disimpan tanpa perubahan. OCR belum tersedia."
                        )
                        page_index = st.selectbox(
                            "Halaman atau bagian",
                            options=range(len(pages)),
                            format_func=lambda i, pages=pages: (
                                pages[i].get("locator") or f"Halaman PDF {pages[i]['page']}"
                            ),
                            key=f"extraction_page_{doc.id}",
                        )
                        selected_page = pages[page_index]
                        original = services.documents.original_file(selected_id, doc.id)
                        preview, editor = st.columns(2)
                        with preview:
                            st.markdown("**Dokumen asli**")
                            if original:
                                if doc.file_type == "pdf":
                                    st.caption(
                                        f"Halaman {selected_page['page']} dari PDF asli. "
                                        "Viewer mengikuti pilihan halaman editor."
                                    )
                                    try:
                                        page_pdf = pdf_page_preview(original, selected_page["page"])
                                        st.pdf(
                                            page_pdf,
                                            height=480,
                                            key=f"original_pdf_{doc.id}_{page_index}",
                                            alt=f"Halaman {selected_page['page']} · {doc.filename}",
                                        )
                                    except DocumentError as error:
                                        st.error(str(error))
                                else:
                                    st.info(
                                        "Pratinjau visual DOCX belum tersedia. Unduh file aslinya."
                                    )
                                st.download_button(
                                    "Unduh dokumen asli",
                                    data=original,
                                    file_name=doc.filename,
                                    mime="application/pdf"
                                    if doc.file_type == "pdf"
                                    else (
                                        "application/vnd.openxmlformats-officedocument."
                                        "wordprocessingml.document"
                                    ),
                                    key=f"download_original_{doc.id}",
                                )
                            else:
                                st.info(
                                    "File asli belum tersimpan pada versi lama. "
                                    "Unggah ulang untuk melihatnya."
                                )
                        with editor:
                            st.markdown(f"**Teks kerja · revisi {doc.revision}**")
                            with st.form(f"edit_text_{doc.id}_{doc.revision}_{page_index}"):
                                edited_text = st.text_area(
                                    "Teks hasil ekstraksi",
                                    value=selected_page["text"],
                                    height=480,
                                    key=f"extraction_text_{doc.id}_{doc.revision}_{page_index}",
                                    disabled=doc.status == "processing",
                                )
                                save_text = st.form_submit_button(
                                    "Simpan koreksi dan indeks ulang",
                                    disabled=doc.status == "processing",
                                )
                            st.caption(
                                "Belum disimpan sampai tombol simpan ditekan. "
                                "Simpan tiap halaman sebelum berpindah."
                            )
                            if save_text:
                                try:
                                    with st.spinner("Menyimpan koreksi dan mengindeks ulang…"):
                                        updated = services.documents.edit_page(
                                            selected_id,
                                            doc.id,
                                            page_index,
                                            edited_text,
                                            doc.revision,
                                        )
                                    if updated.status == "ready":
                                        show_notice(
                                            "Koreksi tersimpan. Chat menggunakan revisi terbaru."
                                        )
                                    else:
                                        st.error(updated.error_message)
                                except (DocumentError, AgentNotFoundError) as error:
                                    st.error(str(error))
                            if doc.revision > 1:
                                st.warning(
                                    "Knowledge memakai teks koreksi pengguna, bukan ekstraksi asli."
                                )
                                if st.checkbox(
                                    "Lihat teks parser awal", key=f"raw_parser_{doc.id}"
                                ):
                                    raw = services.documents.extraction(
                                        selected_id, doc.id, original=True
                                    )
                                    st.text(raw[page_index]["text"])
                    else:
                        st.info(
                            "Hasil ekstraksi belum tersimpan. Dokumen dari versi lama perlu "
                            "diunggah ulang untuk membuat pratinjau dan editor. "
                            "File yang gagal dibaca tidak mempunyai teks untuk ditinjau."
                        )
                with right:
                    if st.button("Hapus", key=f"delete_doc_{doc.id}"):
                        try:
                            services.documents.delete(selected_id, doc.id)
                            show_notice(
                                "Dokumen dihapus dan tidak lagi digunakan untuk jawaban baru."
                            )
                        except DocumentError as error:
                            st.error(str(error))
    else:
        st.info("Buat atau pilih agent dahulu untuk mengunggah dokumen.")

with chat:
    st.subheader("Percakapan")
    if selected_id:
        current = services.agents.get(selected_id)
        st.caption(f"{current.name} · {current.provider.value} · {current.model}")
        conversation_key = f"conversation_{selected_id}"
        if conversation_key not in st.session_state:
            st.session_state[conversation_key] = services.chat.latest_conversation(
                selected_id
            ) or services.chat.new_conversation(selected_id)
        if st.button("Percakapan baru", key=f"new_chat_{selected_id}"):
            st.session_state[conversation_key] = services.chat.new_conversation(selected_id)
            st.rerun()
        conversation_id = st.session_state[conversation_key]
        active_documents = services.documents.ready_versions(selected_id)
        if not active_documents:
            st.info("Belum ada dokumen siap. Agent akan menyatakan informasi belum tersedia.")

        def render_answer(message):
            st.markdown(message["content"])
            for citation in message["citations"]:
                active = active_documents.get(citation["document_id"]) == citation.get(
                    "revision", 1
                )
                location = citation.get("locator") or (
                    f"halaman PDF {citation['page']}" if citation.get("page") else "Bagian dokumen"
                )
                with st.expander(f"[{citation['tag']}] Sumber · {location}"):
                    st.write(citation["filename"])
                    if citation.get("revision", 1) > 1:
                        st.caption(f"Teks koreksi pengguna · revisi {citation['revision']}")
                    if not active:
                        st.warning(
                            "Dokumen sumber sudah dihapus, tidak aktif, atau telah dikoreksi. "
                            "Jawaban ini merupakan riwayat lama."
                        )
                    st.caption(
                        f"Kemiripan retrieval: {citation['score']:.3f} "
                        "(bukan tingkat kepastian jawaban)"
                    )
                    st.text(citation["text"])

        for message in services.chat.messages(selected_id, conversation_id):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_answer(message)
                else:
                    st.markdown(message["content"])
        if question := st.chat_input(
            "Tanyakan isi dokumen agent ini", max_chars=2000, key=f"question_{selected_id}"
        ):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Mencari sumber dan menyusun jawaban…"):
                try:
                    answer = services.chat.ask(selected_id, conversation_id, question)
                    render_answer(answer)
                except (ProviderError, ValueError) as error:
                    st.error(str(error))
                except Exception as error:
                    logger.error("chat_failed error_type=%s", type(error).__name__)
                    st.error(
                        "Jawaban belum dapat dibuat. Periksa embedding dan penyimpanan aplikasi."
                    )
    else:
        st.info("Pilih agent untuk memulai percakapan.")
