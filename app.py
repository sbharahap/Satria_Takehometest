import logging
import sqlite3
from html import escape

import httpx
import streamlit as st
from pydantic import ValidationError

from src.bootstrap import initialize
from src.config import get_settings
from src.environment import ollama_status
from src.knowledge.parsing import DocumentError, normalize_pdf_text, pdf_page_preview
from src.models import AgentInput, ProviderName
from src.providers.base import ProviderError
from src.services.agents import AgentDeletionError, AgentNotFoundError
from src.services.uploads import UploadQueue

st.set_page_config(page_title="Tanya AI", page_icon="🤖", layout="wide")
logger = logging.getLogger(__name__)


@st.cache_resource
def resources():
    settings = get_settings()
    return settings, initialize(settings)


@st.cache_resource
def upload_queue():
    return UploadQueue()


@st.cache_data(ttl=30)
def local_status(base_url: str, model: str):
    from src.config import Settings

    return ollama_status(Settings(ollama_base_url=base_url, litellm_model=model))


@st.cache_data(ttl=30)
def installed_local_models(base_url: str) -> list[str]:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3, trust_env=False)
        response.raise_for_status()
        return [f"ollama_chat/{item['name']}" for item in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def show_notice(message: str) -> None:
    st.session_state["notice"] = message
    st.rerun()


try:
    settings, services = resources()
except Exception:
    st.title("Tanya AI")
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
active_ids = {agent.id for agent in agents}
for selector_key in ("documents_agent", "chat_agent"):
    if st.session_state.get(selector_key) not in active_ids:
        st.session_state.pop(selector_key, None)
with st.sidebar:
    st.markdown("### 🤖 Tanya AI")
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
        if st.button("Muat ulang konfigurasi", key="reload_config", disabled=upload_queue().busy()):
            get_settings.cache_clear()
            resources.clear()
            local_status.clear()
            st.rerun()

st.title(settings.app_name)
st.write("Kelola asisten dan sumber pengetahuannya dalam satu ruang kerja.")
if notice := st.session_state.pop("notice", None):
    st.success(notice)
manage, documents, chat = st.tabs(
    ["Kelola agent", "Dokumen", "Percakapan"], key="main_tabs", on_change="rerun"
)

DEFAULT_AGENT_INSTRUCTIONS = "Gunakan sumber aktif agent ini dan jawab dalam Bahasa Indonesia."


@st.fragment(run_every=0.5 if upload_queue().busy() else None)
def upload_panel(agent_id):
    job_key = f"upload_job_{agent_id}"
    receipt_key = f"upload_receipt_{agent_id}"
    job_id = st.session_state.get(job_key)
    result = upload_queue().get(job_id, agent_id) if job_id else None
    busy = result == "processing"
    if result is not None and not busy and st.session_state.get(receipt_key) != job_id:
        st.session_state[receipt_key] = job_id
        st.rerun()
    phase = (
        "processing"
        if busy
        else ("error" if result and result.errors else "done" if result else "idle")
    )
    if st.session_state.get(f"upload_files_{agent_id}") and not busy:
        phase = "idle"
    labels = {
        "idle": "Proses dokumen",
        "processing": "⏳ Memproses dokumen…",
        "done": "✅ SELESAI",
        "error": "⚠️ Ada dokumen gagal · coba lagi",
    }
    st.markdown(
        """<style>
    @keyframes uploadPulse { 0%,100% {background:#2563eb} 50% {background:#7c3aed} }
    .st-key-upload_action_processing button[kind="primaryFormSubmit"] {
        animation:uploadPulse 1.3s ease-in-out infinite; color:white; opacity:1; }
    .st-key-upload_action_done button[kind="primaryFormSubmit"] {
        background:#15803d; color:white; border-color:#15803d; }
    .st-key-upload_action_error button[kind="primaryFormSubmit"] {
        background:#b91c1c; color:white; }
    </style>""",
        unsafe_allow_html=True,
    )
    with st.container(key=f"upload_action_{phase}"):
        with st.form(f"upload_{agent_id}", clear_on_submit=True):
            uploads = st.file_uploader(
                "Pilih PDF atau DOCX",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                max_upload_size=settings.max_upload_mb,
                disabled=busy,
                key=f"upload_files_{agent_id}",
            )
            submitted = st.form_submit_button(
                labels[phase],
                type="primary",
                disabled=busy,
                key=f"upload_submit_{agent_id}",
            )
    if submitted:
        if not uploads:
            st.warning("Pilih minimal satu file PDF atau DOCX.")
        else:
            try:
                st.session_state[job_key] = upload_queue().start(
                    agent_id, [(f.name, f.getvalue()) for f in uploads], services.documents
                )
                st.rerun()
            except DocumentError as error:
                st.error(str(error))
    if result and not busy:
        for name in result.ready:
            st.caption(f"✅ {name} siap digunakan")
        for message in result.errors:
            st.error(message)


def agent_fields(key: str, current=None):
    name = st.text_input("Nama agent", value=current.name if current else "", key=f"{key}_name")
    description = st.text_input(
        "Deskripsi", value=current.description if current else "", key=f"{key}_description"
    )
    instructions = st.text_area(
        "Instruksi tambahan (opsional)",
        height=130,
        value=current.instructions if current else "",
        placeholder="Contoh: Utamakan kebijakan keselamatan kerja.",
        help=(
            "Aturan dasar grounding, sitasi, bahasa, dan pengakuan jika bukti kurang selalu aktif."
        ),
        key=f"{key}_instructions",
    )
    initial = current.provider.value if current else settings.default_chat_provider
    provider = st.selectbox(
        "Provider",
        options=providers,
        index=None
        if f"{key}_provider" in st.session_state
        else providers.index(ProviderName(initial)),
        format_func=lambda value: provider_labels[value.value],
        key=f"{key}_provider",
    )
    previous_provider = st.session_state.get(f"{key}_provider_seen", provider.value)
    if previous_provider != provider.value:
        st.session_state[f"{key}_model"] = default_models[provider.value]
    st.session_state[f"{key}_provider_seen"] = provider.value
    if provider.value == "litellm":
        choices = list(
            dict.fromkeys(
                [
                    *installed_local_models(settings.ollama_base_url),
                    settings.litellm_model,
                    settings.local_fallback_model,
                ]
            )
        )
    elif provider.value == "openrouter":
        choices = list(dict.fromkeys([settings.openrouter_model, "openrouter/free"]))
    else:
        choices = [settings.openai_model]
    if current and current.provider == provider and current.model not in choices:
        choices.insert(0, current.model)
    selected_model = st.session_state.get(
        f"{key}_model", current.model if current else default_models[initial]
    )
    if selected_model not in choices:
        selected_model = default_models[provider.value]
        st.session_state[f"{key}_model"] = selected_model
    model = st.selectbox(
        "Model",
        options=choices,
        index=None if f"{key}_model" in st.session_state else choices.index(selected_model),
        key=f"{key}_model",
        help="Model lokal ditampilkan dari model yang terpasang di Ollama.",
    )
    st.caption("Model bawaan: " + " · ".join(f"{p}: {m}" for p, m in default_models.items()))
    return name, description, instructions, provider, model


def validated_agent(values):
    name, description, instructions, provider, model = values
    try:
        return AgentInput(
            name=name,
            description=description,
            instructions=instructions.strip() or DEFAULT_AGENT_INSTRUCTIONS,
            provider=provider,
            model=model,
        )
    except ValidationError:
        st.error("Isi nama, instruksi, dan ID model yang valid. Nama maksimal 100 karakter.")
        return None


with manage:
    st.subheader("Agent Anda")
    st.caption("Contoh: HR PT XYZ dan Legal PT XYZ (perusahaan simulasi).")
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
        # Deleting a rendered widget's key lets its browser value repopulate state.
        # Assign defaults explicitly before rendering the same widgets again.
        for field in ("name", "description", "instructions"):
            st.session_state[f"create_{field}"] = ""
        st.session_state["create_provider"] = ProviderName(settings.default_chat_provider)
        st.session_state["create_model"] = default_models[settings.default_chat_provider]
        st.session_state["create_provider_seen"] = settings.default_chat_provider
    just_saved = st.session_state.pop("_create_saved", False)
    with st.expander("Buat agent", expanded=not agents or just_saved):
        if just_saved:
            st.success("Agent berhasil disimpan. Form siap untuk agent berikutnya.")
        values = agent_fields("create")
        if st.button("Simpan agent", type="primary"):
            if payload := validated_agent(values):
                try:
                    services.agents.create(payload)
                    st.session_state["_reset_create"] = True
                    st.session_state["_create_saved"] = True
                    show_notice("✅ Tersimpan: Agent berhasil dibuat.")
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
            delete_key = "delete_agent_target"
            if st.session_state.get(delete_key) not in {a.id for a in agents}:
                st.session_state[delete_key] = selected_id
            delete_id = st.selectbox(
                "Pilih agent yang akan dihapus",
                options=[a.id for a in agents],
                format_func=lambda value: next(a.name for a in agents if a.id == value),
                key=delete_key,
            )
            delete_target = services.agents.get(delete_id)
            document_count = len(services.documents.list_documents(delete_id))
            st.warning(
                f"Agent yang akan dihapus: “{delete_target.name}”. "
                f"{document_count} dokumen dan seluruh percakapannya akan ikut dihapus."
            )
            confirmed = st.checkbox(
                f"Saya ingin menghapus “{delete_target.name}” beserta seluruh datanya",
                key=f"confirm_{delete_id}",
            )
            if st.button(f"Hapus {delete_target.name}", disabled=not confirmed):
                try:
                    services.agents.delete(delete_id)
                    show_notice(f"Agent “{delete_target.name}” dan data terkait telah dihapus.")
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
    if agents:
        document_selected_id = st.selectbox(
            "Pilih agent untuk dokumen",
            options=[a.id for a in agents],
            index=next((i for i, a in enumerate(agents) if a.id == selected_id), 0),
            format_func=lambda value: next(a.name for a in agents if a.id == value),
            key="documents_agent",
        )
    else:
        document_selected_id = None
    if document_selected_id:
        st.caption(f"Agent: {services.agents.get(document_selected_id).name}")
        st.caption(
            "PDF atau DOCX dengan teks, maksimal "
            f"{settings.max_upload_mb} MB/file. "
            f"OCR lokal: {'aktif' if settings.ocr_enabled else 'nonaktif'}, "
            f"maksimal {settings.ocr_max_pages} halaman scan."
        )
        upload_panel(document_selected_id)
        uploaded_docs = services.documents.list_documents(document_selected_id)
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
                    pages = services.documents.extraction(document_selected_id, doc.id)
                    if pages:
                        st.caption(
                            "Bandingkan dengan dokumen asli, lalu koreksi kesalahan ekstraksi. "
                            "Simpan akan mengganti knowledge dengan teks koreksi "
                            "dan mengindeks ulang. "
                            "PDF asli tetap disimpan tanpa perubahan. "
                            "Periksa kembali hasil OCR sebelum digunakan sebagai knowledge."
                        )
                        page_key = f"extraction_page_{doc.id}"
                        if st.session_state.get(page_key, 0) >= len(pages):
                            st.session_state[page_key] = 0
                        current_index = st.session_state.get(page_key, 0)
                        st.caption(
                            f"{len(pages)} halaman/bagian teks berhasil diekstrak. "
                            "Pratinjau dan editor menampilkan satu halaman/bagian yang dipilih."
                        )

                        def change_extraction_page(key, index):
                            st.session_state[key] = index

                        previous_page, page_position, next_page = st.columns([1, 2, 1])
                        previous_page.button(
                            "← Sebelumnya",
                            key=f"previous_page_{doc.id}",
                            disabled=current_index == 0,
                            on_click=change_extraction_page,
                            args=(page_key, current_index - 1),
                        )
                        page_position.caption(f"Bagian {current_index + 1} dari {len(pages)}")
                        next_page.button(
                            "Berikutnya →",
                            key=f"next_page_{doc.id}",
                            disabled=current_index == len(pages) - 1,
                            on_click=change_extraction_page,
                            args=(page_key, current_index + 1),
                        )
                        page_index = st.selectbox(
                            "Halaman atau bagian",
                            options=range(len(pages)),
                            format_func=lambda i, pages=pages: (
                                pages[i].get("locator") or f"Halaman PDF {pages[i]['page']}"
                            ),
                            key=page_key,
                        )
                        selected_page = pages[page_index]
                        original = services.documents.original_file(document_selected_id, doc.id)
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
                                    st.caption(
                                        f"{selected_page.get('locator') or 'Naskah DOCX'}. "
                                        "Ekstraksi awal DOCX, bukan tata letak Word. "
                                        "Unduh file asli untuk memeriksa formatnya."
                                    )
                                    initial_pages = services.documents.extraction(
                                        document_selected_id, doc.id, original=True
                                    )
                                    docx_text = (
                                        initial_pages[page_index]["text"]
                                        if page_index < len(initial_pages)
                                        else "Ekstraksi awal tidak tersedia. Unduh file asli."
                                    )
                                    box_style = (
                                        "background-color: white; color: #1e293b; "
                                        "padding: 20px 24px; border-radius: 6px; "
                                        "border: 1px solid #cbd5e1; height: 480px; "
                                        "overflow-y: auto; font-family: 'Times New Roman', serif; "
                                        "font-size: 15px; line-height: 1.6; white-space: pre-wrap; "
                                        "box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);"
                                    )
                                    st.markdown(
                                        f'<div style="{box_style}">\n{escape(docx_text)}\n</div>',
                                        unsafe_allow_html=True,
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
                            st.caption(
                                "Periksa teks hasil ekstraksi terhadap dokumen asli. "
                                "Simpan tiap halaman sebelum berpindah."
                            )
                            editor_key = f"extraction_text_{doc.id}_{doc.revision}_{page_index}"
                            pending_key = f"pending_text_{editor_key}"
                            just_normalized = False
                            if pending_key in st.session_state:
                                st.session_state[editor_key] = st.session_state.pop(pending_key)
                                just_normalized = True
                            elif editor_key not in st.session_state:
                                st.session_state[editor_key] = selected_page["text"]

                            if just_normalized:
                                st.info(
                                    st.session_state.pop(
                                        f"normalize_notice_{editor_key}",
                                        "Baris telah dirapikan. "
                                        "Simpan untuk menerapkan ke knowledge.",
                                    )
                                )

                            with st.form(f"edit_text_{doc.id}_{doc.revision}_{page_index}"):
                                edited_text = st.text_area(
                                    "Teks hasil ekstraksi",
                                    height=480,
                                    key=editor_key,
                                    disabled=doc.status == "processing",
                                )
                                col_save, col_norm = st.columns([2, 1])
                                save_text = col_save.form_submit_button(
                                    "Simpan koreksi dan indeks ulang",
                                    type="primary",
                                    disabled=doc.status == "processing",
                                )
                                normalize_clicked = col_norm.form_submit_button(
                                    "Rapikan baris",
                                    disabled=doc.status == "processing",
                                    help=(
                                        "Gabungkan baris kalimat dan spasi berlebih. Parser sudah "
                                        "melakukannya saat upload, jadi teks dapat tetap sama. "
                                        "Tidak memperbaiki salah baca OCR atau susunan tabel."
                                    ),
                                )
                            if normalize_clicked:
                                draft = edited_text
                                normalized = normalize_pdf_text(draft)
                                if normalized == draft:
                                    st.info(
                                        "Tidak ada perubahan: teks sudah rapi menurut aturan "
                                        "spasi dan baris. Isi OCR tetap perlu diperiksa."
                                    )
                                else:
                                    st.session_state[f"normalize_undo_{editor_key}"] = draft
                                    st.session_state[pending_key] = normalized
                                    before = draft.count("\n") + 1
                                    after = normalized.count("\n") + 1
                                    st.session_state[f"normalize_notice_{editor_key}"] = (
                                        f"Perapian mengubah {before} baris menjadi {after} baris "
                                        "dan menormalkan spasi. Belum disimpan ke knowledge. "
                                        "Periksa hasilnya, lalu simpan atau urungkan."
                                    )
                                    st.rerun()
                            undo_key = f"normalize_undo_{editor_key}"
                            if undo_key in st.session_state and st.button(
                                "Urungkan perapian terakhir", key=f"undo_{editor_key}"
                            ):
                                st.session_state[pending_key] = st.session_state.pop(undo_key)
                                st.session_state[f"normalize_notice_{editor_key}"] = (
                                    "Perapian diurungkan. Draft belum disimpan ke knowledge."
                                )
                                st.rerun()
                            st.caption("Belum disimpan sampai tombol simpan ditekan.")
                            if save_text:
                                try:
                                    with st.spinner("Menyimpan koreksi dan mengindeks ulang…"):
                                        updated = services.documents.edit_page(
                                            document_selected_id,
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
                                        document_selected_id, doc.id, original=True
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
                            services.documents.delete(document_selected_id, doc.id)
                            show_notice(
                                "Dokumen dihapus dan tidak lagi digunakan untuk jawaban baru."
                            )
                        except DocumentError as error:
                            st.error(str(error))
    else:
        st.info("Buat atau pilih agent dahulu untuk mengunggah dokumen.")

with chat:
    st.subheader("Percakapan")
    if agents:
        chat_selected_id = st.selectbox(
            "Pilih agent untuk percakapan",
            options=[a.id for a in agents],
            index=next((i for i, a in enumerate(agents) if a.id == selected_id), 0),
            format_func=lambda value: next(a.name for a in agents if a.id == value),
            key="chat_agent",
        )
    else:
        chat_selected_id = None
    if chat_selected_id:
        current = services.agents.get(chat_selected_id)
        st.caption(f"{current.name} · {current.provider.value} · {current.model}")
        conversation_key = f"conversation_{chat_selected_id}"
        if conversation_key not in st.session_state:
            st.session_state[conversation_key] = services.chat.latest_conversation(
                chat_selected_id
            ) or services.chat.new_conversation(chat_selected_id)
        history_key = f"history_{chat_selected_id}"
        if st.button("➕ Buat Percakapan Baru", key=f"new_chat_{chat_selected_id}", type="primary"):
            new_id = services.chat.new_conversation(chat_selected_id)
            st.session_state[conversation_key] = new_id
            st.session_state[history_key] = new_id
            st.rerun()
        conversations = services.chat.conversations(chat_selected_id)
        by_id = {item["id"]: item for item in conversations}
        if st.session_state.get(history_key) not in by_id:
            st.session_state[history_key] = st.session_state[conversation_key]
        col_history, col_mode, col_del_conv = st.columns([4, 3, 1])
        with col_history:
            conversation_id = st.selectbox(
                "Riwayat percakapan",
                options=list(by_id),
                key=history_key,
                format_func=lambda value: (
                    by_id[value]["title"]
                    + " · "
                    + by_id[value]["created_at"][:16].replace("T", " ")
                    + " UTC"
                ),
            )
        with col_mode:
            reasoning_mode = st.radio(
                "Mode jawaban",
                options=["fast", "thinking"],
                format_func=lambda mode: (
                    "Fast" if mode == "fast" else "Thinking · teliti"
                ),
                horizontal=True,
                key=f"reasoning_mode_{chat_selected_id}",
                help=(
                    "Thinking meminta model meluangkan lebih banyak token untuk memilih bukti. "
                    "Teks penalaran internal tidak ditampilkan."
                ),
            )
        with col_del_conv:
            st.write("")
            st.write("")
            if st.button(
                "🗑️ Hapus",
                key=f"del_conv_{chat_selected_id}",
                disabled=not by_id,
                help="Hapus riwayat percakapan yang dipilih",
            ):
                if conversation_id:
                    services.chat.delete_conversation(chat_selected_id, conversation_id)
                    st.session_state.pop(conversation_key, None)
                    st.session_state.pop(history_key, None)
                    show_notice("Percakapan berhasil dihapus.")
        st.session_state[conversation_key] = conversation_id
        active_documents = services.documents.ready_versions(chat_selected_id)
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

        chat_container = st.container()
        with chat_container:
            for message in services.chat.messages(chat_selected_id, conversation_id):
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        render_answer(message)
                    else:
                        st.markdown(message["content"])

        if question := st.chat_input(
            "Tanyakan isi dokumen agent ini", max_chars=2000, key=f"question_{chat_selected_id}"
        ):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                with (
                    st.chat_message("assistant"),
                    st.spinner("Mencari sumber dan menyusun jawaban…"),
                ):
                    try:
                        services.chat.ask(
                            chat_selected_id,
                            conversation_id,
                            question,
                            reasoning_mode=reasoning_mode,
                        )
                        st.rerun()
                    except (ProviderError, ValueError) as error:
                        st.error(str(error))
                    except Exception as error:
                        logger.error("chat_failed error_type=%s", type(error).__name__)
                        st.error(
                            "Jawaban belum dapat dibuat. "
                            "Periksa embedding dan penyimpanan aplikasi."
                        )
    else:
        st.info("Pilih agent untuk memulai percakapan.")
