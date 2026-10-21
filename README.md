# Ruang Dokumen

Aplikasi web multi-agent document chatbot untuk take-home AI Engineer. Setiap agent dirancang mempunyai instruksi, provider/model, dan sumber dokumen sendiri.

**Status: fondasi aplikasi.** Buat, pilih, edit, dan hapus agent sudah berjalan. SQLite menyimpan konfigurasi agent dan skema dokumen/percakapan; Chroma menyimpan vektor secara persisten. Upload PDF/DOCX, ingestion, retrieval, chat bersitasi, dan tiga adapter provider belum diimplementasikan. Tab dokumen dan percakapan menjelaskan status ini.

## Menjalankan secara lokal

Prasyarat: [uv](https://docs.astral.sh/uv/) dan [Ollama](https://ollama.com/) terpasang; daemon Ollama berjalan. Jalankan dari root repository:

1. Pasang dependensi terkunci dan Python 3.12:
   ```sh
   uv sync --frozen --extra dev --python 3.12
   ```
2. Salin konfigurasi contoh:
   ```sh
   cp .env.example .env
   ```
   Sesuaikan bila perlu. Key cloud boleh kosong untuk jalur lokal. Jangan commit `.env`.
3. Siapkan model chat lokal (unduhan hanya diperlukan sekali):
   ```sh
   ollama pull llama3.1:8b
   ```
4. Unduh dan verifikasi model embedding:
   ```sh
   uv run python -m src.cli download-embedding
   ```
5. Jalankan aplikasi:
   ```sh
   uv run streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
   ```
   Buka [localhost:8501](http://localhost:8501). Penyimpanan dibuat saat aplikasi dibuka.

Model chat sekitar 4,9 GB; embedding memerlukan unduhan tambahan. UI pengelolaan agent dapat berjalan tanpa model; model diperlukan untuk tahap chat/retrieval berikutnya. Agent adalah konfigurasi asisten yang dibuat di UI, bukan paket yang diunduh.

## Pemeriksaan dan pengujian

```sh
uv run python -m src.cli init
uv run python -m src.cli doctor
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

`doctor` memeriksa Python/dependensi, integritas SQLite, foreign key, Chroma, cache embedding, koneksi Ollama, model yang tersedia, serta status Docker. Exit code 1 berarti prasyarat pengembangan lokal belum lengkap; key cloud dan Docker opsional untuk jalur native. Nilai key tidak dicetak. Deteksi cache membaca file model; `download-embedding` melakukan inference embedding sebenarnya. Probe Ollama pada doctor membaca daftar model, bukan menguji generasi.

Enam test fondasi mencakup persistensi agent, foreign key lintas agent, cascade dokumen/percakapan, retry penghapusan saat pembersihan vektor gagal, penghapusan vektor per agent, validasi konfigurasi, dan alur buat/edit/hapus melalui Streamlit AppTest. Test ingestion dan adapter provider masih harus ditambahkan untuk hasil akhir.

## Docker

Setelah `.env` tersedia dan Docker Desktop berjalan:

```sh
docker compose up --build -d
```

Compose memakai port lokal 8501 dan volume `app_data`. Hentikan server native dahulu jika port terpakai. Ollama berjalan di host; Compose memakai `host.docker.internal:11434`, dapat diganti dengan `DOCKER_OLLAMA_BASE_URL`. Cache model native tidak otomatis masuk ke volume Docker. Siapkan embedding di container dengan:

```sh
docker compose exec app /app/.venv/bin/python -m src.cli download-embedding
```

`docker compose down` mempertahankan volume. Hindari opsi `-v` bila data masih diperlukan. PyTorch CPU untuk Linux dikonfigurasi melalui [index resmi sesuai panduan uv](https://docs.astral.sh/uv/guides/integration/pytorch/), untuk menghindari unduhan CUDA pada demo ini. Validasi 22 Oktober 2026: build berhasil; container non-root berhasil menjalankan `src.cli init` (SQLite/Chroma ready), HTTP health check merespons `ok`, dan probe Ollama dari container menemukan model pilihan. Container uji dihentikan setelah pemeriksaan. Inference embedding di container belum diuji; inference native berhasil.

## Struktur dan keputusan teknis

```text
app.py                     UI Streamlit
src/config.py              Konfigurasi tervalidasi dari environment/.env
src/bootstrap.py           Perakitan service dan penyimpanan
src/database/connection.py  SQLite, schema, koneksi per operasi
src/services/agents.py     Operasi agent dan koordinasi penghapusan
src/knowledge/             Chroma dan embedding lokal
src/cli.py                 Init, doctor, unduhan/verifikasi embedding
tests/                     Pengujian fondasi dan interaksi UI
```

Streamlit mempercepat implementasi aplikasi web dalam tenggat singkat. Backend berupa service Python modular dalam proses yang sama; REST API terpisah belum dibuat. SQLite menghindari server database tambahan. Chroma menyimpan indeks dengan metadata agent; tahap retrieval nantinya wajib memfilter agent dan dokumen aktif.

Setiap koneksi SQLite mengaktifkan foreign key dan memakai transaksi sendiri. Penghapusan agent menonaktifkannya dahulu, membersihkan vektor, kemudian menghapus data relasional melalui cascade. Karena SQLite dan Chroma tidak berbagi transaksi, kegagalan pembersihan meninggalkan agent berstatus `deleting`, tersembunyi dari pilihan aktif, dan dapat dicoba ulang melalui UI.

Embedding memakai `intfloat/multilingual-e5-small`, 384 dimensi, normalisasi, dan prefix `query:`/`passage:`. Model multilingual dipilih untuk dokumen Indonesia dan berjalan lokal tanpa biaya API embedding. Embedding terpisah dari model chat. Mengubah model/dimensi membutuhkan collection baru dan reindex; aplikasi menolak konfigurasi yang tidak sesuai indeks tersimpan. Threshold similarity masih nilai awal, belum dikalibrasi.

## Batasan dan pekerjaan berikutnya

- Belum ada upload/parsing, retrieval, chat bersitasi, multi-turn, atau OCR.
- OpenAI, OpenRouter, dan LiteLLM saat ini tersimpan sebagai konfigurasi agent; adapter chat independen belum tersedia. Model lokal berhasil menjawab satu prompt melalui LiteLLM secara terpisah dari UI. OpenAI/OpenRouter belum diuji live.
- Skema mendukung percakapan, tetapi UI percakapan belum aktif. Isolasi retrieval dan penghapusan dokumen pada sesi lama masih harus diuji.
- Aplikasi lokal untuk satu pengguna; belum mempunyai autentikasi atau isolasi multi-user.
- Demo direncanakan memakai kebijakan publik Bahasa Indonesia PT Vale Indonesia Tbk. Aset demo dan evaluasi belum dimasukkan ke aplikasi.

## Bantuan AI

Codex/AI membantu perencanaan, implementasi, pengujian, dan dokumentasi. Masukan AI ditinjau terhadap brief; status implementasi dan validasi dijelaskan sesuai bukti. Catatan koordinasi pribadi, rahasia, cache model, serta data runtime tidak disertakan dalam repository.

