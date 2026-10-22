# Dataset contoh Bahasa Indonesia

Gunakan dua agent untuk satu perusahaan **simulasi PT XYZ**. Template tidak diklaim sebagai kebijakan perusahaan riil atau ketentuan hukum terkini.

| Agent | File | Peran |
|---|---|---|
| HR PT XYZ | `Contoh-PP-Sederhana-2023.pdf` (14 halaman) | Template peraturan perusahaan: cuti, upah, jam kerja |
| HR PT XYZ | `Permenaker 28-2014.pdf` (25 halaman) | Regulasi umum pendamping PP/PKB |
| Legal PT XYZ | `Permenaker 28-2014.pdf` | Prosedur regulasi PP/PKB |
| Legal PT XYZ | `uu40-2007.pdf` (140 halaman) | Organ perseroan, RUPS, direksi dan komisaris |
| OCR Jamkrida — contoh terpisah | `Scan-Surat24-07-2023-094710.pdf` (34 halaman) | Bonus demonstrasi OCR, tidak dicampur menjadi policy PT XYZ |

Template berasal dari [Disnakertrans Bantul](https://disnakertrans.bantulkab.go.id/storage/disnakertrans/menu/294/Contoh-PP-Sederhana-2023.pdf), 14 halaman dengan identitas PT XYZ. Regulasi tersedia dari [BPHN: Permenaker 28/2014](https://bphn.go.id/data/documents/14pm028.pdf), [metadata Kemnaker](https://jdih.kemnaker.go.id/peraturan/detail/579/peraturan-menteri-nomor-28-tahun-2014) dan [portal peraturan: UU 40/2007](https://peraturan.go.id/id/uu-no-40-tahun-2007). Salinan scan dipublikasikan [PT Jamkrida Jateng](https://jamkrida-jateng.co.id/wp-content/uploads/Scan-Surat24-07-2023-094710.pdf). Bukan verifikasi versi konsolidasi atau status keberlakuan hukum.

Binary tidak disertakan ke Git karena provenance/izin distribusi perlu dipertahankan. Aplikasi juga dapat dicoba dengan dokumen reviewer sendiri. Untuk menata salinan lokal yang sudah diperoleh:

```sh
uv run python -m scripts.prepare_demo_current --source-dir /path/to/pdf-folder
uv run python -m scripts.seed_demo --include-ocr
```

Gunakan nama `Permenaker 28-2014.pdf`, atau nama unduhan macOS `Permenaker 28:2014.pdf`. Hasil dan SHA256 tersimpan pada `data/demo-current/manifest.json`. Skrip tidak mengubah teks atau identitas perusahaan.

## Contoh pertanyaan

- HR: “Menurut Contoh PP Sederhana 2023, berapa hari cuti tahunan dan syarat masa kerjanya?” Sumber: halaman PDF6.
- Follow-up: “Kapan permohonan cuti tahunan tersebut diajukan?”
- Legal: “Menurut uu40-2007, apa saja organ Perseroan?” Sumber: halaman PDF2/3.
- Untuk memeriksa isolasi, tanyakan pertanyaan UU40 pada HR yang tidak memiliki file itu.

Aplikasi dapat digunakan dengan dokumen lain. Contoh ini tidak mengukur akurasi umum atau memastikan status hukum terbaru dokumen.
