# Pemeriksaan integrasi final

Tanggal: 22 Oktober 2026. Model lokal Qwen3:8b melalui LiteLLM/Ollama; embedding multilingual E5-small. Ini smoke acceptance terarah, bukan benchmark akurasi umum. Tidak ada sweep model atau panggilan cloud dalam pemeriksaan dataset baru.

| Pemeriksaan | Hasil dan batas bukti |
|---|---|
| Suite otomatis | 111 pytest lulus: mock provider, ingestion, OCR failure/locator, CRUD, isolasi, revisi, antrean dan AppTest |
| HR template: cuti + masa kerja | Live PASS: 12 hari kerja, setelah 12 bulan, sitasi PDF halaman 6 |
| Follow-up waktu pengajuan cuti | Live PASS: seminggu sebelumnya, retrieval baru |
| Tanggal pembayaran upah | Live PASS: tanggal 5; guard nominal tidak lagi menyalahartikan pertanyaan tanggal sebagai jumlah uang |
| Legal: organ perseroan | Awal gagal: threshold 0,83 menyaring halaman yang tepat. Setelah 0,82, model memilih empat ID sah tetapi batas awal tiga menolak respons. Batas lima bukti literal memperbaiki kasus; recheck live PASS dengan sitasi halaman 2/3 |
| UU40 ditanyakan pada HR | Live PASS: unknown tanpa sitasi; dokumen tidak berada di namespace HR |
| Nomor telepon pribadi yang tidak tersedia | Live PASS: unknown tanpa sitasi palsu |
| Template HR dihapus, regulasi tetap ada | Live PASS: pertanyaan lama tidak dapat memakai dokumen terhapus; satu dokumen HR tetap ready |
| Riwayat setelah service dibuat ulang | PASS: pesan SQLite tetap terbaca; jawaban HR bersitasi dan riwayat setelah reload juga diperiksa melalui browser aplikasi utama |
| Scan Jamkrida lengkap | OCR lokal PASS: 34 halaman menghasilkan sekitar 62 ribu karakter; proses ekstraksi sekitar 43 detik pada mesin uji. Angka ini bukan SLA perangkat lain. Tabel/list masih memiliki artefak dan memerlukan review |
| Upload melalui Chrome headless | PASS: file 18,3 MB diunggah melalui file input; tombol disabled saat processing, CSS animation `uploadPulse`, selesai hijau; tab Dokumen tetap aktif |

Model dapat memilih kutipan berlebihan atau tidak relevan meskipun provenance valid. Hasil kasus terpilih tidak membuktikan semua pertanyaan, hasil OCR, atau dokumen lain benar. Threshold 0,82 tetap perlu dikalibrasi untuk corpus berbeda.

| Browser editor OCR | PASS: halaman 14 dipilih, teks OCR tersedia di editor, tombol rapikan dan simpan menghasilkan dokumen ready revisi 2. Canvas PDF asli ter-render dan screenshot berdampingan diinspeksi |
| Docker final dari data kosong | PASS: build dependensi terkunci, proses non-root, SQLite/Chroma baru, unduhan embedding dari Hub tanpa cache host, inference 384 dimensi |
| Doctor di container | PASS: integritas SQLite ok, foreign key aktif, schema 3, model Ollama terjangkau dan OCR ind+eng tersedia; tanpa key cloud |
| OCR → RAG di container | PASS: cuplikan satu halaman scan (halaman 14 sumber) di-OCR oleh language pack container, diindeks, lalu Qwen3 menjawab cuti 12 hari dengan sitasi. Nomor halaman cuplikan adalah 1 dan bukan diklaim sebagai nomor PDF lengkap |

Browser diuji dengan Playwright/Chrome headless pada storage terisolasi; bukan rekaman pengujian manusia. Docker memakai Ollama di host, sehingga pengujian ini tidak membuktikan server chat berjalan sepenuhnya di container. File PDF demo dipasang read-only untuk pengujian; runtime/embedding dibuat dari kosong. Native clean clone tidak diulang terpisah.
