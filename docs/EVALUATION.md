# Evaluasi jawaban pada dokumen HR dan Legal

Tanggal evaluasi: 22 Oktober 2026.

Evaluasi ini menjawab tiga pertanyaan: apakah sumber yang ditemukan relevan, apakah jawaban didukung sumber, dan apakah jawaban menjawab pertanyaan pengguna. Fokusnya kualitas hasil RAG pada dokumen demo, bukan jumlah unit test atau peringkat model secara umum.

## Dataset dan cara menjalankan

- **20 pertanyaan Bahasa Indonesia:** 10 HR dan 10 Legal. Ada 16 pertanyaan dengan referensi jawaban, dua follow-up HR, dua pertanyaan informasi yang tidak tersedia, dan dua pertanyaan lintas agent yang seharusnya ditolak. Follow-up termasuk dalam 16 pertanyaan, bukan tambahan di luar 20.
- **HR:** template `Contoh-PP-Sederhana-2023.pdf` dan `Permenaker 28-2014.pdf`.
- **Legal:** `Permenaker 28-2014.pdf` dan `uu40-2007.pdf`.
- **Generator:** Qwen3:8b melalui LiteLLM/Ollama, mode Fast dan extractive. Setiap kasus mendapat percakapan sendiri, kecuali dua rangkaian follow-up yang sengaja memakai riwayat.
- **Judge:** Llama3.1:8b di Ollama, temperature 0 dan seed 7. Judge menerima pertanyaan, query hasil rewrite, jawaban, dan seluruh konteks yang benar-benar diberikan kepada generator. Tidak memakai pengetahuan internet atau referensi jawaban sebagai input penilaian.
- **Retrieval:** E5-small 384 dimensi; chunk 350 token, overlap 50; threshold 0,82; top-k 4; reranking TF-IDF karakter 3-5.

Proses berjalan di storage terisolasi dan hanya memakai model lokal. Tidak membaca riwayat pengguna atau memanggil API cloud. Dokumen dan asalnya dijelaskan pada [panduan dataset](DEMO_DATASET.md).

```sh
ollama pull llama3.1:8b
uv run python -m scripts.evaluate_rag --run
```

Siapkan file sesuai panduan dataset lebih dahulu. Perintah di atas membutuhkan waktu dan resource model; tidak dijalankan saat aplikasi dibuka. `--resume-judge` melanjutkan penilaian dari jawaban yang sudah disimpan tanpa membuat jawaban ulang. Pastikan konfigurasi model dan corpus sesuai saat membandingkan run.

Semua [pertanyaan dan referensi](../eval/rag_cases.json), [hasil lengkap](../eval/rag_results.json), dan [kode evaluasi](../scripts/evaluate_rag.py) tersedia. Hasil memuat konteks, sitasi, alasan judge, hash dokumen, hash source aplikasi, model digest, serta konfigurasi. Referensi disusun dari PDF sumber; belum merupakan anotasi independen beberapa penilai manusia.

## Rubrik

Metrik mengikuti gagasan [RAG Triad dari TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/). Implementasinya memakai prompt dan validator sendiri, bukan SDK TruLens atau skor resmi RAGAS.

| Metrik | Yang dinilai | Skala 0-4 |
|---|---|---|
| Answer relevance | Kesesuaian jawaban dengan kebutuhan pertanyaan | 0: tidak menjawab; 1: hampir tidak menjawab; 2: sebagian; 3: langsung dengan kekurangan kecil; 4: langsung dan lengkap |
| Groundedness | Dukungan konteks terhadap klaim faktual jawaban | 0: tidak didukung/bertentangan; 1: sebagian besar tidak didukung; 2: campuran; 3: hampir semua didukung; 4: semua didukung |
| Context relevance | Relevansi masing-masing sumber yang diambil terhadap pertanyaan | 0: tidak terkait; 1: topik sama tanpa bukti jawaban; 2: sedikit bukti/banyak pengalih; 3: bukti dengan sedikit pengalih; 4: bukti fokus menjawab |

Skor dibagi 4 agar berada pada rentang 0-1. Context relevance dirata-ratakan per pertanyaan lebih dahulu, lalu antarpertanyaan. Pertanyaan dengan empat konteks tidak otomatis berbobot empat kali pertanyaan dengan satu konteks.

Jawaban yang hanya abstain diberi groundedness **N/A**, bukan 1. Pertanyaan tanpa konteks diberi context relevance **0** sebagai konvensi laporan. Output judge yang tidak lolos pemeriksaan nilai dan ID tidak diberi skor perkiraan. Karena itu setiap angka mencantumkan denominator `n`.

## Hasil

**Angka berikut adalah penilaian judge pada sampel ini, bukan persentase akurasi.** Tabel pertama mencakup semua kasus dengan skor valid, termasuk unknown yang memang diharapkan.

| Kelompok | Answer relevance | Groundedness | Context relevance |
|---|---|---|---|
| Gabungan | 0.588 (n=20) | 1.000 (n=15) | 0.328 (n=20) |
| HR | 0.625 (n=10) | 1.000 (n=8) | 0.367 (n=10) |
| Legal | 0.550 (n=10) | 1.000 (n=7) | 0.290 (n=10) |

Untuk memisahkan kualitas menjawab dari kemampuan menolak, berikut hasil hanya pada pertanyaan yang memiliki referensi jawaban dalam corpus agent:

| Kelompok | Answer relevance | Groundedness | Context relevance |
|---|---|---|---|
| Gabungan | 0.703 (n=16) | 1.000 (n=15) | 0.348 (n=16) |
| HR | 0.750 (n=8) | 1.000 (n=8) | 0.333 (n=8) |
| Legal | 0.656 (n=8) | 1.000 (n=7) | 0.362 (n=8) |

Unknown yang memang diharapkan: **4/4** menolak tanpa sitasi. Ada **1 pertanyaan dengan referensi yang ikut ditolak**: LG06. Judge menghasilkan skor valid pada 20/20 kasus.

### Apa yang terlihat dari hasilnya?

- **Kutipan literal menjaga asal fakta, tetapi tidak menyelesaikan retrieval.** Groundedness pada jawaban faktual dapat tinggi sementara context relevance jauh lebih rendah karena sumber yang dimasukkan juga berisi bagian yang tidak menjawab.
- **LG06 gagal pada retrieval:** definisi PKWT tersedia di Permenaker halaman PDF 3, tetapi konteks akhir berasal dari halaman 25, 8, 9, dan 15. Model memilih abstain. Tidak mengarang definisi dari luar konteks adalah perilaku yang baik; gagal membawa halaman yang diperlukan tetap merupakan kekurangan aplikasi.
- **Follow-up HR02 dan HR06** menghasilkan jawaban dengan sumber aktif. Ini bukti dua contoh follow-up, bukan semua bentuk multi-turn.
- **Judge perlu dibaca kritis.** Pada HR01, alasan judge menyebut jawaban langsung dan lengkap tetapi memberinya 3/4. Skor tersebut tetap disimpan apa adanya. Judge juga memberi relevance rendah pada beberapa penolakan yang justru benar menurut skenario isolasi/unknown; itulah sebabnya hasil penolakan dilaporkan terpisah.

Format penilaian pertama memakai array ID konteks dan menghasilkan 7 output yang ditolak validator. Setelah kontrak output diperketat menjadi object dengan key ID yang diwajibkan, seluruh jawaban tersimpan dinilai ulang; budget output judge naik dari 1.100 menjadi 1.600 token. Tidak ada generation ulang, pergantian model, atau pemilihan skor tertinggi antarpercobaan. Output/galat percobaan pertama tetap ada pada `judge_attempts` di hasil JSON; skor utama memakai protokol kedua yang sama untuk seluruh kasus.

## Pertanyaan dan skor per kasus

AR = answer relevance, G = groundedness, CR = context relevance. Nilai N/A tidak dimasukkan ke rata-rata metrik tersebut.

| ID | Pertanyaan | AR | G | CR |
|---|---|---:|---:|---:|
| HR01 | Menurut Contoh PP Sederhana 2023, berapa hari cuti tahunan dan syarat masa kerjanya? | 0.750 | 1.000 | 0.438 |
| HR02 | Kapan permohonan cuti tahunan tersebut harus diajukan? | 0.750 | 1.000 | 0.333 |
| HR03 | Menurut Contoh PP Sederhana 2023, pada tanggal berapa upah bulanan dibayarkan? | 0.750 | 1.000 | 0.333 |
| HR04 | Menurut Contoh PP Sederhana 2023, berapa hari kerja per minggu dan berapa jam kerja maksimalnya? | 0.750 | 1.000 | 0.333 |
| HR05 | Menurut Contoh PP Sederhana 2023, berapa lama masa percobaan pekerja tetap? | 0.750 | 1.000 | 0.312 |
| HR06 | Kalau yang kontrak, apakah juga menjalani masa percobaan? | 0.750 | 1.000 | 0.333 |
| HR07 | Menurut Contoh PP Sederhana 2023, apa dua program BPJS yang diikuti perusahaan? | 0.750 | 1.000 | 0.333 |
| HR08 | Menurut Contoh PP Sederhana 2023, apa fasilitas bagi karyawan wanita yang memberikan ASI eksklusif? | 0.750 | 1.000 | 0.250 |
| HR09 | Berapa nomor WhatsApp pribadi kepala HR PT XYZ? | 0.250 | N/A | 0.000 |
| HR10 | Menurut uu40-2007.pdf, apa saja organ Perseroan? | 0.000 | N/A | 1.000 |
| LG01 | Menurut uu40-2007.pdf, apa saja organ Perseroan? | 0.750 | 1.000 | 0.562 |
| LG02 | Menurut uu40-2007.pdf, apa kewenangan dan tanggung jawab Direksi? | 0.750 | 1.000 | 0.312 |
| LG03 | Menurut uu40-2007.pdf, apa tugas Dewan Komisaris? | 0.750 | 1.000 | 0.500 |
| LG04 | Menurut uu40-2007.pdf, apa yang dimaksud dengan RUPS? | 0.750 | 1.000 | 0.500 |
| LG05 | Menurut uu40-2007.pdf, apa yang dimaksud dengan Tanggung Jawab Sosial dan Lingkungan? | 0.750 | 1.000 | 0.333 |
| LG06 | Menurut Permenaker 28-2014.pdf, apa definisi PKWT? | 0.000 | N/A | 0.188 |
| LG07 | Menurut Permenaker 28-2014.pdf Pasal 29, berapa lama masa berlaku PKB? | 0.750 | 1.000 | 0.250 |
| LG08 | Menurut Permenaker 28-2014.pdf, apa saja yang sekurang-kurangnya dimuat dalam Peraturan Perusahaan? | 0.750 | 1.000 | 0.250 |
| LG09 | Berapa nomor WhatsApp pribadi direktur PT XYZ? | 0.250 | N/A | 0.000 |
| LG10 | Menurut Contoh-PP-Sederhana-2023.pdf, berapa hari cuti tahunan karyawan? | 0.000 | N/A | 0.000 |

## Jawaban, referensi, dan alasan penilaian

Bagian ini dapat dibuka per kasus; seluruh konteks lengkap tetap ada di JSON.

<details>
<summary>HR01 - Menurut Contoh PP Sederhana 2023, berapa hari cuti tahunan dan syarat masa kerjanya?</summary>

**Acuan:** 12 hari kerja setelah bekerja 12 bulan berturut-turut. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 6.

**Jawaban aplikasi:**

Setiap pekerja yang telah bekerja selama 12 bulan berturut -turut, berhak atas cuti tahunan selama 12 hari kerja dengan mendapat uang penuh. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 6; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 13; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 7; S4: `Contoh-PP-Sederhana-2023.pdf` hlm. 3.

**Alasan judge untuk jawaban:** Jawaban langsung dan lengkap tentang cuti tahunan dan syarat masa kerjanya dapat ditemukan di Pasal 11 CUTI TAHUNAN dari konteks S1.

**Alasan groundedness:** Pasal 11 CUTI TAHUNAN dari konteks S1 mendeskripsikan bahwa setiap pekerja yang telah bekerja selama 12 bulan berturut-turut, berhak atas cuti tahunan selama 12 hari kerja dengan mendapat uang penuh.

**Alasan per konteks:**

- S1: 4/4 - Jawaban langsung dan lengkap tentang cuti tahunan dan syarat masa kerjanya.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang cuti tahunan dan syarat masa kerjanya.
- S3: 1/4 - Mengandung sedikit bukti tentang cuti tahunan, tetapi banyak pengalih.
- S4: 2/4 - Sebagian kebutuhan terjawab, tetapi tidak lengkap dan ada informasi pengalih.

</details>

<details>
<summary>HR02 - Kapan permohonan cuti tahunan tersebut harus diajukan?</summary>

**Acuan:** Seminggu sebelumnya, kepada atasan. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 6.

**Jawaban aplikasi:**

Bagi pekerja yang akan menggunakan cuti tahunannya, seminggu sebelumnya telah mengajukan permohonan kepada pimpinan perusahaan. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 6; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 7; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 5.

**Alasan judge untuk jawaban:** Jawaban didukung oleh Pasal 11 CUTI TAHUNAN, ayat 3 yang menjelaskan bahwa bagi pekerja yang akan menggunakan cuti tahunannya, seminggu sebelumnya telah mengajukan permohonan kepada pimpinan perusahaan.

**Alasan groundedness:** Pasal 11 CUTI TAHUNAN, ayat 3 merupakan sumber hukum yang sah dan relevan dengan pertanyaan.

**Alasan per konteks:**

- S1: 4/4 - Pasal 11 CUTI TAHUNAN, ayat 3 menjabarkan bahwa bagi pekerja yang akan menggunakan cuti tahunannya, seminggu sebelumnya telah mengajukan permohonan kepada pimpinan perusahaan.
- S2: 0/4 - Pasal 12 CUTI HAID, HAMIL DAN GUGUR KANDUNG tidak membahas tentang kapan permohonan cuti tahunan harus diajukan.
- S3: 0/4 - Pasal 5 KESEMPATAN LAKTASI dan Pasal 6 JAMINAN SOSIAL PEKERJA tidak membahas tentang kapan permohonan cuti tahunan harus diajukan.

</details>

<details>
<summary>HR03 - Menurut Contoh PP Sederhana 2023, pada tanggal berapa upah bulanan dibayarkan?</summary>

**Acuan:** Tanggal 5 setiap bulan. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 4.

**Jawaban aplikasi:**

Pembayaran upah para pekerja diberikan setiap bulan tepatnya pada tanggal 5 ( lima ). [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 4; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 5; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 7.

**Alasan judge untuk jawaban:** Pembayaran upah para pekerja diberikan setiap bulan tepatnya pada tanggal 5 (lima).

**Alasan groundedness:** Informasi ini dapat ditemukan di teks S1, Pasal 4 Sistem Pengupahan dan Tunjangan Hari Raya.

**Alasan per konteks:**

- S1: 4/4 - Teks S1 secara spesifik menyebutkan tanggal pembayaran upah bulanan pada tanggal 5 (lima)
- S2: 0/4 - Teks S2 tidak membahas tentang tanggal pembayaran upah bulanan
- S3: 0/4 - Teks S3 juga tidak membahas tentang tanggal pembayaran upah bulanan

</details>

<details>
<summary>HR04 - Menurut Contoh PP Sederhana 2023, berapa hari kerja per minggu dan berapa jam kerja maksimalnya?</summary>

**Acuan:** 6 hari per minggu dan maksimal 40 jam seminggu. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 3.

**Jawaban aplikasi:**

Dengan memperhatikan ketentuan Perundang -undangan yang berlaku, hari kerja di perusahaan adalah 6 (enam) hari kerja per minggu, jam kerja maksimal 40 jam dalam seminggu. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 3; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 4; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 12.

**Alasan judge untuk jawaban:** Dengan memperhatikan ketentuan Perundang-undangan yang berlaku, hari kerja di perusahaan adalah 6 (enam) hari kerja per minggu, jam kerja maksimal 40 jam dalam seminggu.

**Alasan groundedness:** Pasal 2 HARI KERJA DAN JAM KERJA dari dokumen S1.

**Alasan per konteks:**

- S1: 4/4 - Jawaban langsung menjawab pertanyaan tentang hari kerja dan jam kerja maksimal di perusahaan.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang hari kerja dan jam kerja maksimal.
- S3: 0/4 - Tidak terkait dengan pertanyaan tentang hari kerja dan jam kerja maksimal.

</details>

<details>
<summary>HR05 - Menurut Contoh PP Sederhana 2023, berapa lama masa percobaan pekerja tetap?</summary>

**Acuan:** 3 bulan terhitung sejak mulai bekerja. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 3.

**Jawaban aplikasi:**

Calon yang telah memenuhi persyaratan yang telah di tetapkan perusahaan di terima sebagai pekerja tetap dengan masa percobaan selama 3 (tiga) bulan terhitung sejak ia mulai be kerja di perusahaan. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 3; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 12; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 6; S4: `Contoh-PP-Sederhana-2023.pdf` hlm. 13.

**Alasan judge untuk jawaban:** Informasi tentang masa percobaan pekerja tetap dapat ditemukan dalam Pasal 1 HUBUNGAN KERJA DAN MASA PERCOBAAN di S1.

**Alasan groundedness:** Pasal 1 HUBUNGAN KERJA DAN MASA PERCOBAAN di S1 menjelaskan bahwa calon yang telah memenuhi persyaratan yang telah ditetapkan perusahaan dapat diterima sebagai pekerja tetap dengan masa percobaan selama 3 (tiga) bulan terhitung sejak ia mulai bekerja di perusahaan.

**Alasan per konteks:**

- S1: 4/4 - Mengandung informasi tentang masa percobaan pekerja tetap yang sesuai dengan pertanyaan.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang masa percobaan pekerja tetap.
- S3: 1/4 - Mengandung informasi tentang tunjangan kecelakaan kerja, tetapi tidak terkait dengan pertanyaan tentang masa percobaan pekerja tetap.
- S4: 0/4 - Tidak terkait dengan pertanyaan tentang masa percobaan pekerja tetap.

</details>

<details>
<summary>HR06 - Kalau yang kontrak, apakah juga menjalani masa percobaan?</summary>

**Acuan:** Pekerja PKWT/kontrak tidak disyaratkan masa percobaan. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 3.

**Jawaban aplikasi:**

Bagi pekerja dengan sistem Perjanjian Kerj waktu Tertentu / Kontrak tidak disyaratkan adanya masa percobaan. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 3; S2: `Permenaker 28-2014.pdf` hlm. 13; S3: `Permenaker 28-2014.pdf` hlm. 12.

**Alasan judge untuk jawaban:** Pasal 5 HUBUNGAN KERJA DAN MASA PERCOBAAN menguraikan bahwa bagi pekerja dengan sistem Perjanjian Kerj waktu Tertentu / Kontrak tidak disyaratkan adanya masa percobaan.

**Alasan groundedness:** Pasal 5 HUBUNGAN KERJA DAN MASA PERCOBAAN

**Alasan per konteks:**

- S1: 4/4 - Pasal 5 menguraikan syarat-syarat penerimaan pekerja baru dan masa percobaan.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang kontrak dan masa percobaan.
- S3: 0/4 - Tidak terkait dengan pertanyaan tentang kontrak dan masa percobaan.

</details>

<details>
<summary>HR07 - Menurut Contoh PP Sederhana 2023, apa dua program BPJS yang diikuti perusahaan?</summary>

**Acuan:** BPJS Kesehatan dan BPJS Ketenagakerjaan. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 5.

**Jawaban aplikasi:**

Perusahaan mengikutsertakan karyawan dalam program Badan P enyelengara Jaminan Sosial (BPJS) berdasarkan undang -undang no 24 Tahun 2011 yang terdiri dari: BPJS Kesehatan dan BPJS Ketenagakerjaan. 2. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 5; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 11; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 13.

**Alasan judge untuk jawaban:** Perusahaan mengikutsertakan karyawan dalam program Badan P enyelengara Jaminan Sosial (BPJS) berdasarkan undang -undang no 24 Tahun 2011 yang terdiri dari: BPJS Kesehatan dan BPJS Ketenagakerjaan.

**Alasan groundedness:** Informasi tentang program BPJS yang diikuti perusahaan dapat ditemukan dalam Pasal 6 JAMINAN SOSIAL PEKERJA, yaitu BPJS Kesehatan dan BPJS Ketenagakerjaan.

**Alasan per konteks:**

- S1: 0/4 - Tidak terkait dengan pertanyaan tentang program BPJS yang diikuti perusahaan.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang program BPJS yang diikuti perusahaan.
- S3: 4/4 - Mengandung informasi tentang kriteria pengunduran diri dan mangkir, serta besaran uang pisah yang berhubungan dengan program BPJS

</details>

<details>
<summary>HR08 - Menurut Contoh PP Sederhana 2023, apa fasilitas bagi karyawan wanita yang memberikan ASI eksklusif?</summary>

**Acuan:** Kesempatan dan fasilitas laktasi. Sumber: `Contoh-PP-Sederhana-2023.pdf`, halaman PDF 5.

**Jawaban aplikasi:**

Prestasi kerja Pasal 5 KESEMPATAN LAKTASI Perusahaan memberikan kesempatan dan fasilitas Laktasi kepada karyawan wanita yang sedang dalam masa pemberian ASI Eksklusif agar dapat memberikan kepada anaknya secara baik dan layak. [S1]

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 5; S2: `Contoh-PP-Sederhana-2023.pdf` hlm. 7; S3: `Contoh-PP-Sederhana-2023.pdf` hlm. 12; S4: `Contoh-PP-Sederhana-2023.pdf` hlm. 8.

**Alasan judge untuk jawaban:** Fasilitas bagi karyawan wanita yang memberikan ASI eksklusif adalah KESEMPATAN LAKTASI, yaitu perusahaan memberikan kesempatan dan fasilitas laktasi kepada karyawan wanita yang sedang dalam masa pemberian ASI Eksklusif agar dapat memberikan kepada anaknya secara baik dan layak.

**Alasan groundedness:** Pasal 5 KESEMPATAN LAKTASI Perusahaan memberikan kesempatan dan fasilitas Laktasi kepada karyawan wanita yang sedang dalam masa pemberian ASI Eksklusif agar dapat memberikan kepada anaknya secara baik dan layak.

**Alasan per konteks:**

- S1: 4/4 - Jawaban langsung dan tepat mengenai fasilitas bagi karyawan wanita yang memberikan ASI eksklusif.
- S2: 0/4 - Tidak terkait dengan pertanyaan tentang fasilitas bagi karyawan wanita yang memberikan ASI eksklusif.
- S3: 0/4 - Tidak terkait dengan pertanyaan tentang fasilitas bagi karyawan wanita yang memberikan ASI eksklusif.
- S4: 0/4 - Tidak terkait dengan pertanyaan tentang fasilitas bagi karyawan wanita yang memberikan ASI eksklusif.

</details>

<details>
<summary>HR09 - Berapa nomor WhatsApp pribadi kepala HR PT XYZ?</summary>

**Acuan:** Informasi tidak tersedia; nyatakan tidak tahu.

**Jawaban aplikasi:**

Saya tidak menemukan informasi yang cukup dalam dokumen aktif agent ini.

**Konteks yang diberikan:** S1: `Contoh-PP-Sederhana-2023.pdf` hlm. 1.

**Alasan judge untuk jawaban:** Informasi tidak tersedia dalam dokumen yang diberikan

**Alasan groundedness:** Dokumen hanya berisi peraturan perusahaan tanpa informasi kontak

**Alasan per konteks:**

- S1: 0/4 - Tidak ada informasi tentang nomor WhatsApp kepala HR

</details>

<details>
<summary>HR10 - Menurut uu40-2007.pdf, apa saja organ Perseroan?</summary>

**Acuan:** UU40 tidak diunggah ke HR; tidak boleh memakai knowledge Legal.

**Jawaban aplikasi:**

Saya tidak menemukan informasi yang cukup dalam dokumen aktif agent ini.

**Konteks yang diberikan:** S1: `Permenaker 28-2014.pdf` hlm. 2.

**Alasan judge untuk jawaban:** Tidak ada informasi yang cukup dalam dokumen aktif agent ini.

**Alasan groundedness:** Jawaban tidak didukung oleh konteks karena tidak ada klaim faktual tentang organ Perseroan.

**Alasan per konteks:**

- S1: 4/4 - Konteks relevan dengan pertanyaan tentang organ Perseroan.

</details>

<details>
<summary>LG01 - Menurut uu40-2007.pdf, apa saja organ Perseroan?</summary>

**Acuan:** RUPS, Direksi, dan Dewan Komisaris. Sumber: `uu40-2007.pdf`, halaman PDF 2.

**Jawaban aplikasi:**

1. Organ Perseroan adalah Rapat Umum Pemegang Saham, Direksi, dan Dewan Komisaris. 3. [S1]
2. Direksi adalah Organ Perseroan yang berwenang dan bertanggung jawab penuh atas pengurusan Perseroan untuk kepentingan Perseroan, sesuai dengan maksud dan tujuan Perseroan serta mewakili Perseroan, baik di dalam maupun di luar pengadilan sesuai dengan ketentuan anggaran dasar. 6. [S1]
3. Dewan Komisaris adalah Organ Perseroan yang bertugas melakukan pengawasan secara umum dan/atau khusu s sesuai dengan anggaran dasar serta memberi nasihat kepada Direksi. 7. [S3]

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 2; S2: `uu40-2007.pdf` hlm. 119; S3: `uu40-2007.pdf` hlm. 3; S4: `uu40-2007.pdf` hlm. 83.

**Alasan judge untuk jawaban:** Jawaban yang diberikan mencakup tiga jenis organ Perseroan, yaitu Rapat Umum Pemegang Saham, Direksi, dan Dewan Komisaris.

**Alasan groundedness:** Klaim faktual jawaban didukung oleh konteks S1 dan S3.

**Alasan per konteks:**

- S1: 4/4 - Konteks S1 merupakan sumber hukum yang relevan dengan pertanyaan tentang organ Perseroan.
- S2: 0/4 - Konteks S2 tidak terkait dengan pertanyaan tentang organ Perseroan.
- S3: 4/4 - Konteks S3 juga merupakan sumber hukum yang relevan dengan pertanyaan tentang organ Perseroan.
- S4: 1/4 - Konteks S4 merupakan penjelasan umum tentang Undang-Undang Nomor 40 Tahun 2007, tetapi tidak secara langsung terkait dengan pertanyaan tentang organ Perseroan.

</details>

<details>
<summary>LG02 - Menurut uu40-2007.pdf, apa kewenangan dan tanggung jawab Direksi?</summary>

**Acuan:** Mengurus Perseroan untuk kepentingan dan tujuan Perseroan serta mewakilinya di dalam/luar pengadilan sesuai anggaran dasar. Sumber: `uu40-2007.pdf`, halaman PDF 2.

**Jawaban aplikasi:**

(1) Direksi bertanggung jawab atas pengurusan Perseroan sebagaimana dimaksud dalam Pasal 92 ayat (1). (2) Pengurusan sebagaimana dimaksud pada ayat (1), wajib dilaksanakan setiap anggota Direksi dengan itikad baik dan penuh tanggung jawab. (3) Setiap anggota Direksi bertanggung jawab penuh secara pribadi atas kerugian Perseroan apabila yang bersangkutan bersalah atau lalai menjalankan tugasnya sesuai dengan ketentuan sebagaimana dimaksud pada ayat (2). (4) Dalam hal Direksi terdiri atas 2 (dua) anggota Direksi atau lebih, tanggung jawab sebagaimana dimaksud pada ayat (3) berlaku secara tanggung renteng bagi setiap anggota Direksi. (5) Anggota Direksi tidak dapat dipertanggungjawabkan atas kerugian sebagaimana dimaksud pada ayat (3) apabila dapat membuktikan: a. kerugian tersebut bukan karena kesalahan atau kelalaiannya; b. telah melakukan pengurusan dengan itikad baik dan kehati-hatian untuk kepentingan dan sesuai dengan maksud dan tujuan Perseroan; c. tidak mempunyai benturan kepentingan baik langsung maupun tidak langsung atas tindakan pengurusan yang mengakibatkan kerugian; dan d. telah mengambil tindakan untuk mencegah timbul atau berlanjutnya kerugian tersebut. (6) Atas . . . [S1]

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 50; S2: `uu40-2007.pdf` hlm. 73; S3: `uu40-2007.pdf` hlm. 54; S4: `uu40-2007.pdf` hlm. 91.

**Alasan judge untuk jawaban:** Jawaban menjelaskan secara rinci tentang kewenangan dan tanggung jawab Direksi berdasarkan Pasal 97 uu40-2007.pdf.

**Alasan groundedness:** Jawaban didukung oleh konteks S1 yang secara langsung menjawab pertanyaan.

**Alasan per konteks:**

- S1: 4/4 - Konteks S1 secara langsung menjawab pertanyaan tentang kewenangan dan tanggung jawab Direksi.
- S2: 0/4 - Konteks S2 tidak terkait dengan pertanyaan tentang kewenangan dan tanggung jawab Direksi.
- S3: 1/4 - Konteks S3 memiliki topik yang sama, tetapi tidak secara langsung menjawab pertanyaan tentang kewenangan dan tanggung jawab Direksi.
- S4: 0/4 - Konteks S4 tidak terkait dengan pertanyaan tentang kewenangan dan tanggung jawab Direksi.

</details>

<details>
<summary>LG03 - Menurut uu40-2007.pdf, apa tugas Dewan Komisaris?</summary>

**Acuan:** Melakukan pengawasan umum/khusus sesuai anggaran dasar dan memberi nasihat kepada Direksi. Sumber: `uu40-2007.pdf`, halaman PDF 3.

**Jawaban aplikasi:**

Undang-Undang ini juga memperjelas dan mempertegas tugas dan tanggung jawab Direksi dan Dewan Komisaris. [S1]

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 84; S2: `uu40-2007.pdf` hlm. 62; S3: `uu40-2007.pdf` hlm. 59; S4: `uu40-2007.pdf` hlm. 127.

**Alasan judge untuk jawaban:** Dewan Komisaris bertanggung jawab atas pengawasan Perseroan sebagaimana dimaksud dalam Pasal 108 ayat (1) dan setiap anggota Dewan Komisaris wajib dengan itikad baik, kehati-hatian, dan bertanggung jawab dalam menjalankan tugas pengawasan dan pemberian nasihat kepada Direksi sebagaimana dimaksud dalam Pasal 108 ayat (1) untuk kepentingan Perseroan dan sesuai dengan maksud dan tujuan Perseroan.

**Alasan groundedness:** Pasal 114 Undang-Undang No. 40 Tahun 2007 tentang Perseroan Terbatas.

**Alasan per konteks:**

- S1: 0/4 - Tidak terkait dengan tugas Dewan Komisaris
- S2: 4/4 - Mengatur tentang tugas dan tanggung jawab Dewan Komisaris
- S3: 4/4 - Mengatur tentang tugas dan tanggung jawab Dewan Komisaris
- S4: 0/4 - Tidak terkait dengan tugas Dewan Komisaris

</details>

<details>
<summary>LG04 - Menurut uu40-2007.pdf, apa yang dimaksud dengan RUPS?</summary>

**Acuan:** Organ Perseroan berwenang atas hal yang tidak diberikan kepada Direksi atau Dewan Komisaris dalam batas UU/anggaran dasar. Sumber: `uu40-2007.pdf`, halaman PDF 2.

**Jawaban aplikasi:**

1. - 31 - Pasal 78 Ayat (1) Yang dimaksud dengan “RUPS lainnya” dalam praktik sering dikenal sebagai RUPS luar biasa. [S2]
2. Ayat (4) Yang dimaksud dengan “ketentuan sebagaimana dimaksud pada ayat (3)” adalah RUPS harus diadakan di wilayah negara Republik Indonesia. [S3]

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 106; S2: `uu40-2007.pdf` hlm. 113; S3: `uu40-2007.pdf` hlm. 112; S4: `uu40-2007.pdf` hlm. 118.

**Alasan judge untuk jawaban:** RUPS luar biasa menurut uu40-2007.pdf adalah RUPS yang diadakan di wilayah negara Republik Indonesia, seperti yang disebutkan dalam Pasal 78 Ayat (4).

**Alasan groundedness:** Jawaban didukung oleh teks uu40-2007.pdf

**Alasan per konteks:**

- S1: 0/4 - Tidak terkait dengan RUPS
- S2: 4/4 - Jawaban langsung dari uu40-2007.pdf
- S3: 4/4 - Jawaban langsung dari uu40-2007.pdf
- S4: 0/4 - Tidak terkait dengan RUPS

</details>

<details>
<summary>LG05 - Menurut uu40-2007.pdf, apa yang dimaksud dengan Tanggung Jawab Sosial dan Lingkungan?</summary>

**Acuan:** Komitmen Perseroan berperan serta dalam pembangunan ekonomi berkelanjutan untuk meningkatkan kualitas kehidupan dan lingkungan, bagi Perseroan, komunitas setempat, dan masyarakat. Sumber: `uu40-2007.pdf`, halaman PDF 2.

**Jawaban aplikasi:**

1. Dalam Undang-Undang ini diatur mengenai Tanggung Jawab Sosial dan Lingkungan yang bertujuan mewujudkan pembangunan ekonomi berkelanjutan guna meningkatkan kualitas kehidupan dan lingkungan yang bermanfaat bagi Perseroan itu sendiri, komunitas setempat, dan masyarakat pada umumnya. [S2]
2. Ketentuan ini dimaksudkan untuk mendukung terjalinnya hubungan Perseroan yang serasi, seimbang, dan sesuai dengan lingkungan, nilai, norma, dan budaya masyarakat setempat, maka ditentukan bahwa Perseroan yang kegiatan usahanya di bidang dan/atau berkaitan dengan sumber daya alam wajib melaksanakan Tanggung Jawab Sosial dan Lingkungan. [S2]
3. Untuk melaksanakan kewajiban Perseroan tersebut, kegiatan Tanggung Jawab Sosial dan Lingkungan harus dianggarkan dan diperhitungkan sebagai biaya Perseroan yang dilaksanakan dengan memperhatikan kepatutan dan kewajaran. [S2]
4. Dalam hal Perseroan tidak melaksanakan Tanggung Jawab Sosial dan Lingkungan maka Perseroan yang bersangkutan dikenai sanksi sesuai dengan ketentuan peraturan perundang-undangan. [S2]

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 37; S2: `uu40-2007.pdf` hlm. 85; S3: `uu40-2007.pdf` hlm. 111.

**Alasan judge untuk jawaban:** Jawaban relevan dengan pertanyaan dan mendukung klaim faktual dari konteks S2.

**Alasan groundedness:** Klaim faktual jawaban didukung oleh teks konteks S2.

**Alasan per konteks:**

- S1: 0/4 - Tidak ada definisi Tanggung Jawab Sosial dan Lingkungan di bab ini
- S2: 4/4 - Bab ini menjelaskan definisi dan tujuan Tanggung Jawab Sosial dan Lingkungan secara lengkap
- S3: 0/4 - Tidak ada definisi Tanggung Jawab Sosial dan Lingkungan di bab ini

</details>

<details>
<summary>LG06 - Menurut Permenaker 28-2014.pdf, apa definisi PKWT?</summary>

**Acuan:** Perjanjian kerja pekerja dengan pengusaha untuk hubungan kerja waktu tertentu atau pekerjaan tertentu. Sumber: `Permenaker 28-2014.pdf`, halaman PDF 3.

**Jawaban aplikasi:**

Saya tidak menemukan informasi yang cukup dalam dokumen aktif agent ini.

**Konteks yang diberikan:** S1: `Permenaker 28-2014.pdf` hlm. 25; S2: `Permenaker 28-2014.pdf` hlm. 8; S3: `Permenaker 28-2014.pdf` hlm. 9; S4: `Permenaker 28-2014.pdf` hlm. 15.

**Alasan judge untuk jawaban:** Jawaban tidak relevan dengan pertanyaan karena tidak menjelaskan definisi PKWT secara spesifik.

**Alasan groundedness:** Konteks-konteks yang disediakan tidak memberikan informasi yang cukup untuk menjelaskan definisi PKWT.

**Alasan per konteks:**

- S1: 0/4 - Tidak ada definisi PKWT di konteks S1
- S2: 3/4 - Konteks S2 menjelaskan tentang perundingan PKB, tetapi tidak secara spesifik menguraikan definisi PKWT
- S3: 0/4 - Tidak ada definisi PKWT di konteks S3
- S4: 0/4 - Tidak ada definisi PKWT di konteks S4

</details>

<details>
<summary>LG07 - Menurut Permenaker 28-2014.pdf Pasal 29, berapa lama masa berlaku PKB?</summary>

**Acuan:** Paling lama 2 tahun sejak ditandatangani atau diatur lain dalam PKB. Sumber: `Permenaker 28-2014.pdf`, halaman PDF 13.

**Jawaban aplikasi:**

(1) Masa berlaku PKB paling lama 2 (dua) tahun terhitung sejak ditandatangani atau diatur lain dalam PKB. [S1]

**Konteks yang diberikan:** S1: `Permenaker 28-2014.pdf` hlm. 13; S2: `Permenaker 28-2014.pdf` hlm. 9; S3: `Permenaker 28-2014.pdf` hlm. 8; S4: `Permenaker 28-2014.pdf` hlm. 7.

**Alasan judge untuk jawaban:** Pasal 29 secara spesifik membahas masa berlaku PKB dan menyebutkan bahwa paling lama 2 tahun terhitung sejak ditandatangani atau diatur lain dalam PKB.

**Alasan groundedness:** Informasi ini didapatkan dari Pasal 29 Permenaker 28-2014.pdf.

**Alasan per konteks:**

- S1: 4/4 - Pasal 29 secara spesifik membahas masa berlaku PKB.
- S2: 0/4 - Tidak ada informasi tentang masa berlaku PKB di dalam konteks ini.
- S3: 0/4 - Konteks ini membahas pembaharuan PP, bukan masa berlaku PKB.
- S4: 0/4 - Konteks ini membahas proses pengesahan PP, bukan masa berlaku PKB.

</details>

<details>
<summary>LG08 - Menurut Permenaker 28-2014.pdf, apa saja yang sekurang-kurangnya dimuat dalam Peraturan Perusahaan?</summary>

**Acuan:** Hak/kewajiban pengusaha dan pekerja, syarat kerja, tata tertib, jangka waktu PP, serta pengaturan lebih lanjut perundang-undangan. Sumber: `Permenaker 28-2014.pdf`, halaman PDF 4.

**Jawaban aplikasi:**

(2) PP sekurang-kurangnya memuat: a. hak dan kewajiban pengusaha; b. hak dan kewajiban pekerja/buruh; c. syarat kerja; d. tata tertib perusahaan; e. jangka waktu berlakunya PP; dan f. hal-hal yang merupakan pengaturan lebih lanjut dari peraturan perundang-undangan. [S2]

**Konteks yang diberikan:** S1: `Permenaker 28-2014.pdf` hlm. 3; S2: `Permenaker 28-2014.pdf` hlm. 4; S3: `Permenaker 28-2014.pdf` hlm. 2; S4: `Permenaker 28-2014.pdf` hlm. 15.

**Alasan judge untuk jawaban:** Daftar isi yang dimuat dalam Peraturan Perusahaan menurut Permenaker 28-2014.pdf adalah hak dan kewajiban pengusaha, hak dan kewajiban pekerja/buruh, syarat kerja, tata tertib perusahaan, jangka waktu berlakunya PP, dan hal-hal yang merupakan pengaturan lebih lanjut dari peraturan perundang-undangan.

**Alasan groundedness:** Dapat dilihat pada Pasal 2 ayat (2) Permenaker 28-2014.pdf

**Alasan per konteks:**

- S1: 0/4 - Tidak terkait dengan pertanyaan tentang isi Peraturan Perusahaan
- S2: 4/4 - Mengandung daftar isi yang dimuat dalam Peraturan Perusahaan
- S3: 0/4 - Tidak terkait dengan pertanyaan tentang isi Peraturan Perusahaan
- S4: 0/4 - Tidak terkait dengan pertanyaan tentang isi Peraturan Perusahaan

</details>

<details>
<summary>LG09 - Berapa nomor WhatsApp pribadi direktur PT XYZ?</summary>

**Acuan:** Informasi tidak tersedia; nyatakan tidak tahu.

**Jawaban aplikasi:**

Saya tidak menemukan informasi yang cukup dalam dokumen aktif agent ini.

**Konteks yang diberikan:** tidak ada.

**Alasan judge untuk jawaban:** Jawaban tidak menjawab pertanyaan secara langsung.

**Alasan groundedness:** Tidak ada informasi yang cukup untuk menjawab pertanyaan.

</details>

<details>
<summary>LG10 - Menurut Contoh-PP-Sederhana-2023.pdf, berapa hari cuti tahunan karyawan?</summary>

**Acuan:** Template HR tidak diunggah ke Legal; tidak boleh memakai knowledge HR.

**Jawaban aplikasi:**

Saya tidak menemukan informasi yang cukup dalam dokumen aktif agent ini.

**Konteks yang diberikan:** S1: `uu40-2007.pdf` hlm. 32; S2: `uu40-2007.pdf` hlm. 106; S3: `uu40-2007.pdf` hlm. 107; S4: `Permenaker 28-2014.pdf` hlm. 7.

**Alasan judge untuk jawaban:** Informasi tidak tersedia dalam dokumen yang diberikan.

**Alasan groundedness:** Dokumen hanya berisi tentang peraturan dan prosedur, tidak ada informasi tentang kebijakan atau aturan terkait cuti tahunan karyawan.

**Alasan per konteks:**

- S1: 0/4 - Tidak ada informasi tentang cuti tahunan karyawan
- S2: 0/4 - Tidak ada informasi tentang cuti tahunan karyawan
- S3: 0/4 - Tidak ada informasi tentang cuti tahunan karyawan
- S4: 0/4 - Tidak ada informasi tentang cuti tahunan karyawan

</details>

## Batas evaluasi dan tindak lanjut

Satu judge 8B, satu dataset kecil dan satu jawaban per kasus belum mengukur akurasi umum. Judge belum dikalibrasi dengan penilaian manusia dan bisa tidak konsisten. Groundedness tinggi pada mode literal juga tidak membuktikan pilihan bukti sudah lengkap, sumber benar, atau regulasi masih berlaku.

Pertanyaan disusun setelah aplikasi dikembangkan, sehingga ini evaluasi pada corpus yang dikenal, bukan held-out benchmark. Scan OCR, DOCX, penghapusan saat chat, kegagalan provider, dan restart tidak diukur oleh skor triad ini. Perilaku tersebut perlu diperiksa terpisah.

Prioritas berikutnya adalah meninjau candidate retrieval/reranker pada LG06 dan sumber pengalih, menambah pertanyaan serta dokumen baru, lalu membandingkan penilaian Llama dengan penilai manusia. Threshold atau prompt tidak diubah untuk menaikkan skor pada 20 soal ini.

## Pemeriksaan perilaku di luar skor triad

| Kebutuhan | Cara memeriksa |
|---|---|
| CRUD agent | Buat/edit dua agent; hapus target bernama jelas; agent lain tetap ada |
| PDF/DOCX dan banyak file | Upload ke agent tujuan, lihat status, tinjau halaman/bagian |
| File rusak/kosong/scan | Periksa error yang jelas atau OCR, tanpa dokumen gagal dipakai untuk chat |
| Koreksi ekstraksi | Rapikan/urungkan draft; simpan menaikkan revisi dan reindex |
| Hapus dokumen | Ulangi pertanyaan tentang fakta unik dokumen terhapus; arsip chat lama bukan sumber baru |
| Multi-turn dan history | Tanya lanjutan, buat sesi baru, lalu buka sesi sebelumnya |
| Provider dan error | Ganti konfigurasi per agent; periksa pesan timeout/key/limit dan pemberitahuan fallback |
| Persistence/setup | Restart tanpa pekerjaan aktif, periksa data; ikuti README pada instalasi baru |

Unit test ingestion dan adapter provider tetap berada di `tests/` sesuai brief. Tes provider memakai mock; bukan bukti semua endpoint cloud sedang berhasil. OpenAI belum diverifikasi live, dan diagnostik OpenRouter terakhir mengalami timeout. Skrip `smoke_*` serta laporan corpus lama tidak diperlukan untuk mengulang evaluasi ini dan dipindahkan dari paket submission ke arsip lokal.
