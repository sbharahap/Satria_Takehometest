# Perbandingan model lokal untuk document RAG

Pengujian membandingkan instalasi `qwen3:8b` dan `llama3.1:8b` melalui adapter LiteLLM/Ollama aplikasi ini. Ini evaluasi workload PT Vale dalam Bahasa Indonesia, bukan peringkat kemampuan umum model. Pengujian selesai pada 22 Oktober 2026; konfigurasi default aplikasi tidak diubah. Aplikasi yang diuji berasal dari commit `db57a7a`. Ringkasan tanpa respons mentah tersedia di [`eval/local_model_results.json`](../../eval/local_model_results.json).

## Protokol

- Empat PDF publik PT Vale: People Policy, Diversity and Inclusion, Anti-Corruption, dan Human Rights. Gunakan `scripts.prepare_demo`; pertanyaan/sumber/halaman yang diharapkan ada pada `eval/local_model_cases.json`. Nomor halaman merujuk PDF Indonesia hasil persiapan demo, bukan nomor tercetak atau PDF bilingual asli.
- Sepuluh pertanyaan tetap per agent (HR/Legal), ditambah enam skenario HR untuk DOCX, rujukan nomor poin, sumber dihapus, revisi diubah, dan jawaban dari revisi terbaru. DOCX merupakan konversi fixture di memori dari cuplikan People Policy publik, bukan DOCX asli perusahaan. Koreksi revisi hanya whitespace, tanpa mengubah fakta kebijakan.
- Dua pengulangan per model, urutan Qwen→Llama lalu Llama→Qwen. Model berjalan secara serial pada server dan komputer yang sama. Setiap blok menggunakan database/indeks/percakapan tersendiri di `data/model-comparison/`; data aplikasi utama tidak diubah.
- Temperature 0, JSON mode, context 8192 token, output maksimal700 token (rewrite 300). Qwen3 memakai reasoning none sesuai adapter aplikasi. Embedding, chunking, threshold 0,83, reranker, prompt, dan validator sitasi sama. Warm-up dicatat terpisah dan tidak dimasukkan ke latensi kasus.
- Verifier semantik dinonaktifkan pada perbandingan generator agar Qwen tidak menjadi penilai Llama atau sebaliknya. Enam klaim tetap (tiga benar/tiga salah) kemudian diuji secara terpisah pada verifier masing-masing model, dengan pemeriksaan kutipan literal yang sama.
- Tidak ada panggilan cloud, download model chat, perubahan `.env`, atau retry untuk memperbaiki jawaban selama pengukuran.

## Metrik dan batas interpretasi

Rubrik otomatis memeriksa keputusan unknown yang tepat, sumber/halaman yang diharapkan, serta pola kata kunci dan larangan yang eksplisit. Ini bukan metrik entailment otomatis atau penilaian manusia independen. Jawaban benar dengan sinonim yang tidak tercakup rubrik dapat gagal; jawaban dengan kata kunci yang tepat masih dapat memiliki kekurangan. Pemeriksaan kualitatif contoh hasil oleh coding assistant dilaporkan terpisah; ini bukan penilaian manusia independen.

Retrieval gold hit berarti potongan dari file/halaman rujukan tersedia di top 4 setelah threshold/reranking. Fingerprint konteks dibandingkan untuk kasus statis agar perbedaan retrieval tidak keliru disebut perbedaan model. Pada pertanyaan lanjutan, query dan konteks dapat berbeda karena rewriter masing-masing model merupakan bagian alur yang diuji.

Beberapa penolakan unknown dihasilkan langsung oleh guard aplikasi saat sumber tidak tersedia atau revisi berubah; itu tidak membuktikan kemampuan abstention model. Jumlah kasus tanpa panggilan model dipisahkan. Median/maksimum latensi dilaporkan hanya untuk kasus yang memanggil provider, termasuk retrieval dan rewriting jika ada.

Kegagalan verifier pada klaim salah adalah false positive; penolakan klaim benar adalah false negative. Sampel enam klaim per model kecil dan tidak membuktikan verifier bebas halusinasi.

## Menjalankan ulang

Setelah setup dan persiapan dokumen/model:

```bash
uv run python -m scripts.compare_local_models --repeats 2 --answer-mode abstractive
```

Laporan lengkap, respons mentah, konteks, digest model, hash dokumen, dan latensi tersimpan di direktori run yang dicetak, di bawah `data/model-comparison/` yang di-ignore. Jangan memasukkan key atau data pengguna ke laporan publik.

## Hasil

Kedua instalasi menggunakan GGUF Q4_K_M: Qwen3 8.2B dan Llama3.1 8.0B, macOS 27 arm64. Digest model serta SHA dokumen disimpan pada ringkasan JSON. Ini membandingkan konfigurasi yang benar-benar dipakai aplikasi, bukan seluruh mode atau potensi model setelah optimasi prompt terpisah.

| Metrik, dua pengulangan | Qwen3:8b | Llama3.1:8b |
|---|---:|---:|
| Lolos rubrik keseluruhan | 48/52 | 48/52 |
| Lolos rubrik pada pertanyaan yang dapat dijawab | 32/36 | 34/36 |
| Unknown tepat pada kasus yang memanggil model | 4/4 | 2/4 |
| Unknown melalui guard aplikasi, tanpa model | 12/12 | 12/12 |
| Gold file/halaman tersedia di top 4, kasus statis positif | 28/28 | 28/28 |
| Respons JSON berbentuk objek | 42/42 | 42/42 |
| Error provider | 0 | 0 |
| Median latensi kasus dengan panggilan model, tanpa verifier | 3.99 detik | 3.74 detik |
| Maksimum latensi kasus dengan panggilan model, tanpa verifier | 10.49 detik | 13.53 detik |
| Keputusan verifier benar pada klaim tetap | 8/12 | 8/12 |
| Verifier menerima klaim salah (lebih rendah lebih baik) | 2/6 | 4/6 |
| Verifier menolak klaim benar (lebih rendah lebih baik) | 2/6 | 0/6 |

Total 104 eksekusi pertanyaan dan 24 pemeriksaan klaim. Ada 26 skenario unik dan 6 klaim unik per model; dua pengulangan tidak berarti jumlah pertanyaan independen menjadi dua kali lebih banyak. Seluruh jawaban teks identik antar-pengulangan untuk masing-masing model. Semua 20 konteks kasus statis memiliki fingerprint identik pada kedua model dan kedua putaran. Pada empat panggilan unknown per model hanya ada dua pertanyaan unik; angka itu terlalu kecil untuk kesimpulan abstention umum.

## Kegagalan dan pemeriksaan kualitatif

- **Kedua model, hr03:** angka peninjauan tiga tahun benar, tetapi sitasi menunjuk Diversity & Inclusion ketika pertanyaan meminta People Policy. Potongan People Policy yang benar tersedia dalam konteks. Ini kesalahan pemilihan sumber dan tidak dapat ditemukan hanya dengan memeriksa apakah ID sitasi valid.
- **Qwen, legal04:** jawaban memiliki sumber untuk aturan hadiah, tetapi tidak menyebut hukum yang berlaku secara eksplisit. Rubrik kelengkapan menganggap gagal. Ini bukan bukti bahwa seluruh jawaban tersebut merupakan halusinasi.
- **Llama, legal08:** saat batas nominal hadiah dalam rupiah tidak tersedia, model menjawab aturan larangan hadiah tunai. Model tidak mengarang angka, tetapi gagal mengakui bahwa informasi yang diminta tidak ada.
- **Llama, legal04:** lolos rubrik otomatis, tetapi menambahkan larangan hadiah tunai dan hanya mengutip halaman2. Larangan tunai ada pada halaman5 yang tersedia sebagai sumber lain, tetapi tidak disertakan dalam sitasinya. Ini contoh celah coverage sitasi yang tidak tertangkap oleh keyword dan keberadaan satu halaman rujukan.
- **Llama, dynamic02:** jawaban poin kedua juga mengulang bagian poin pertama. Rubrik kata kunci tetap meloloskan jawaban; ketepatan fokus multi-turn lebih lemah dari yang tergambar oleh skor agregat.
- **Verifier Qwen:** menolak klaim benar tentang hadiah bersyarat dan menerima klaim salah tentang larangan seluruh hadiah tanpa pengecualian, pada kedua putaran.
- **Verifier Llama:** menerima klaim salah tentang siklus bulanan dan larangan seluruh hadiah tanpa pengecualian, pada kedua putaran. Semua klaim benar diterima.

Enam input verifier: siklus tahunan (benar), siklus bulanan (salah), cuti 60 hari (salah), hadiah dengan syarat aturan/prosedur/hukum (benar), larangan semua hadiah tanpa pengecualian (salah), dan persetujuan Legal/Compliance sebelum perjalanan pejabat (benar). Kasus ini menguji verdict model beserta validasi kutipan literal aplikasi. Kutipan yang benar-benar ada pada sumber tetap tidak menjamin klaim didukung.

## Keputusan dan keterbatasan

Tidak ada pemenang mutlak dari rubrik keseluruhan. Qwen dipertahankan sebagai default sementara karena keputusan unknown dan jumlah false-positive verifier lebih konservatif pada sampel ini. Llama tetap pilihan model yang dapat dipilih pengguna dan memiliki median latensi sedikit lebih rendah. Hasil ini tidak membuktikan Qwen lebih unggul secara umum, ataupun bahwa verifier salah satu model sudah cukup andal untuk menjamin kebenaran.

Prioritas perbaikan yang terungkap adalah ketepatan dokumen yang dimaksud pertanyaan, coverage bukti untuk seluruh klaim, dan evaluasi kontradiksi verifier. Mengganti model saja tidak memperbaiki seluruh masalah tersebut. Perbaikan aplikasi berdasarkan hasil ini belum dilakukan pada milestone evaluasi ini.

Prompt aplikasi telah dipakai selama development dengan Qwen; tidak dilakukan tuning khusus Llama. Temperature 0, satu mesin, empat kebijakan, sedikit pertanyaan unknown yang benar-benar mencapai model, dan pengulangan identik membatasi generalisasi. Tidak ada uji model lain, OCR, beban banyak pengguna, kegagalan server, atau dokumen reviewer yang belum diketahui. Latensi produksi akan bertambah ketika verifier aktif.

Satu run eksplorasi parsial dibatalkan untuk memperbaiki rubrik yang keliru mengenali negasi pada hadiah; run tersebut tidak masuk angka di atas. Dataset/rubrik final dipakai sama untuk kedua model. Tidak ada penggantian prompt, threshold, atau model default selama pengukuran final.


Baseline ini mengevaluasi mode abstractive sebelum mitigasi extractive. Memilih flag abstractive pada kode yang sudah diperbaiki tidak mengembalikan seluruh baseline, karena pembatasan retrieval juga berubah; gunakan commit baseline dan dataset yang tercatat untuk reproduksi perilaku lama. Hasil baseline tidak ditimpa oleh evaluasi berikutnya.
