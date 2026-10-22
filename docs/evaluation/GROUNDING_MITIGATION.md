# Mitigasi provenance setelah perbandingan model

Baseline dan kegagalannya tetap ada di [LOCAL_MODEL_COMPARISON.md](LOCAL_MODEL_COMPARISON.md). Ini evaluasi regresi pada pertanyaan yang sudah diketahui selama development, bukan pengukuran akurasi pada data baru.

## Perubahan

- Default `ANSWER_MODE=extractive`: model memilih maksimal lima `evidence_ids`; kode mengabaikan teks jawaban bebas dan merender unit sumber terpilih dengan sitasi. Whitespace dinormalisasi. ID asing/duplikat ditolak.
- Nama file atau alias eksplisit membatasi dokumen sebelum pencarian; filter agent, ready, dan revisi diperiksa kembali sesudah pencarian. Alias tidak memahami semua sinonim judul dan dapat ambigu.
- Permintaan nominal/durasi/telepon memerlukan unit bukti dengan jenis nilai yang diminta. Pola ini konservatif dan dapat menolak jawaban valid yang menyatakan nilai secara tidak langsung. Ini bukan pemahaman semantik umum.
- Follow-up nomor poin mencari item dalam daftar jawaban yang ditampilkan, kemudian menemukan ulang teks literal dalam hasil retrieval aktif dengan identitas dokumen/page/revisi yang cocok. Riwayat tidak menjadi bukti baru. Item hilang/di luar daftar ditolak.
- Mode opsional `abstractive` mempertahankan parafrase dan verifier LLM lama. False positive/negative verifier baseline tidak diperbaiki dengan mengganti label; default extractive memakai pemeriksaan provenance deterministik.

## Status dan batas interpretasi

111 tes otomatis (mock/provider dan Streamlit AppTest) lulus pada versi final. Suite tidak memanggil model live. Pemeriksaan terarah dataset baru, OCR, browser, dan Docker dijelaskan pada [INTEGRATION_CHECKS.md](INTEGRATION_CHECKS.md). Ini belum mengukur akurasi umum. Konteks hit dilengkapi dengan halaman/bagian yang sama bila muat, dan batas lima ID mengakomodasi jawaban beberapa bagian. Masalah kualitas tidak diklaim tuntas untuk semua dokumen/pertanyaan.

## Batas interpretasi

Kutipan yang berasal dari sumber tidak menjamin relevansi, kelengkapan, atau kebenaran. Source chunk dapat terpotong; sebuah kutipan dapat kehilangan syarat di paragraf lain. Dokumen dan koreksi pengguna bisa keliru, termasuk hasil OCR. Model masih dapat memilih bukti yang salah. Tidak ada klaim nol halusinasi atau model paling robust secara umum. Periksa file, halaman, dan konteks sumber saat demo/review.
