## 🚀 Cara Menggunakan

1. Install dependencies:
   `pip install -r requirements.txt`

2. Pastikan folder dataset ada dengan struktur yang benar

3. Jalankan program:
   `python main.py`

4. Pilih menu:

- [1] Load & Index Dataset - Jalankan ini dulu untuk memproses semua dokumen
- [2] Search Query - Cari dokumen dengan query Anda
- [3] Exit - Keluar program

## ✨ Fitur Utama

✅ Text Preprocessing:

- Case folding
- Tokenization
- Stopword removal (Indonesian)
- Special character removal

✅ Document Representation:

- Bag of Words menggunakan CountVectorizer
- 5000 features maksimal

✅ Indexing:

- Whoosh index untuk pencarian cepat
- Stemming analyzer

✅ Search & Ranking:

- Whoosh untuk initial retrieval
- Cosine similarity untuk ranking
- Menampilkan top 5 dokumen
