import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer

class IRSystem:
    def __init__(self, dataset_path="dataset", index_dir="indexdir"):
        self.dataset_path = dataset_path
        self.index_dir = index_dir
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None

        # Stopwords bahasa Indonesia
        self.stopwords = set([
            'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 
            'dia', 'dua', 'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali',
            'dan', 'di', 'dari', 'ini', 'itu', 'dengan', 'adalah', 'ada', 'atau',
            'se', 'ter', 'dapat', 'akan', 'oleh', 'tersebut', 'telah', 'dalam',
            'tidak', 'karena', 'telah', 'bahwa', 'sebagai', 'hal', 'ketika',
            'sudah', 'saya', 'bisa', 'mereka', 'kami', 'kita', 'anda', 'belum',
            'saat', 'harus', 'saja', 'masih', 'sebuah', 'agar', 'lebih', 'sangat'
        ])
    
    def preprocess_text(self, text):
        """Text Preprocessing: Case folding, hapus non-alphabet, 
           tokenisasi, dan hapus stopwords"""

        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Case Folding
        text = text.lower()

        # Hapus karakter selain huruf, angka, dan spasi
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Tokenisasi
        tokens = text.split()

        # Hapus stopwords
        # Hapus stopwords tapi pertahankan semua angka
        tokens = [token for token in tokens 
          if token not in self.stopwords and (len(token) > 2 or token.isdigit())]

        return ' '.join(tokens)
    
    def load_documents(self):
        """Load dokumen dari file csv"""
        print("Loading dokumen dari folder dataset...")

        if not os.path.exists(self.dataset_path):
            print(f"Error: '{self.dataset_path}' folder tidak ditemukan")
            return False
        
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]

        if not csv_files:
            print(f"Error: file csv tidak ditemukan di '{self.dataset_path}'")
            return False
        
        print(f"Ditemukan {len(csv_files)} file CSV\n")

        for csv_file in csv_files:
            csv_path = os.path.join(self.dataset_path, csv_file)
            dataset_name = csv_file.replace('.csv', '')

            print(f"Loading: {csv_file}")

            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                print(f" Jumlah data: {len(df)}")

                for idx, row in df.iterrows():
                    judul = str(row['judul']) if pd.notna(row['judul']) else ""
                    konten = str(row['konten']) if pd.notna(row['konten']) else ""
                    full_text = f"{judul} {konten}"

                    if full_text.strip():
                        self.documents.append({
                            'id': f"{dataset_name}_{idx}",
                            'source': dataset_name,
                            'judul': judul,
                            'konten': konten,
                            'full_text': full_text,
                            'preprocessed': self.preprocess_text(full_text)
                        })
                
                print(f" Berhasil load {len(df)} dokumen \n")
            
            except Exception as e:
                print(f" Error: {e}\n")

        print(f"{'='*50}")
        print(f'Total dokumen: {len(self.documents)}')
        print(f"{'='*50}")

        return len(self.documents) > 0
    
    def create_whoosh_index(self):
        """Membuat Whoosh index"""
        print("\nMembuat Whoosh index...")
        
        if os.path.exists(self.index_dir):
            import shutil
            shutil.rmtree(self.index_dir)
        
        os.makedirs(self.index_dir)
        
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            source=TEXT(stored=True),
            judul=TEXT(stored=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            preprocessed=TEXT(stored=True)
        )
        
        ix = index.create_in(self.index_dir, schema)
        writer = ix.writer()
        
        for doc in self.documents:
            writer.add_document(
                doc_id=doc['id'],
                source=doc['source'],
                judul=doc['judul'],
                content=doc['full_text'],
                preprocessed=doc['preprocessed']
            )
        
        writer.commit()
        print("Whoosh index bebrhasil dibuat!")

    def create_bow_vectors(self):
        """Membuat Bag of Words vectors"""
        print("\nMembuat BoW vectors...")
        
        texts = [doc['preprocessed'] for doc in self.documents]
        self.vectorizer = CountVectorizer(max_features=5000, min_df=1)
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"BoW vectors brehasil dibuat: {self.doc_vectors.shape[1]} fitur")
    
    def search_query(self, query, top_k=5):
        """Search dan rank dokumen"""
        processed_query = self.preprocess_text(query)
        
        if not processed_query:
            print("Query kosong setelah preprocessing!")
            return []
        
        print(f"\nQuery: {query}")
        print(f"Processed: {processed_query}")
        
        # === TAMBAHKAN DEBUG INI ===
        print(f"\n[DEBUG] Query terms: {processed_query.split()}")

        # Cek apakah term ada di vocabulary
        query_terms = processed_query.split()
        for term in query_terms:
            if term in self.vectorizer.vocabulary_:
                print(f"  ✓ '{term}' ADA di vocabulary (index: {self.vectorizer.vocabulary_[term]})")
            else:
                print(f"  ✗ '{term}' TIDAK ADA di vocabulary")

        # Cek query vector
        query_vector = self.vectorizer.transform([processed_query])
        print(f"\n[DEBUG] Query vector sum: {query_vector.sum()}")
        print(f"[DEBUG] Query vector non-zero: {query_vector.nnz}")
        # === END DEBUG ===


        # Whoosh search
        ix = index.open_dir(self.index_dir)
        
        with ix.searcher() as searcher:
            query_parser = QueryParser("preprocessed", ix.schema)
            whoosh_query = query_parser.parse(processed_query)
            whoosh_results = searcher.search(whoosh_query, limit=50)
            
            if len(whoosh_results) == 0:
                print("\nDokumen tidak ditemukan.")
                return []
            
            candidate_ids = [hit['doc_id'] for hit in whoosh_results]
        
        # Cosine similarity ranking
        query_vector = self.vectorizer.transform([processed_query])
        results = []
        
        for doc_id in candidate_ids:
            doc_idx = next((i for i, d in enumerate(self.documents) if d['id'] == doc_id), None)
            
            if doc_idx is not None:
                doc_vector = self.doc_vectors[doc_idx]
                similarity = cosine_similarity(query_vector, doc_vector)[0][0]
                
                results.append({
                    'doc_id': doc_id,
                    'source': self.documents[doc_idx]['source'],
                    'judul': self.documents[doc_idx]['judul'],
                    'konten': self.documents[doc_idx]['konten'],
                    'similarity': similarity
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def display_results(self, results):
        """Menampilkan hasil pencarian"""
        if not results:
            print("\nHasil tidak ditemukan.")
            return
        
        print(f"\n{'='*80}")
        print(f"Hasil Top {len(results)}:")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Skor: {result['similarity']:.16f}")
            print(f"    Sumber: {result['source']}")
            print(f"    Judul: {result['judul'][:100]}...")
            print(f"    Konten: {result['konten'][:150]}...")
            print(f"{'-'*80}\n")

def main():
    """Main CLI interface"""
    ir_system = IRSystem()
    indexed = False
    
    while True:
        print("\n" + "="*40)
        print("=== INFORMATION RETRIEVAL SYSTEM ===")
        print("="*40)
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("="*40)
        
        choice = input("\nOpsi pilihan: ").strip()
        
        if choice == '1':
            if ir_system.load_documents():
                ir_system.create_whoosh_index()
                ir_system.create_bow_vectors()
                indexed = True
                print("\n✓ Dataset berhasil di load dan di index!")
            else:
                print("\n✗ Gagal untuk load dataset.")
        
        elif choice == '2':
            if not indexed:
                print("\n✗ Mohon load dan index dataset terlebih dahulu (opsi 1)!")
                continue
            
            query = input("\nMasukkan query pencarian: ").strip()
            
            if not query:
                print("Query tidak boleh kosong!")
                continue
            
            results = ir_system.search_query(query, top_k=5)
            ir_system.display_results(results)
        
        elif choice == '3':
            print("\nTerima kasih sudah menggunakan sistem IR, sampai jumpa!")
            break
        
        else:
            print("\n✗ Opsi invalid, masukkan opsi 1, 2, atau 3!")


if __name__ == "__main__":
    main()