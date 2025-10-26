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

        # Hapus non-alphabet
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Tokenisasi
        tokens = text.split()

        # Hapus stopwords
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]

        return ' '.join(tokens)
    
    def load_documents(self):
        """Load do"""