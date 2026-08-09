"""
Local Vector Store & Cosine Similarity Indexer.
Provides fast semantic document retrieval over H100 Knowledge Base.
"""

import math
import re
from .knowledge_base import H100_KNOWLEDGE_DOCUMENTS


class LocalVectorStore:

    def __init__(self, documents: list[dict] = None):
        self.documents = documents or H100_KNOWLEDGE_DOCUMENTS
        self.vocab = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        return [w for w in words if len(w) > 2]

    def _build_index(self):
        # Build vocabulary across all documents
        doc_tokens_list = []
        df_counts = {}

        for doc in self.documents:
            full_text = f"{doc['title']} {doc['category']} {doc['content']}"
            tokens = self._tokenize(full_text)
            doc_tokens_list.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] = df_counts.get(t, 0) + 1

        # Build vocabulary mapping
        self.vocab = {word: idx for idx, word in enumerate(sorted(df_counts.keys()))}
        num_docs = len(self.documents)

        # Build TF-IDF vectors for documents
        self.doc_vectors = []
        for tokens in doc_tokens_list:
            vec = [0.0] * len(self.vocab)
            tf = {}
            for t in tokens:
                if t in self.vocab:
                    tf[t] = tf.get(t, 0) + 1

            doc_len = max(1, len(tokens))
            for t, count in tf.items():
                idx = self.vocab[t]
                idf = math.log((num_docs + 1) / (df_counts[t] + 1)) + 1.0
                vec[idx] = (count / doc_len) * idf

            # Normalize vector
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            norm_vec = [v / norm for v in vec]
            self.doc_vectors.append(norm_vec)

    def _query_vector(self, query: str) -> list[float]:
        tokens = self._tokenize(query)
        vec = [0.0] * len(self.vocab)
        tf = {}
        for t in tokens:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1

        doc_len = max(1, len(tokens))
        for t, count in tf.items():
            idx = self.vocab[t]
            vec[idx] = count / doc_len

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Performs cosine similarity search and returns top_k matching documents."""
        q_vec = self._query_vector(query)
        results = []

        for idx, d_vec in enumerate(self.doc_vectors):
            score = sum(q_vec[i] * d_vec[i] for i in range(len(q_vec)))
            doc = self.documents[idx].copy()
            doc["score"] = round(float(score), 4)
            results.append(doc)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
