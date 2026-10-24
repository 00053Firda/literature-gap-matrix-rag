import numpy as np
from sentence_transformers import SentenceTransformer


class PaperRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = {}      # {paper_id: [teks chunk]}
        self.embeddings = {}  # {paper_id: matriks vektor}

    def index(self, papers):
        """Mengubah semua chunk menjadi vektor dan menyimpannya."""
        for paper_id, chunks in papers.items():
            vectors = self.model.encode(
                chunks, normalize_embeddings=True, show_progress_bar=False
            )
            self.chunks[paper_id] = chunks
            self.embeddings[paper_id] = vectors

    def search(self, paper_id, query, top_k=4):
        """Mencari top_k chunk paling relevan dengan query, hanya dalam satu paper."""
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings[paper_id] @ query_vec
        best_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[paper_id][i] for i in best_idx]
