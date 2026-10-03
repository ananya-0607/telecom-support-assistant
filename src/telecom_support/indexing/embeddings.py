"""CPU embeddings; downloads weights once, never sends source text to an API."""
from telecom_support.ingestion.chunking import TOKENIZER_MODEL, token_count

class LocalEmbedder:
    def __init__(self, cache_dir):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(TOKENIZER_MODEL, device="cpu", cache_folder=str(cache_dir))
        self.tokenizer = self.model.tokenizer

    def encode(self, texts, batch_size=16):
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must contain text.")
        if any(token_count(self.tokenizer, text) > self.model.max_seq_length for text in texts):
            raise ValueError("Embedding input exceeds model limit; split it before encoding.")
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
