"""Build and query the shared retrieval layer used by both RAG systems."""

from .config import Settings
from .documents import chunk_documents, load_documents
from .embeddings import Embedder, SentenceTransformerEmbedder
from .models import Chunk, SearchHit
from .store import VectorStore


class Retriever:
    """One shared retriever for a fair baseline-versus-hybrid comparison."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model, revision=settings.embedding_revision
        )
        self.chunks: list[Chunk] = chunk_documents(
            load_documents(settings.knowledge_base_dir)
        )
        self.store = VectorStore(self.embedder.dimension, settings.storage_path)
        vectors = self.embedder.encode(
            [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        )
        self.store.replace(self.chunks, vectors)

    def search(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        """Retrieve the nearest policy chunks for a question."""

        limit = top_k or self.settings.top_k
        query_vector = self.embedder.encode([question])[0].tolist()
        return self.store.search(query_vector, limit=limit)
