"""Qdrant-backed storage that retains citation metadata in every point."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .embeddings import VectorMatrix
from .models import Chunk, SearchHit


class VectorStore:
    """Rebuild and search a local Qdrant collection."""

    collection_name = "northstar_hybrid_project"

    def __init__(self, dimension: int, storage_path: Path | None = None) -> None:
        self.client = (
            QdrantClient(path=str(storage_path))
            if storage_path is not None
            else QdrantClient(":memory:")
        )
        self.dimension = dimension

    def replace(self, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        """Replace the complete collection to avoid stale policy chunks."""

        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=[float(value) for value in vector],
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int) -> list[SearchHit]:
        """Return scored chunks reconstructed from Qdrant payloads."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload["document_id"]),
                    title=str(payload["title"]),
                    text=str(payload["text"]),
                    position=int(payload["position"]),
                    score=float(point.score),
                )
            )
        return hits
