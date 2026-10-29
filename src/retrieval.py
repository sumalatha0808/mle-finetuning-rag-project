"""Chunk Northstar policy sources, index them in Qdrant, and retrieve evidence.

Every chunk keeps its document ID, title, position, and text so citations can
be validated after generation. Questions whose best match scores below
``min_score`` abstain instead of using the nearest weak match.
"""

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
COLLECTION_NAME = "northstar_policies"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Chunk:
    """One piece of a policy document with its citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    position: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk with its cosine similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved evidence plus the answer-or-abstain decision."""

    question: str
    hits: list[SearchHit]
    abstain: bool

    @property
    def source_ids(self) -> list[str]:
        """Unique retrieved document IDs, best match first."""

        return list(dict.fromkeys(hit.chunk.document_id for hit in self.hits))


# ---------------------------------------------------------------------------
# Loading and chunking
# ---------------------------------------------------------------------------


def load_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> dict[str, str]:
    """Read every Markdown file; the filename stem is the stable document ID."""

    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown sources found in {directory}")
    return {path.stem: path.read_text(encoding="utf-8") for path in files}


def chunk_document(document_id: str, markdown: str, max_chars: int = 600) -> list[Chunk]:
    """Split a document on blank lines, merging short paragraphs up to max_chars.

    The first ``# Heading`` becomes the title and is not repeated as a chunk.
    Paragraph boundaries are kept so a policy rule is never cut mid-sentence.
    """

    title = document_id
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        block = " ".join(line.strip() for line in block.splitlines()).strip()
        if not block:
            continue
        if block.startswith("# ") and title == document_id:
            title = block[2:].strip()
            continue
        paragraphs.append(block.lstrip("#").strip())

    merged: list[str] = []
    for paragraph in paragraphs:
        if merged and len(merged[-1]) + len(paragraph) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]} {paragraph}"
        else:
            merged.append(paragraph)

    return [
        Chunk(
            chunk_id=f"{document_id}#{position}",
            document_id=document_id,
            title=title,
            position=position,
            text=text,
        )
        for position, text in enumerate(merged)
    ]


def build_chunks(directory: Path = KNOWLEDGE_BASE_DIR, max_chars: int = 600) -> list[Chunk]:
    """Chunk every policy source in the knowledge base."""

    chunks: list[Chunk] = []
    for document_id, markdown in load_documents(directory).items():
        chunks.extend(chunk_document(document_id, markdown, max_chars))
    return chunks


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------


class Retriever:
    """Dense retriever over a Qdrant collection with a no-answer threshold."""

    def __init__(
        self,
        embedding_model: str | None = None,
        embedding_revision: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        storage_path: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.top_k = top_k if top_k is not None else int(os.getenv("TOP_K", "3"))
        self.min_score = (
            min_score if min_score is not None else float(os.getenv("MIN_SCORE", "0.30"))
        )
        # Pin the default model's revision so reruns embed with the same weights.
        self.embedding_revision = (
            embedding_revision
            or os.getenv("EMBEDDING_REVISION")
            or (DEFAULT_EMBEDDING_REVISION if self.embedding_model == DEFAULT_EMBEDDING_MODEL else None)
        )
        storage_path = storage_path or os.getenv("VECTOR_STORAGE_PATH") or None

        self.model = SentenceTransformer(self.embedding_model, revision=self.embedding_revision)
        self.dimension = self.model.get_embedding_dimension()
        self.client = (
            QdrantClient(path=storage_path) if storage_path else QdrantClient(":memory:")
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()

    def index(self, chunks: list[Chunk]) -> None:
        """Rebuild the collection from scratch so no stale policy chunks remain."""

        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=self.dimension, distance=models.Distance.COSINE
            ),
        )
        vectors = self._embed([f"{c.title}\n{c.text}" for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=asdict(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, question: str, limit: int | None = None) -> list[SearchHit]:
        """Return the top matches regardless of score (used for threshold analysis)."""

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=self._embed([question])[0],
            limit=limit or self.top_k,
            with_payload=True,
        )
        return [
            SearchHit(chunk=Chunk(**point.payload), score=float(point.score))
            for point in response.points
        ]

    def retrieve(self, question: str) -> RetrievalResult:
        """Retrieve evidence; abstain when the best score is below min_score.

        Hits under the threshold are dropped so weak matches can never be cited.
        """

        hits = [hit for hit in self.search(question) if hit.score >= self.min_score]
        return RetrievalResult(question=question, hits=hits, abstain=not hits)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Fraction of expected sources present in the retrieved document IDs."""

    if not expected_ids:
        return 1.0
    return len(set(retrieved_ids) & set(expected_ids)) / len(set(expected_ids))


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1 / rank of the first relevant document, or 0 if none was retrieved."""

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in expected_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(retriever: Retriever, cases: list[dict]) -> dict:
    """Compute recall@k and MRR on supported cases and abstention accuracy on all.

    Ranking metrics ignore the threshold so they measure the embedding ranking
    itself; abstention accuracy uses the thresholded ``retrieve`` decision.
    """

    recalls, ranks, correct_decisions = [], [], 0
    per_case = []
    for case in cases:
        hits = retriever.search(case["question"])
        ranked_ids = list(dict.fromkeys(hit.chunk.document_id for hit in hits))
        top_score = hits[0].score if hits else 0.0
        abstained = retriever.retrieve(case["question"]).abstain
        correct_decisions += abstained == case["should_abstain"]

        row = {
            "id": case["id"],
            "top_score": round(top_score, 3),
            "retrieved": ranked_ids,
            "expected": case["expected_sources"],
            "should_abstain": case["should_abstain"],
            "abstained": abstained,
        }
        if not case["should_abstain"]:
            row["recall"] = recall_at_k(ranked_ids, case["expected_sources"])
            row["rr"] = reciprocal_rank(ranked_ids, case["expected_sources"])
            recalls.append(row["recall"])
            ranks.append(row["rr"])
        per_case.append(row)

    return {
        "top_k": retriever.top_k,
        "min_score": retriever.min_score,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(ranks) / len(ranks) if ranks else 0.0,
        "decision_accuracy": correct_decisions / len(cases) if cases else 0.0,
        "per_case": per_case,
    }


if __name__ == "__main__":
    import json

    chunks = build_chunks()
    print(f"Indexed {len(chunks)} chunks from {len({c.document_id for c in chunks})} documents")
    for chunk in chunks[:3]:
        print(f"  {chunk.chunk_id} | {chunk.title} | {chunk.text[:70]}...")

    retriever = Retriever()
    retriever.index(chunks)

    cases = json.loads(Path("data/evaluation/cases.json").read_text(encoding="utf-8"))
    report = evaluate_retrieval(retriever, cases)
    print(
        f"\ntop_k={report['top_k']} min_score={report['min_score']}  "
        f"recall@k={report['recall_at_k']:.3f}  MRR={report['mrr']:.3f}  "
        f"decision_acc={report['decision_accuracy']:.3f}\n"
    )
    for row in report["per_case"]:
        flag = "ABSTAIN" if row["abstained"] else "answer "
        print(
            f"  {row['id']:<32} score={row['top_score']:.3f} {flag} "
            f"(should_abstain={row['should_abstain']}) retrieved={row['retrieved']}"
        )
