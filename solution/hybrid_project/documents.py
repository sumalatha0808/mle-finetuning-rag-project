"""Load policy documents and create paragraph-aware retrieval chunks."""

from pathlib import Path

from .models import Chunk, SourceDocument


def load_documents(source_dir: Path) -> list[SourceDocument]:
    """Load Markdown policies and derive stable source IDs from filenames."""

    documents: list[SourceDocument] = []
    for path in sorted(source_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8").strip()
        lines = raw_text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path} must start with a level-one Markdown title")
        documents.append(
            SourceDocument(
                document_id=path.stem,
                title=lines[0].removeprefix("# ").strip(),
                text="\n".join(lines[1:]).strip(),
                path=path,
            )
        )
    if not documents:
        raise ValueError(f"No Markdown policy documents found in {source_dir}")
    return documents


def _window_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split an overlong paragraph while retaining boundary context."""

    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words")
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), step)
        if words[start : start + max_words]
    ]


def chunk_documents(
    documents: list[SourceDocument],
    max_words: int = 90,
    overlap_words: int = 15,
) -> list[Chunk]:
    """Create citable chunks while keeping short policy paragraphs intact."""

    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = [
            paragraph.replace("\n", " ").strip()
            for paragraph in document.text.split("\n\n")
            if paragraph.strip()
        ]
        position = 0
        for paragraph in paragraphs:
            for passage in _window_text(paragraph, max_words, overlap_words):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-{position}",
                        document_id=document.document_id,
                        title=document.title,
                        text=passage,
                        position=position,
                    )
                )
                position += 1
    return chunks
