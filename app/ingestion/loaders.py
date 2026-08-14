"""Document loaders for the three supported formats: PDF, DOCX, and TXT/MD.

Text is extracted into a single normalized string. The heavy parsing
dependencies (``pypdf``, ``python-docx``) are imported lazily so the module
imports cheaply and only pulls in what a given format needs.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".md"}


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
}


def load_document(path: Path | str) -> tuple[str, str]:
    """Return ``(text, format)`` for a supported document.

    Raises ``ValueError`` for unsupported extensions.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    text = _LOADERS[suffix](path)
    return _normalize(text), suffix.lstrip(".")


def _normalize(text: str) -> str:
    # Collapse excessive whitespace while preserving paragraph breaks.
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()
