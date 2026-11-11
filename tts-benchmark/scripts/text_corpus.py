"""Shared helpers for locating and parsing the text corpus.

Expected convention (the group's shared README.txt in the Drive folder):
one folder per person named "Imię Nazwisko" under data/texts/raw/ (or
data/texts/processed/ after preprocess_texts.py has run), each containing 5
UTF-8 .txt files named "<nrindeksu>-<N>-<slug>.txt":

    nrindeksu-1-info.txt             krótkie komunikaty informacyjne
    nrindeksu-2-qna.txt              pytania i odpowiedzi
    nrindeksu-3-numbers.txt          liczby, daty, godziny, skróty
    nrindeksu-4-long.txt             zdania złożone i dłuższe wypowiedzi
    nrindeksu-5-unique-foreign.txt   nazwy własne i wyrazy obcojęzyczne
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTS_RAW_DIR = PROJECT_ROOT / "data" / "texts" / "raw"
TEXTS_PROCESSED_DIR = PROJECT_ROOT / "data" / "texts" / "processed"

CATEGORY_BY_SLUG = {
    "info": "Krótki komunikat informacyjny",
    "qna": "Pytanie / odpowiedź",
    "numbers": "Liczby, daty, godziny, skróty",
    "long": "Zdanie złożone / dłuższa wypowiedź",
    "unique-foreign": "Nazwy własne i wyrazy obcojęzyczne",
}

FILENAME_PATTERN = re.compile(
    r"^(?P<nrindeksu>\d+)-(?P<n>\d)-(?P<slug>info|qna|numbers|long|unique-foreign)$"
)


class UnrecognizedFilenameError(ValueError):
    """Raised when a .txt filename doesn't match the agreed convention."""


@dataclass(frozen=True)
class TextItem:
    path: Path
    person: str  # "Imię Nazwisko" folder name
    nrindeksu: str
    n: int
    slug: str
    category: str

    @property
    def text_id(self) -> str:
        return f"{self.person}/{self.nrindeksu}-{self.n}-{self.slug}"


def parse_filename(stem: str) -> tuple[str, int, str]:
    """Parse "<nrindeksu>-<N>-<slug>" -> (nrindeksu, N, slug). Raises UnrecognizedFilenameError."""
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise UnrecognizedFilenameError(
            f"Filename {stem!r}.txt doesn't match the agreed "
            "'<nrindeksu>-<N>-<slug>.txt' convention (slug one of: "
            f"{', '.join(CATEGORY_BY_SLUG)})."
        )
    return match["nrindeksu"], int(match["n"]), match["slug"]


def resolve_texts_dir() -> Path:
    """Prefer data/texts/processed/ when it has content, else fall back to data/texts/raw/."""
    if TEXTS_PROCESSED_DIR.exists() and any(TEXTS_PROCESSED_DIR.glob("*/*.txt")):
        return TEXTS_PROCESSED_DIR
    return TEXTS_RAW_DIR


def iter_corpus(texts_dir: Path | None = None) -> list[TextItem]:
    """List all recognized text items under texts_dir (default: resolve_texts_dir())."""
    texts_dir = texts_dir if texts_dir is not None else resolve_texts_dir()

    items: list[TextItem] = []
    for path in sorted(texts_dir.glob("*/*.txt")):
        try:
            nrindeksu, n, slug = parse_filename(path.stem)
        except UnrecognizedFilenameError as exc:
            print(f"WARNING: {exc}")
            continue
        items.append(
            TextItem(
                path=path,
                person=path.parent.name,
                nrindeksu=nrindeksu,
                n=n,
                slug=slug,
                category=CATEGORY_BY_SLUG[slug],
            )
        )
    return items
