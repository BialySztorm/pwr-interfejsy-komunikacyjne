"""Normalize all texts in data/texts/raw/ to a common encoding in data/texts/processed/.

The corpus in data/texts/raw/ is hand-typed/pasted by many people, so it's a mix of
encodings (UTF-8, UTF-8 with BOM, occasionally Windows-1250), line endings (\\r\\n vs \\n),
trailing whitespace, and Unicode normalization forms (precomposed vs. decomposed
diacritics) — all of which could make two TTS systems receive subtly different input
for "the same" text. This converts every <person>/<file>.txt under data/texts/raw/ to
UTF-8 (no BOM), LF line endings, no trailing whitespace, NFC-normalized Unicode, and
writes it to the same relative path under data/texts/processed/.

Also (re)writes data/texts/processed/raport_zbioru.md with dataset-wide counts.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_corpus import CATEGORY_BY_SLUG, TEXTS_PROCESSED_DIR, TEXTS_RAW_DIR, iter_corpus  # noqa: E402

# Encodings to try in order when a file isn't valid UTF-8.
FALLBACK_ENCODINGS = ("utf-8-sig", "cp1250", "latin-1")


class DecodeError(RuntimeError):
    """Raised when a text file can't be decoded with any known encoding."""


def read_text_robust(path: Path) -> str:
    raw_bytes = path.read_bytes()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for encoding in FALLBACK_ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
            print(f"  (decoded as {encoding}, not plain UTF-8)")
            return text
        except UnicodeDecodeError:
            continue
    raise DecodeError(f"Could not decode {path} with any of: utf-8, {', '.join(FALLBACK_ENCODINGS)}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize all data/texts/raw/*/*.txt to a common encoding under data/texts/processed/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if the destination file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed without actually writing anything.",
    )
    return parser.parse_args(argv)


def compute_dataset_stats(raw_dir: Path) -> dict:
    """Liczy liczbę osób, tekstów per kategoria i słów/znaków, z data/texts/raw/."""
    items = iter_corpus(raw_dir)
    person_count = len({item.person for item in items})

    per_category_counts: dict[str, int] = {label: 0 for label in CATEGORY_BY_SLUG.values()}
    total_words = 0
    total_chars = 0
    for item in items:
        per_category_counts[item.category] += 1
        text = item.path.read_text(encoding="utf-8", errors="replace").strip()
        total_words += len(text.split())
        total_chars += len(text)

    return {
        "persons": person_count,
        "texts": len(items),
        "per_category_counts": per_category_counts,
        "total_words": total_words,
        "total_chars": total_chars,
    }


def write_dataset_report(stats: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Raport zbioru tekstów (data/texts/raw/)",
        "",
        f"Wygenerowano: {generated_at} (przez scripts/preprocess_texts.py)",
        "",
        f"- Liczba osób: {stats['persons']}",
        f"- Liczba tekstów (.txt): {stats['texts']}",
        "- Liczba tekstów per kategoria:",
    ]
    for category, count in stats["per_category_counts"].items():
        lines.append(f"  - {category}: {count}")
    lines += [
        f"- Łączna liczba słów: {stats['total_words']}",
        f"- Łączna liczba znaków: {stats['total_chars']}",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Zapisano raport zbioru do pliku: {destination}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    source_files = sorted(TEXTS_RAW_DIR.glob("*/*.txt"))
    if not source_files:
        print(
            f"Brak plików .txt w {TEXTS_RAW_DIR}. Uruchom najpierw scripts/sync_drive.py.",
            file=sys.stderr,
        )
        return 1

    processed = skipped = errors = 0

    for source_path in source_files:
        rel_path = source_path.relative_to(TEXTS_RAW_DIR)
        destination_path = TEXTS_PROCESSED_DIR / rel_path

        if destination_path.exists() and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            print(f"Would process: {rel_path}")
            processed += 1
            continue

        print(f"Processing: {rel_path}")
        try:
            text = read_text_robust(source_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(normalize_text(text).encode("utf-8"))
            processed += 1
        except DecodeError as exc:
            print(f"Błąd dekodowania: {exc}", file=sys.stderr)
            errors += 1

    print("\n--- Podsumowanie ---")
    print(f"Przetworzone: {processed}")
    print(f"Pominięte (już istnieją w data/texts/processed): {skipped}")
    print(f"Błędy: {errors}")

    if not args.dry_run:
        stats = compute_dataset_stats(TEXTS_RAW_DIR)
        write_dataset_report(stats, TEXTS_PROCESSED_DIR / "raport_zbioru.md")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
