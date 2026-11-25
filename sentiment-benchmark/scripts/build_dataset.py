"""Parse data/raw/ into a single dataset.csv with reference sentiment/emotion labels.

Expected layout: data/raw/<Imie Nazwisko>/<index>-<sentiment>-<emotion>.txt, one UTF-8
utterance per file (may contain emoji). The reference labels come from the filename:

    sentiment in {positive, negative, neutral}
    emotion   in {joy, sadness, anger, fear, irony, none}

Validation problems are reported as warnings, never raised, so a single bad file from
the shared Drive folder doesn't block everyone else's data. Unknown labels, malformed
filenames, and empty files are skipped entirely. Duplicate texts are only flagged, not
skipped -- two people legitimately writing the same short utterance (e.g. "Dzien dobry")
is not an error, so both rows stay in dataset.csv; downstream analysis can filter them
out using the warning list in dataset_stats.json if that duplication turns out to matter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_EMOTIONS = {"joy", "sadness", "anger", "fear", "irony", "none"}

# "<index>-<sentiment>-<emotion>.txt", index is any non-empty run of non-"-" characters.
FILENAME_PATTERN = re.compile(r"^(?P<index>[^-]+)-(?P<sentiment>[a-z]+)-(?P<emotion>[a-z]+)\.txt$")

MIN_REQUIRED_TEXTS = 50

# Simple heuristic: a standalone Polish negation word anywhere in the text.
NEGATION_WORDS = ["nie", "nigdy", "żaden", "żadna", "żadne", "żadnych", "bez"]
NEGATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in NEGATION_WORDS) + r")\b",
    re.IGNORECASE | re.UNICODE,
)

# Emoji / pictographic Unicode ranges (covers the common emoji blocks).
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, supplemental symbols, emoticons, etc.
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]",
    re.UNICODE,
)


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    return slug or "unknown"


def has_negation(text: str) -> bool:
    return bool(NEGATION_PATTERN.search(text))


def has_emoji(text: str) -> bool:
    return bool(EMOJI_PATTERN.search(text))


def iter_raw_files(raw_dir: Path) -> list[tuple[str, Path]]:
    """Return (speaker_name, file_path) pairs for every .txt file under a speaker folder."""
    pairs: list[tuple[str, Path]] = []
    for speaker_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for txt_path in sorted(speaker_dir.glob("*.txt")):
            pairs.append((speaker_dir.name, txt_path))
    return pairs


def build_dataset(raw_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    seen_texts: dict[str, str] = {}  # normalized text -> first id that used it
    seq_by_speaker: Counter = Counter()  # per-speaker running count, used to build unique ids

    for speaker, txt_path in iter_raw_files(raw_dir):
        match = FILENAME_PATTERN.match(txt_path.name)
        if not match:
            warnings.append(f"Malformed filename, skipped: {txt_path}")
            continue

        index = match.group("index")
        sentiment = match.group("sentiment").lower()
        emotion = match.group("emotion").lower()

        if sentiment not in VALID_SENTIMENTS:
            warnings.append(
                f"Unknown sentiment label '{sentiment}' in {txt_path}, skipped."
            )
            continue
        if emotion not in VALID_EMOTIONS:
            warnings.append(f"Unknown emotion label '{emotion}' in {txt_path}, skipped.")
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            warnings.append(f"Empty file, skipped: {txt_path}")
            continue

        # The filename's <index> is the student's constant index number, repeated
        # across all 5 of their files -- it can't be used as a per-file unique id.
        # Use a running per-speaker sequence number instead.
        seq_by_speaker[speaker] += 1
        row_id = f"{slugify(speaker)}-{seq_by_speaker[speaker]}"

        normalized_text = " ".join(text.lower().split())
        duplicate_of = seen_texts.get(normalized_text)
        if duplicate_of is not None:
            warnings.append(
                f"Duplicate text in {txt_path} (id={row_id}), same as id={duplicate_of}."
            )
        else:
            seen_texts[normalized_text] = row_id

        rows.append(
            {
                "id": row_id,
                "speaker": speaker,
                "index": index,
                "text": text,
                "ref_sentiment": sentiment,
                "ref_emotion": emotion,
                "length_words": len(text.split()),
                "has_emoji": has_emoji(text),
                "has_negation": has_negation(text),
            }
        )

    return rows, warnings


def write_dataset_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "speaker",
        "index",
        "text",
        "ref_sentiment",
        "ref_emotion",
        "length_words",
        "has_emoji",
        "has_negation",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_stats(rows: list[dict], warnings: list[str]) -> dict:
    total = len(rows)
    speakers = {row["speaker"] for row in rows}
    stats = {
        "total_texts": total,
        "meets_minimum": total >= MIN_REQUIRED_TEXTS,
        "minimum_required": MIN_REQUIRED_TEXTS,
        "speaker_count": len(speakers),
        "sentiment_distribution": dict(Counter(row["ref_sentiment"] for row in rows)),
        "emotion_distribution": dict(Counter(row["ref_emotion"] for row in rows)),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    return stats


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset.csv and dataset_stats.json from data/raw/."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing per-speaker folders (default: data/raw).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset.csv",
        help="Output CSV path (default: data/dataset.csv).",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "dataset_stats.json",
        help="Output stats JSON path (default: results/dataset_stats.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.raw_dir.exists():
        print(f"Raw data directory not found: {args.raw_dir}", file=sys.stderr)
        return 1

    rows, warnings = build_dataset(args.raw_dir)
    for warning in warnings:
        print(f"WARNING: {warning}")

    write_dataset_csv(rows, args.out)

    stats = compute_stats(rows, warnings)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Wrote stats to {args.stats_out}")
    if not stats["meets_minimum"]:
        print(
            f"WARNING: only {stats['total_texts']} texts, below the minimum of "
            f"{MIN_REQUIRED_TEXTS}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
