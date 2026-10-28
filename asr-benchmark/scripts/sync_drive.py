"""Sync audio/transcript files from a public Google Drive folder into data/raw/.

Uses the Google Drive API v3 with an API key only (no OAuth), since the source
folder is shared as "anyone with the link". Downloads only new or changed files
(compared by md5Checksum against local MD5), preserves the speaker subfolder
structure, and never deletes local files that were removed from Drive.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps"
FOLDER_MIME_TYPE = f"{GOOGLE_NATIVE_MIME_PREFIX}.folder"

FOLDER_URL_PATTERN = re.compile(r"/folders/([a-zA-Z0-9_-]+)")

# Google's abuse-detection flags the default "python-requests/x.y" User-Agent on
# high-volume anonymous traffic, so we send a normal browser-like one instead.
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
)

# Small pause between downloads to avoid tripping rate/abuse limits.
DOWNLOAD_DELAY_SECONDS = 0.5

# Retry/backoff for transient errors (rate limiting, anti-abuse blocks, server errors).
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0


class ConfigError(RuntimeError):
    """Raised when required configuration (.env) is missing or invalid."""


class DriveApiError(RuntimeError):
    """Raised when the Google Drive API returns an error response."""


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str
    md5_checksum: str | None


@dataclass
class SyncStats:
    downloaded: int = 0
    skipped_current: int = 0
    warnings: list[str] = field(default_factory=list)
    speaker_folders: int = 0
    wav_count: int = 0
    txt_count: int = 0
    unpaired: list[str] = field(default_factory=list)


def extract_folder_id(folder_url: str) -> str:
    """Extract the Drive folder ID from a folder share URL.

    Expects a URL like https://drive.google.com/drive/folders/<ID> optionally
    followed by a query string (e.g. ?usp=sharing).
    """
    match = FOLDER_URL_PATTERN.search(folder_url)
    if not match:
        raise ConfigError(
            "Could not extract a folder ID from GDRIVE_FOLDER_URL "
            f"({folder_url!r}). Expected a URL like "
            "https://drive.google.com/drive/folders/<ID>."
        )
    return match.group(1)


def load_config() -> tuple[str, str]:
    """Load GDRIVE_FOLDER_URL and GDRIVE_API_KEY from .env, returning (folder_id, api_key)."""
    load_dotenv()

    folder_url = os.environ.get("GDRIVE_FOLDER_URL", "").strip()
    api_key = os.environ.get("GDRIVE_API_KEY", "").strip()

    if not folder_url:
        raise ConfigError(
            "GDRIVE_FOLDER_URL is not set. Copy .env.example to .env and fill it in "
            "(see docs/GDRIVE_API_KEY.md)."
        )
    if not api_key:
        raise ConfigError(
            "GDRIVE_API_KEY is not set. Copy .env.example to .env and fill it in "
            "(see docs/GDRIVE_API_KEY.md)."
        )

    folder_id = extract_folder_id(folder_url)
    return folder_id, api_key


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Perform an HTTP request, retrying with exponential backoff on transient errors.

    Retries on rate limiting, anti-abuse blocks, and server errors (see
    RETRYABLE_STATUS_CODES). Non-retryable errors (e.g. invalid API key) are
    returned immediately so the caller's error handling can report them.
    """
    last_response: requests.Response | None = None

    for attempt in range(MAX_RETRIES + 1):
        response = SESSION.request(method, url, **kwargs)
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        last_response = response
        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE_SECONDS * (2**attempt)
            print(
                f"  Transient error ({response.status_code}), retrying in "
                f"{delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(delay)

    assert last_response is not None
    return last_response


def list_folder_children(folder_id: str, api_key: str) -> list[DriveFile]:
    """List all children of a Drive folder, handling pagination."""
    files: list[DriveFile] = []
    page_token: str | None = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType, md5Checksum)",
            "pageSize": 1000,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        response = request_with_retry(
            "GET", f"{DRIVE_API_BASE}/files", params=params, timeout=30
        )
        handle_api_errors(response)
        payload = response.json()

        for item in payload.get("files", []):
            files.append(
                DriveFile(
                    id=item["id"],
                    name=item["name"],
                    mime_type=item["mimeType"],
                    md5_checksum=item.get("md5Checksum"),
                )
            )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return files


def handle_api_errors(response: requests.Response) -> None:
    if response.status_code == 403:
        raise DriveApiError(
            "Google Drive API returned 403 Forbidden.\n"
            "Check that:\n"
            "  - GDRIVE_API_KEY in .env is correct\n"
            "  - the Google Drive API is enabled in your Google Cloud project\n"
            "See docs/GDRIVE_API_KEY.md for setup steps.\n"
            f"Raw response: {response.text}"
        )
    if not response.ok:
        raise DriveApiError(
            f"Google Drive API request failed: {response.status_code} {response.text}"
        )


def local_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(file_id: str, api_key: str, destination: Path) -> None:
    params = {"alt": "media", "key": api_key}
    response = request_with_retry(
        "GET", f"{DRIVE_API_BASE}/files/{file_id}", params=params, stream=True, timeout=60
    )
    handle_api_errors(response)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)

    time.sleep(DOWNLOAD_DELAY_SECONDS)


def sync_folder(
    folder_id: str,
    api_key: str,
    dest_dir: Path,
    stats: SyncStats,
    dry_run: bool,
    is_root: bool = True,
) -> None:
    """Recursively sync a Drive folder into dest_dir, preserving subfolder structure."""
    children = list_folder_children(folder_id, api_key)

    if not is_root:
        stats.speaker_folders += 1

    for child in children:
        if child.mime_type == FOLDER_MIME_TYPE:
            sync_folder(
                child.id,
                api_key,
                dest_dir / child.name,
                stats,
                dry_run,
                is_root=False,
            )
            continue

        if child.mime_type.startswith(GOOGLE_NATIVE_MIME_PREFIX):
            warning = (
                f"Skipping native Google file (no binary content to download): "
                f"{dest_dir / child.name}"
            )
            stats.warnings.append(warning)
            print(f"WARNING: {warning}")
            continue

        local_path = dest_dir / child.name
        if local_path.suffix.lower() == ".wav":
            stats.wav_count += 1
        elif local_path.suffix.lower() == ".txt":
            stats.txt_count += 1

        needs_download = True
        if local_path.exists() and child.md5_checksum:
            if local_md5(local_path) == child.md5_checksum:
                needs_download = False

        if not needs_download:
            stats.skipped_current += 1
            continue

        if dry_run:
            print(f"Would download: {local_path}")
            stats.downloaded += 1
            continue

        print(f"Downloading: {local_path}")
        download_file(child.id, api_key, local_path)
        stats.downloaded += 1


def find_unpaired_files(dest_dir: Path) -> list[str]:
    """Find .wav files without a matching .txt (and vice versa) under dest_dir."""
    unpaired: list[str] = []
    for wav_path in sorted(dest_dir.rglob("*.wav")):
        if not wav_path.with_suffix(".txt").exists():
            unpaired.append(str(wav_path))
    for txt_path in sorted(dest_dir.rglob("*.txt")):
        if not txt_path.with_suffix(".wav").exists():
            unpaired.append(str(txt_path))
    return unpaired


def print_summary(stats: SyncStats, dest_dir: Path) -> None:
    stats.unpaired = find_unpaired_files(dest_dir)

    print("\n--- Sync summary ---")
    print(f"Downloaded: {stats.downloaded}")
    print(f"Skipped (already up to date): {stats.skipped_current}")
    print(f"Warnings: {len(stats.warnings)}")
    print(f"Speaker folders: {stats.speaker_folders}")
    print(f"WAV files: {stats.wav_count}")
    print(f"TXT files: {stats.txt_count}")
    if stats.unpaired:
        print(f"Unpaired files ({len(stats.unpaired)}):")
        for path in stats.unpaired:
            print(f"  - {path}")
    else:
        print("Unpaired files: none")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync a public Google Drive folder into data/raw/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading anything.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        folder_id, api_key = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    stats = SyncStats()

    try:
        sync_folder(folder_id, api_key, args.dest, stats, dry_run=args.dry_run)
    except DriveApiError as exc:
        print(f"Drive API error: {exc}", file=sys.stderr)
        return 1

    print_summary(stats, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
