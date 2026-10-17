"""Download and validate the Tiny Shakespeare dataset."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.utils import PROJECT_ROOT, ensure_dir, load_config, save_json


DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_text_file(path: Path, encoding: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Dataset file is empty: {path}")
    text = path.read_text(encoding=encoding)
    if not text.strip():
        raise ValueError(f"Dataset decoded but contains no visible text: {path}")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": size,
        "characters": len(text),
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "sha256": _sha256(path),
        "encoding": encoding,
        "decoded": True,
        "non_empty": True,
    }


def download_dataset(config: dict | None = None, force: bool = False) -> dict:
    config = config or load_config()
    url = config["dataset"]["source_url"]
    encoding = config["dataset"].get("encoding", "utf-8")
    dest = PROJECT_ROOT / config["paths"]["raw_text"]
    ensure_dir(dest.parent)
    metadata_path = dest.parent / "dataset_metadata.json"

    reused = False
    if dest.exists() and not force:
        try:
            stats = _validate_text_file(dest, encoding)
            reused = True
            status = "validated_existing"
        except (OSError, UnicodeError, ValueError):
            stats = None
            status = "re_download_required"
    else:
        stats = None
        status = "download_required"

    if not reused:
        tmp_path = dest.with_suffix(".tmp")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ict4403-assignment2"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                http_status = getattr(response, "status", 200)
            if not data:
                raise ValueError("Downloaded payload was empty.")
            tmp_path.write_bytes(data)
            tmp_path.replace(dest)
            stats = _validate_text_file(dest, encoding)
            status = f"downloaded_http_{http_status}"
        except urllib.error.HTTPError as exc:
            status = f"http_error_{exc.code}"
            raise RuntimeError(f"HTTP error while downloading dataset: {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            status = f"connection_error_{exc.reason}"
            raise RuntimeError(f"Connection error while downloading dataset: {exc.reason}") from exc
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    metadata = {
        "dataset_name": config["dataset"]["name"],
        "source_url": url,
        "download_or_validation_status": status,
        "reused_existing_file": reused,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    save_json(metadata_path, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Tiny Shakespeare.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid file exists.")
    parser.add_argument("--config", default=None, help="Optional path to config.yaml")
    args = parser.parse_args()
    try:
        metadata = download_dataset(load_config(args.config), force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Dataset ready.")
    print(f"  path: {metadata['path']}")
    print(f"  bytes: {metadata['bytes']}")
    print(f"  characters: {metadata['characters']}")
    print(f"  status: {metadata['download_or_validation_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
