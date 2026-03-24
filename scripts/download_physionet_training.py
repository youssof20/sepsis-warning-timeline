"""
Download full PhysioNet Challenge 2019 training_setA and training_setB .psv files.

Parses the public directory index pages and downloads missing files (resume-safe).
Run from project root: python scripts/download_physionet_training.py
"""

from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INDEX = {
    "training_setA": (
        "https://physionet.org/static/published-projects/challenge-2019/1.0.0/"
        "training/training_setA/"
    ),
    "training_setB": (
        "https://physionet.org/static/published-projects/challenge-2019/1.0.0/"
        "training/training_setB/"
    ),
}
BASE_FILE = {
    "training_setA": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/",
    "training_setB": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/",
}

PSV_RE = re.compile(r"(p\d+\.psv)")


def _fetch_index(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def _list_psv_names(html: str) -> list[str]:
    names = sorted(set(PSV_RE.findall(html)))
    return [n for n in names if n.endswith(".psv")]


def _download_one(url: str, dest: Path, retries: int = 4) -> tuple[str, str | None]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest.name, None
    last_err: str | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest.name, None
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2**attempt, 30))
    return dest.name, last_err


def download_folder(name: str, workers: int = 12) -> None:
    html = _fetch_index(INDEX[name])
    files = _list_psv_names(html)
    base = BASE_FILE[name]
    out_dir = DATA_DIR / name
    print(f"{name}: {len(files)} files listed in index -> {out_dir}", flush=True)

    jobs = [(base + fn, out_dir / fn) for fn in files]
    errors: list[str] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, u, p): (u, p) for u, p in jobs}
        for fut in as_completed(futs):
            fn, err = fut.result()
            done += 1
            if err:
                errors.append(f"{fn}: {err}")
            if done % 500 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(f"  ... {done}/{len(jobs)} ({elapsed:.0f}s)", flush=True)

    if errors:
        print(f"WARN: {len(errors)} files failed after retries (re-run script to resume).", file=sys.stderr)
        for e in errors[:30]:
            print(e, file=sys.stderr)
        if len(errors) > 30:
            print(f"... and {len(errors) - 30} more", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    download_folder("training_setA")
    download_folder("training_setB")
    print("Download finished.", flush=True)


if __name__ == "__main__":
    main()
