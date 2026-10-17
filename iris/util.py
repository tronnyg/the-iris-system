from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("iris")


def setup_logging(verbose: bool = False, logfile: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def now() -> float:
    return time.time()


def today(offset_days: int = 0) -> str:
    return (_dt.date.today() + _dt.timedelta(days=offset_days)).isoformat()


def iso(ts: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(ts if ts is not None else time.time()).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping bad json line in %s", path)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


_word_re = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return _word_re.findall(text.lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (Levenshtein over words). 0.0 = identical."""
    r, h = words(reference), words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def similarity(a: str, b: str) -> float:
    """Word-set Jaccard similarity, used to spot the user repeating themselves."""
    wa, wb = set(words(a)), set(words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
