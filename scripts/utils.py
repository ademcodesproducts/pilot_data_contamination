"""Shared utilities for the contamination detection pipeline."""

import jsonlines
from pathlib import Path


def load_jsonl(path):
    with jsonlines.open(path) as r:
        return list(r)


def save_jsonl(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as w:
        w.write_all(items)


def load_math500(path="data/math500.jsonl"):
    return load_jsonl(path)


def load_train(project, paths=None):
    """Load training data for a given project name."""
    default_paths = {
        "s1": "data/s1k.jsonl",
        "tulu": "data/tulu_math.jsonl",
        "openthoughts": "data/openthoughts.jsonl",
    }
    p = (paths or default_paths)[project]
    return load_jsonl(p)
