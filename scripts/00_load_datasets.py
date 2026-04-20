"""
00_load_datasets.py

Downloads and caches:
- MATH-500 from HuggingFace
- s1K from HuggingFace
- Tülu 3 math SFT subset from HuggingFace
- OpenThoughts-114K from HuggingFace (20k reproducible sample)

Each dataset is saved as a JSONL file with a consistent schema.
Run this script once before anything else.
"""

from datasets import load_dataset
import jsonlines
import os
import random
from pathlib import Path

os.makedirs("data", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("results/ngram_hits", exist_ok=True)
os.makedirs("results/embedding_candidates", exist_ok=True)
os.makedirs("results/judge_results", exist_ok=True)


# ── MATH-500 ──────────────────────────────────────────────────────────────────

def load_math500():
    out_path = Path("data/math500.jsonl")
    if out_path.exists():
        print("MATH-500 already cached, skipping.")
        return

    print("Loading MATH-500...")
    attempts = [
        ("HuggingFaceH4/MATH-500",   "test",  "subject", "answer",  False),
        ("TIGER-Lab/MATH-500",        "test",  "subject", "answer",  False),
        ("EleutherAI/hendrycks_math", "test",  "subject", "answer",  True),
        ("competition_math",          "test",  "type",    "answer",  True),
    ]

    ds = None
    subject_field = "subject"
    answer_field = "answer"
    trim = False
    for dataset_id, split, sf, af, should_trim in attempts:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split=split)
            subject_field, answer_field, trim = sf, af, should_trim
            print(f"  OK: {len(ds)} items")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load MATH-500 from any known source.")

    items = []
    for i, row in enumerate(ds):
        level_raw = row.get("level", "")
        try:
            level = int(str(level_raw).replace("Level ", ""))
        except (ValueError, TypeError):
            level = -1
        items.append({
            "math500_id": f"math500_{i:04d}",
            "problem": row.get("problem", row.get("question", "")),
            "solution": row.get("solution", ""),
            "answer": row.get(answer_field, ""),
            "subject": row.get(subject_field, ""),
            "level": level,
        })

    if trim and len(items) > 500:
        print(f"  Trimming {len(items)} → 500")
        items = items[:500]

    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(items)
    print(f"  Saved {len(items)} MATH-500 items")


# ── S1K ───────────────────────────────────────────────────────────────────────

def load_s1k():
    out_path = Path("data/s1k.jsonl")
    if out_path.exists():
        print("s1K already cached, skipping.")
        return

    print("Loading s1K...")
    ds = None
    for dataset_id in ["simplescaling/s1K", "simplescaling/s1k"]:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split="train")
            print(f"  OK: {len(ds)} items, columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load s1K.")

    items = []
    for i, row in enumerate(ds):
        items.append({
            "train_id": f"s1k_{i:04d}",
            # s1K HF release uses 'question' not 'problem'
            "problem": row.get("problem") or row.get("question") or row.get("prompt", ""),
            "solution": row.get("solution") or row.get("response") or row.get("cot", ""),
            "source": row.get("source", row.get("source_type", "unknown")),
            "dataset": "s1k",
        })

    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(items)
    print(f"  Saved {len(items)} s1K items")


# ── TÜLU 3 MATH SUBSET ────────────────────────────────────────────────────────

# Sources in tulu-3-sft-mixture that are math-related
TULU_MATH_SOURCES = {
    "numina_math_tir", "numinamath_tir", "numina", "numinamath",
    "math", "gsm8k", "gsm", "metamath", "deepmind_math",
    "orca_math", "camel_math", "mathinstruct", "wizard_math",
    "open_platypus", "tulu_v3.1_math_mixture",
}


def _is_math_source(source: str) -> bool:
    s = source.lower()
    return any(kw in s for kw in ("math", "numina", "gsm", "platypus", "wizard", "orca", "camel"))


def _extract_messages(messages):
    """Extract (problem, solution) from a conversation messages list."""
    problem, solution = "", ""
    for msg in messages:
        role    = msg.get("role") or msg.get("from", "")
        content = msg.get("content") or msg.get("value", "")
        if isinstance(content, list):
            # Some datasets use content as a list of {"type":..., "text":...}
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if role in ("user", "human") and not problem:
            problem = content
        elif role in ("assistant", "gpt") and not solution:
            solution = content
    return problem, solution


def load_tulu():
    out_path = Path("data/tulu_math.jsonl")
    if out_path.exists():
        print("Tülu already cached, skipping.")
        return

    print("Loading Tülu 3 math subset...")
    print("  NOTE: may require HF auth. Set HF_TOKEN env var if this fails.")

    tulu_candidates = [
        ("ai2-adapt-dev/numinamath_tir_math_decontaminated", "train"),
        ("allenai/tulu-3-sft-mixture",                        "train"),
    ]

    ds = None
    for dataset_id, split in tulu_candidates:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split=split)
            print(f"  OK: {len(ds)} items, columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not load Tülu math dataset.\n"
            "Try: huggingface-cli login\n"
            "Then set HF_TOKEN env var."
        )

    items = []
    for i, row in enumerate(ds):
        source = str(row.get("source", ""))

        # allenai/tulu-3-sft-mixture uses a 'messages' conversation column
        messages = row.get("messages", [])
        if messages:
            if not _is_math_source(source):
                continue  # skip non-math sources from the full mixture
            problem, solution = _extract_messages(messages)
        else:
            problem  = row.get("problem") or row.get("question") or row.get("input", "")
            solution = row.get("solution") or row.get("output", row.get("response", ""))

        if not problem.strip():
            continue

        items.append({
            "train_id": f"tulu_{i:06d}",
            "problem":  problem,
            "solution": solution,
            "source":   source or "numinamath",
            "dataset":  "tulu",
        })

    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(items)
    print(f"  Saved {len(items)} Tülu math items")


# ── OPENTHOUGHTS-114K ─────────────────────────────────────────────────────────

def load_openthoughts():
    out_path = Path("data/openthoughts.jsonl")
    if out_path.exists():
        print("OpenThoughts already cached, skipping.")
        return

    print("Loading OpenThoughts-114K...")
    ds = None
    for dataset_id in ["open-thoughts/OpenThoughts-114k", "open-thoughts/OpenThoughts-114K"]:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split="train")
            print(f"  OK: {len(ds)} items, columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load OpenThoughts-114K.")

    # Reproducible 20k sample
    random.seed(42)
    indices = random.sample(range(len(ds)), min(20000, len(ds)))
    sample = ds.select(indices)

    items = []
    for i, row in enumerate(sample):
        # Extract problem from conversation messages
        messages = row.get("conversations") or row.get("messages", [])
        problem = ""
        solution = ""
        if messages:
            for msg in messages:
                role = msg.get("role") or msg.get("from", "")
                content = msg.get("content") or msg.get("value", "")
                if role in ("user", "human") and not problem:
                    problem = content
                elif role in ("assistant", "gpt") and not solution:
                    solution = content
        else:
            # Flat schema fallback
            problem = row.get("problem") or row.get("question", "")
            solution = row.get("solution") or row.get("response", "")

        if not problem.strip():
            continue

        items.append({
            "train_id": f"ot_{indices[i]:06d}",
            "problem": problem,
            "solution": solution[:1000],  # truncate long CoT solutions
            "source": row.get("source", "openthoughts"),
            "dataset": "openthoughts",
            "original_index": indices[i],
        })

    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(items)
    print(f"  Saved {len(items)} OpenThoughts items (sampled from {len(ds)})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_math500()
    load_s1k()
    load_tulu()
    load_openthoughts()
    print("\nAll datasets loaded. Ready for Stage 1.")
