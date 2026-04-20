"""
01_ngram_filter.py

Replicates each project's exact n-gram filter and identifies items
that survived despite having overlap with MATH-500.

Filter specs:
  s1:           8-gram, any overlap (>0 shared 8-grams)
  Tülu 3:       8-gram, >50% of test tokens must match a train token
  OpenThoughts: 13-gram, any overlap (>0 shared 13-grams)

Tokenizer: Qwen2-7B-Instruct for all (matches what projects used)

Output: results/ngram_hits/{project}_ngram_hits.jsonl
"""

from transformers import AutoTokenizer
import jsonlines
from tqdm import tqdm
import os
from pathlib import Path


print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")


def get_ngrams(text, n):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def compute_overlap(train_text, test_text, n, threshold_mode="any"):
    """
    Returns: (is_contaminated: bool, n_shared: int, coverage: float)

    threshold_mode:
      "any"     – flag if any shared n-gram exists (s1, OpenThoughts)
      "percent" – flag if >50% of test tokens are covered (Tülu 3)
    """
    train_ngrams = get_ngrams(train_text, n)
    test_ngrams  = get_ngrams(test_text, n)
    shared       = train_ngrams & test_ngrams
    n_shared     = len(shared)

    if threshold_mode == "any":
        return n_shared > 0, n_shared, 0.0

    # percent mode
    test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
    if not test_tokens:
        return False, 0, 0.0
    covered = set()
    for ngram in shared:
        for i in range(len(test_tokens) - n + 1):
            if tuple(test_tokens[i:i+n]) == ngram:
                covered.update(range(i, i + n))
    coverage = len(covered) / len(test_tokens)
    return coverage > 0.5, n_shared, round(coverage, 4)


def run_ngram_audit(project_name, train_path, math500_path, n, threshold_mode, output_path):
    print(f"\n{'='*60}")
    print(f"Running n-gram audit: {project_name}  (n={n}, threshold={threshold_mode})")

    with jsonlines.open(math500_path) as r:
        math500 = list(r)
    with jsonlines.open(train_path) as r:
        train_data = list(r)

    print(f"  {len(train_data)} train items × {len(math500)} MATH-500 items = {len(train_data)*len(math500):,} pairs")

    # Precompute MATH-500 n-grams to avoid redundant work in inner loop
    print("  Precomputing MATH-500 n-grams...")
    math500_ngrams = []
    for item in math500:
        text = item.get("problem", "")
        math500_ngrams.append((item, get_ngrams(text, n), text))

    hits = []
    for train_item in tqdm(train_data, desc=f"{project_name} n-gram"):
        train_text = train_item.get("problem", "")
        if not train_text.strip():
            continue
        train_ng = get_ngrams(train_text, n)

        for math_item, m_ng, test_text in math500_ngrams:
            if not test_text.strip():
                continue
            shared = train_ng & m_ng
            if not shared:
                continue

            # Compute proper metrics for the match
            if threshold_mode == "any":
                is_contaminated, n_shared, coverage = True, len(shared), 0.0
            else:
                test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
                covered = set()
                for ngram in shared:
                    for i in range(len(test_tokens) - n + 1):
                        if tuple(test_tokens[i:i+n]) == ngram:
                            covered.update(range(i, i + n))
                coverage = len(covered) / max(len(test_tokens), 1)
                is_contaminated = coverage > 0.5
                n_shared = len(shared)

            if is_contaminated:
                hits.append({
                    "project": project_name,
                    "contamination_type": "c_lex",
                    "math500_id": math_item["math500_id"],
                    "math500_problem": test_text,
                    "math500_answer": math_item.get("answer", ""),
                    "math500_subject": math_item.get("subject", ""),
                    "math500_level": math_item.get("level", -1),
                    "train_id": train_item["train_id"],
                    "train_problem": train_text,
                    "n_shared_ngrams": n_shared,
                    "token_coverage": round(coverage, 4),
                    "ngram_n": n,
                    "threshold_mode": threshold_mode,
                    "filter_should_have_caught": True,
                })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as w:
        w.write_all(hits)

    unique_math500 = len({h["math500_id"] for h in hits})
    print(f"  Found {len(hits)} hits  ({unique_math500} unique MATH-500 items)")
    print(f"  Saved to {output_path}")
    return hits


if __name__ == "__main__":
    s1_hits = run_ngram_audit(
        project_name="s1",
        train_path="data/s1k.jsonl",
        math500_path="data/math500.jsonl",
        n=8,
        threshold_mode="any",
        output_path="results/ngram_hits/s1_ngram_hits.jsonl",
    )

    tulu_hits = run_ngram_audit(
        project_name="tulu",
        train_path="data/tulu_math.jsonl",
        math500_path="data/math500.jsonl",
        n=8,
        threshold_mode="percent",
        output_path="results/ngram_hits/tulu_ngram_hits.jsonl",
    )

    ot_hits = run_ngram_audit(
        project_name="openthoughts",
        train_path="data/openthoughts.jsonl",
        math500_path="data/math500.jsonl",
        n=13,
        threshold_mode="any",
        output_path="results/ngram_hits/openthoughts_ngram_hits.jsonl",
    )

    print("\n=== N-GRAM AUDIT COMPLETE ===")
    print(f"s1:           {len(s1_hits)} hits")
    print(f"Tülu 3:       {len(tulu_hits)} hits")
    print(f"OpenThoughts: {len(ot_hits)} hits")
