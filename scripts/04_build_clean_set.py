"""
04_build_clean_set.py

Identifies MATH-500 items with lowest similarity to any training item
across s1, Tülu 3, and OpenThoughts.

Strategy:
1. For each MATH-500 item, find its maximum similarity score to any
   training item across all three projects (from embedding results)
2. Sort ascending — lowest max_sim = most "clean"
3. Exclude any item that appeared in C_lex or C_sem
4. Select bottom N items

Output: data/output/clean.jsonl
"""

import jsonlines
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_contaminated_ids():
    contaminated = set()

    for project in ["s1", "tulu", "openthoughts"]:
        path = f"results/ngram_hits/{project}_ngram_hits.jsonl"
        try:
            with jsonlines.open(path) as r:
                for item in r:
                    contaminated.add(item["math500_id"])
        except FileNotFoundError:
            pass

    for project in ["s1", "tulu", "openthoughts"]:
        path = f"results/judge_results/{project}_judge_results.jsonl"
        try:
            with jsonlines.open(path) as r:
                for item in r:
                    if item.get("classification") == "CONTAMINATED":
                        contaminated.add(item["math500_id"])
        except FileNotFoundError:
            pass

    return contaminated


def get_max_similarities():
    max_sims = defaultdict(float)
    for project in ["s1", "tulu", "openthoughts"]:
        path = f"results/embedding_candidates/{project}_candidates.jsonl"
        try:
            with jsonlines.open(path) as r:
                for item in r:
                    mid = item["math500_id"]
                    max_sims[mid] = max(max_sims[mid], item["similarity_score"])
        except FileNotFoundError:
            pass
    return dict(max_sims)


def build_clean_set(target_n=100):
    contaminated_ids = load_contaminated_ids()
    max_sims = get_max_similarities()

    print(f"Contaminated MATH-500 items (across all projects): {len(contaminated_ids)}")

    with jsonlines.open("data/math500.jsonl") as r:
        math500 = list(r)

    clean_candidates = []
    for item in math500:
        mid = item["math500_id"]
        if mid in contaminated_ids:
            continue
        clean_candidates.append({**item, "max_similarity": max_sims.get(mid, 0.0)})

    print(f"Clean candidates (not contaminated): {len(clean_candidates)}")

    clean_candidates.sort(key=lambda x: x["max_similarity"])
    selected = clean_candidates[:target_n]

    if not selected:
        print("WARNING: no clean candidates found. Check that embedding results exist.")
        return []

    sims = [c["max_similarity"] for c in selected]
    print(f"\nSelected {len(selected)} clean items")
    print(f"Max similarity in clean set:  {max(sims):.3f}")
    print(f"Mean similarity in clean set: {np.mean(sims):.3f}")

    from collections import Counter
    subjects = Counter(c["subject"] for c in selected)
    print("\nClean set subject distribution:")
    for subj, count in subjects.most_common():
        print(f"  {subj}: {count}")

    Path("data/output").mkdir(parents=True, exist_ok=True)
    with jsonlines.open("data/output/clean.jsonl", mode="w") as w:
        w.write_all(selected)

    print(f"\nSaved to data/output/clean.jsonl")
    return selected


if __name__ == "__main__":
    build_clean_set(target_n=100)
