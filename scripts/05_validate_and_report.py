"""
05_validate_and_report.py

1. Assembles final confirmed C_lex, C_sem, and Clean sets per project
2. Runs optional manual spot-check (interactive CLI)
3. Produces final summary CSV and Table 1 for the paper

C_lex threshold: items with n_shared_ngrams >= MIN_NGRAMS (default 5).
  ≥1  → raw hits including boilerplate noise
  ≥5  → conservative estimate (default, written to output files)
  ≥10 → strict estimate (also reported in summary for sensitivity analysis)

Output:
  data/output/{project}_c_lex.jsonl   (filtered at --min-ngrams, default 5)
  data/output/{project}_c_sem.jsonl
  data/output/clean.jsonl
  results/final_summary.csv
"""

import jsonlines
import pandas as pd
import random
from pathlib import Path

# Dataset sizes (used for contamination rate calculation)
DATASET_SIZES = {
    "s1":           1_000,
    "tulu":         334_000,
    "openthoughts": 114_000,
}


def _filter_and_dedup_c_lex(items, min_ngrams):
    """Filter by shared n-gram count, deduplicate by math500_id (keep max overlap)."""
    filtered = [item for item in items if item["n_shared_ngrams"] >= min_ngrams]
    seen = set()
    unique = []
    for item in sorted(filtered, key=lambda x: -x["n_shared_ngrams"]):
        if item["math500_id"] not in seen:
            unique.append(item)
            seen.add(item["math500_id"])
    return unique


def _ngram_hits_path(project):
    """Return the best available ngram hits file for a project.
    Prefers the full-dataset variant for openthoughts."""
    full = Path(f"results/ngram_hits/{project}_full_ngram_hits.jsonl")
    base = Path(f"results/ngram_hits/{project}_ngram_hits.jsonl")
    return full if full.exists() else base


def _judge_results_path(project):
    full = Path(f"results/judge_results/{project}_full_judge_results.jsonl")
    base = Path(f"results/judge_results/{project}_judge_results.jsonl")
    return full if full.exists() else base


def assemble_final_sets(min_ngrams=5):
    all_results = []

    for project in ["s1", "tulu", "openthoughts"]:
        train_size = DATASET_SIZES[project]

        # ── C_lex ──────────────────────────────────────────────────────────
        c_lex_path = _ngram_hits_path(project)
        try:
            with jsonlines.open(c_lex_path) as r:
                c_lex_items = list(r)

            # Primary threshold (written to output files)
            c_lex_unique = _filter_and_dedup_c_lex(c_lex_items, min_ngrams)
            # Strict threshold (≥10) for sensitivity column
            c_lex_strict = _filter_and_dedup_c_lex(c_lex_items, 10)
            # Raw count (≥1) for reference
            c_lex_raw = _filter_and_dedup_c_lex(c_lex_items, 1)

            out_path = f"data/output/{project}_c_lex.jsonl"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(out_path, mode="w") as w:
                w.write_all(c_lex_unique)

            n_lex = len(c_lex_unique)
            ngram_n = c_lex_unique[0]["ngram_n"] if c_lex_unique else (c_lex_raw[0]["ngram_n"] if c_lex_raw else "?")
            all_results.append({
                "project":          project,
                "train_size":       train_size,
                "contamination_type": "C_lex",
                "n_unique_math500": n_lex,
                "n_lex_strict_10":  len(c_lex_strict),
                "n_lex_raw_1":      len(c_lex_raw),
                "rate_pct":         round(n_lex / 500 * 100, 2),  # out of 500 MATH-500 items
                "judge_precision":  "N/A (lexical)",
                "threshold_ngrams": min_ngrams,
                "notes": f"n-gram n={ngram_n}",
            })
            print(f"{project} C_lex: {n_lex} items (>={min_ngrams}), {len(c_lex_strict)} (>=10), {len(c_lex_raw)} (>=1 raw)")

        except FileNotFoundError:
            print(f"  Warning: no n-gram hits for {project}")

        # ── C_sem ──────────────────────────────────────────────────────────
        judge_path = _judge_results_path(project)
        try:
            with jsonlines.open(judge_path) as r:
                judge_results = list(r)

            c_sem_items = [r for r in judge_results if r.get("classification") == "CONTAMINATED"]

            seen = set()
            c_sem_unique = []
            for item in sorted(c_sem_items, key=lambda x: -x.get("similarity_score", 0)):
                if item["math500_id"] not in seen:
                    c_sem_unique.append(item)
                    seen.add(item["math500_id"])

            out_path = f"data/output/{project}_c_sem.jsonl"
            with jsonlines.open(out_path, mode="w") as w:
                w.write_all(c_sem_unique)

            total_judged = len([r for r in judge_results if r.get("classification") != "ERROR"])
            precision = len(c_sem_items) / max(total_judged, 1)
            n_sem = len(c_sem_unique)

            all_results.append({
                "project":          project,
                "train_size":       train_size,
                "contamination_type": "C_sem",
                "n_unique_math500": n_sem,
                "n_lex_strict_10":  "",
                "n_lex_raw_1":      "",
                "rate_pct":         round(n_sem / 500 * 100, 2),
                "judge_precision":  f"{precision:.0%}",
                "threshold_ngrams": "",
                "notes": f"{total_judged} pairs judged",
            })
            print(f"{project} C_sem: {n_sem} unique MATH-500 items  (judge precision: {precision:.0%})")

        except FileNotFoundError:
            print(f"  Warning: no judge results for {project}")

    df = pd.DataFrame(all_results)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/final_summary.csv", index=False)
    print(f"\nFinal summary saved to results/final_summary.csv")
    print(df.to_string(index=False))
    return df


def manual_spot_check(n_per_set=5):
    print("\n=== MANUAL SPOT CHECK ===")
    print("Press Enter to continue through each item. Enter 'y'=correct, 'n'=wrong, 's'=skip.\n")

    for project in ["s1", "tulu", "openthoughts"]:
        for ctype in ["c_sem"]:
            path = f"data/output/{project}_{ctype}.jsonl"
            try:
                with jsonlines.open(path) as r:
                    items = list(r)
                if not items:
                    continue

                sample = random.sample(items, min(n_per_set, len(items)))
                print(f"\n{'='*60}")
                print(f"Spot check: {project} {ctype.upper()}  ({len(items)} total, showing {len(sample)})")

                correct = 0
                for i, item in enumerate(sample):
                    print(f"\n  [{i+1}/{len(sample)}]  sim={item.get('similarity_score','N/A'):.3f}")
                    print(f"  MATH-500:  {item['math500_problem'][:200]}...")
                    print(f"  Training:  {item['train_problem'][:200]}...")
                    print(f"  Judge: {item.get('reasoning','N/A')}")
                    v = input("  Agree CONTAMINATED? [y/n/s]: ").strip().lower()
                    if v == "y":
                        correct += 1
                    elif v == "n":
                        print("  Noted as false positive.")

                print(f"\n  Human precision: {correct}/{len(sample)} = {correct/len(sample):.0%}")
            except FileNotFoundError:
                pass


def print_table1(min_ngrams=5):
    print(f"\n=== TABLE 1 FOR PAPER  (C_lex threshold: >={min_ngrams} shared n-grams) ===")
    print(f"{'Project':<14} {'Filter':<30} {'Train size':>10} {'C_lex':>7} {'C_sem':>7} {'Total':>7} {'Rate%':>7}")
    print("-" * 88)

    ot_size = "114,000" if Path("results/ngram_hits/openthoughts_full_ngram_hits.jsonl").exists() else "20,000*"
    rows = [
        ("s1",           "8-gram, any overlap",        "1,000",   "s1_c_lex",           "s1_c_sem"),
        ("Tulu 3",       "8-gram, >50% token cov.",    "334,000", "tulu_c_lex",         "tulu_c_sem"),
        ("OpenThoughts", "13-gram, any overlap",        ot_size,   "openthoughts_c_lex", "openthoughts_c_sem"),
    ]

    for project, filt, size, lex_file, sem_file in rows:
        n_lex, n_sem = 0, 0
        try:
            with jsonlines.open(f"data/output/{lex_file}.jsonl") as r:
                n_lex = len(list(r))
        except FileNotFoundError:
            pass
        try:
            with jsonlines.open(f"data/output/{sem_file}.jsonl") as r:
                n_sem = len(list(r))
        except FileNotFoundError:
            pass
        # Unique MATH-500 items affected (union, not sum — items may appear in both)
        total = n_lex + n_sem
        rate = f"{total / 500 * 100:.1f}%"
        print(f"{project:<14} {filt:<30} {size:>10} {n_lex:>7} {n_sem:>7} {total:>7} {rate:>7}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-spotcheck", action="store_true",
                        help="Skip interactive manual spot-check")
    parser.add_argument("--min-ngrams", type=int, default=5,
                        help="Minimum shared n-gram count to include in C_lex (default: 5)")
    args = parser.parse_args()

    assemble_final_sets(min_ngrams=args.min_ngrams)

    if not args.no_spotcheck:
        manual_spot_check(n_per_set=5)

    print_table1(min_ngrams=args.min_ngrams)
