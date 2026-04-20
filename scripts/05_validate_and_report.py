"""
05_validate_and_report.py

1. Assembles final confirmed C_lex, C_sem, and Clean sets per project
2. Runs optional manual spot-check (interactive CLI)
3. Produces final summary CSV and Table 1 for the paper

Output:
  data/output/{project}_c_lex.jsonl
  data/output/{project}_c_sem.jsonl
  data/output/clean.jsonl
  results/final_summary.csv
"""

import jsonlines
import pandas as pd
import random
from pathlib import Path


def assemble_final_sets():
    all_results = []

    for project in ["s1", "tulu", "openthoughts"]:

        # ── C_lex ──────────────────────────────────────────────────────────
        c_lex_path = f"results/ngram_hits/{project}_ngram_hits.jsonl"
        try:
            with jsonlines.open(c_lex_path) as r:
                c_lex_items = list(r)

            # Deduplicate by math500_id, keep highest-overlap representative
            seen = set()
            c_lex_unique = []
            for item in sorted(c_lex_items, key=lambda x: -x["n_shared_ngrams"]):
                if item["math500_id"] not in seen:
                    c_lex_unique.append(item)
                    seen.add(item["math500_id"])

            out_path = f"data/output/{project}_c_lex.jsonl"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(out_path, mode="w") as w:
                w.write_all(c_lex_unique)

            all_results.append({
                "project": project,
                "contamination_type": "C_lex",
                "n_unique_math500": len(seen),
                "n_total_pairs": len(c_lex_items),
                "judge_precision": "N/A (lexical)",
                "notes": f"n-gram n={c_lex_unique[0]['ngram_n'] if c_lex_unique else '?'}",
            })
            print(f"{project} C_lex: {len(c_lex_unique)} unique MATH-500 items")

        except FileNotFoundError:
            print(f"  Warning: no n-gram hits for {project}")

        # ── C_sem ──────────────────────────────────────────────────────────
        judge_path = f"results/judge_results/{project}_judge_results.jsonl"
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

            all_results.append({
                "project": project,
                "contamination_type": "C_sem",
                "n_unique_math500": len(seen),
                "n_total_pairs": len(c_sem_items),
                "judge_precision": f"{precision:.0%}",
                "notes": f"{total_judged} pairs judged",
            })
            print(f"{project} C_sem: {len(c_sem_unique)} unique MATH-500 items  (judge precision: {precision:.0%})")

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


def print_table1():
    print("\n=== TABLE 1 FOR PAPER ===")
    print(f"{'Project':<14} {'Filter':<30} {'Train size':>12} {'C_lex':>7} {'C_sem':>7}")
    print("-" * 75)

    rows = [
        ("s1",           "8-gram any overlap",        "1,000",  "s1_c_lex",           "s1_c_sem"),
        ("Tülu 3",       "8-gram 50% token coverage", "~84k",   "tulu_c_lex",         "tulu_c_sem"),
        ("OpenThoughts", "Fuzzy + 13-gram",            "20k",    "openthoughts_c_lex", "openthoughts_c_sem"),
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
        print(f"{project:<14} {filt:<30} {size:>12} {n_lex:>7} {n_sem:>7}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-spotcheck", action="store_true",
                        help="Skip interactive manual spot-check")
    args = parser.parse_args()

    assemble_final_sets()

    if not args.no_spotcheck:
        manual_spot_check(n_per_set=5)

    print_table1()
