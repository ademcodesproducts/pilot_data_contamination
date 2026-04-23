"""
07_tulu_source_breakdown.py

Splits Tülu C_lex and C_sem contamination counts by training data source.
Joins contaminated items back to tulu_math.jsonl via train_id.

Output:
  results/tulu_source_breakdown.csv
  results/tulu_source_breakdown.txt
"""

import jsonlines
import pandas as pd
from collections import defaultdict
from pathlib import Path

SOURCE_LABELS = {
    "ai2-adapt-dev/numinamath_tir_math_decontaminated":        "NuminaMath-TIR",
    "ai2-adapt-dev/personahub_math_v5_regen_149960":           "PersonaHub-Math",
    "allenai/tulu-3-sft-personas-math-grade":                   "PersonaHub-Grade",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k":           "GSM8K",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "PersonaHub-Algebra",
}

MATH500_SIZE = 500


def run():
    # ── Build train_id → source lookup ───────────────────────────────────────
    print("Building train_id -> source lookup...")
    id_to_source = {}
    source_sizes = defaultdict(int)
    with jsonlines.open("data/tulu_math.jsonl") as r:
        for item in r:
            id_to_source[item["train_id"]] = item.get("source", "unknown")
            source_sizes[item.get("source", "unknown")] += 1
    print(f"  {len(id_to_source):,} items indexed")

    # ── Load contaminated items ───────────────────────────────────────────────
    lex_by_source  = defaultdict(set)   # source → set of math500_ids
    sem_by_source  = defaultdict(set)

    with jsonlines.open("data/output/tulu_c_lex.jsonl") as r:
        for item in r:
            src = id_to_source.get(item["train_id"], "unknown")
            lex_by_source[src].add(item["math500_id"])

    with jsonlines.open("data/output/tulu_c_sem.jsonl") as r:
        for item in r:
            src = id_to_source.get(item["train_id"], "unknown")
            sem_by_source[src].add(item["math500_id"])

    # ── Build results table ───────────────────────────────────────────────────
    all_sources = set(source_sizes.keys())
    rows = []
    for src in sorted(all_sources, key=lambda s: -source_sizes[s]):
        label = SOURCE_LABELS.get(src, src)
        size  = source_sizes[src]
        n_lex = len(lex_by_source[src])
        n_sem = len(sem_by_source[src])
        n_tot = len(lex_by_source[src] | sem_by_source[src])
        rows.append({
            "source":         src,
            "label":          label,
            "train_size":     size,
            "n_lex":          n_lex,
            "n_sem":          n_sem,
            "n_total":        n_tot,
            "pct_lex":        round(n_lex / MATH500_SIZE * 100, 1),
            "pct_sem":        round(n_sem / MATH500_SIZE * 100, 1),
            "pct_total":      round(n_tot / MATH500_SIZE * 100, 1),
        })

    df = pd.DataFrame(rows)

    # ── Totals row ────────────────────────────────────────────────────────────
    all_lex = set().union(*lex_by_source.values())
    all_sem = set().union(*sem_by_source.values())
    totals = {
        "source": "TOTAL", "label": "ALL SOURCES",
        "train_size": sum(source_sizes.values()),
        "n_lex": len(all_lex), "n_sem": len(all_sem),
        "n_total": len(all_lex | all_sem),
        "pct_lex":   round(len(all_lex) / MATH500_SIZE * 100, 1),
        "pct_sem":   round(len(all_sem) / MATH500_SIZE * 100, 1),
        "pct_total": round(len(all_lex | all_sem) / MATH500_SIZE * 100, 1),
    }

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/tulu_source_breakdown.csv", index=False)

    # ── Print table ───────────────────────────────────────────────────────────
    lines = []
    lines.append("Tulu Contamination by Source")
    lines.append("=" * 82)
    lines.append(f"{'Source':<22} {'Size':>8} {'C_lex':>6} {'C_lex%':>7} {'C_sem':>6} {'C_sem%':>7} {'Total':>6} {'Tot%':>6}")
    lines.append("-" * 82)
    for r in rows:
        lines.append(
            f"{r['label']:<22} {r['train_size']:>8,} {r['n_lex']:>6} {r['pct_lex']:>6}% "
            f"{r['n_sem']:>6} {r['pct_sem']:>6}% {r['n_total']:>6} {r['pct_total']:>5}%"
        )
    lines.append("-" * 82)
    lines.append(
        f"{'ALL SOURCES':<22} {totals['train_size']:>8,} {totals['n_lex']:>6} {totals['pct_lex']:>6}% "
        f"{totals['n_sem']:>6} {totals['pct_sem']:>6}% {totals['n_total']:>6} {totals['pct_total']:>5}%"
    )
    lines.append("")
    lines.append("Note: C_lex and C_sem count unique MATH-500 items affected (out of 500 total).")
    lines.append("      A single source may affect the same MATH-500 item multiple times;")
    lines.append("      source totals can exceed the dataset total due to de-duplication across sources.")

    report = "\n".join(lines)
    print(report)
    Path("results/tulu_source_breakdown.txt").write_text(report, encoding="utf-8")
    print("\nSaved: results/tulu_source_breakdown.csv")
    print("Saved: results/tulu_source_breakdown.txt")
    return df


if __name__ == "__main__":
    run()
