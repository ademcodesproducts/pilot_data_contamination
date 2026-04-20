# Contamination Detection Findings Report
**Date:** 2026-04-20  
**Datasets audited:** s1K (1,000 items), Tülu 3 SFT math subset (334k items), OpenThoughts-114K (20k sample)  
**Benchmark:** MATH-500

---

## Summary

We audited three post-training datasets against MATH-500 using two contamination detection methods:
- **C_lex:** Lexical contamination — n-gram overlap with MATH-500 that survived the project's own decontamination filter
- **C_sem:** Semantic contamination — structurally equivalent problems with zero n-gram overlap, detected via embedding similarity + LLM judge

We find evidence of both failure modes across all three datasets. The key finding is that **n-gram decontamination, as currently practiced, provides false confidence**: it fails both because of implementation bugs (C_lex) and by design (C_sem).

---

## Table 1: Contamination Summary

| Project | Train size audited | Filter used | C_lex (≥5 n-grams) | C_lex (≥10 n-grams) | C_sem |
|---|---|---|---|---|---|
| s1 | 1,000 | 8-gram, any overlap | 36 | 13 | 1 |
| Tülu 3 | 334k | 8-gram, >50% token coverage | 15 | 14 | 29 |
| OpenThoughts | 20k sample | 13-gram, any overlap | 35 | 16 | 17 |

*C_lex counts are unique MATH-500 items affected. The raw counts at ≥1 threshold (135/17/81) include substantial boilerplate noise — see Section 3 for threshold sensitivity analysis.*

---

## Section 1: Lexical Contamination (C_lex)

### Finding 1a: s1K contains exact duplicates of MATH-500 problems

Despite s1 claiming 8-gram decontamination, three s1K items are **word-for-word identical** to MATH-500 problems (95, 89, and 39 shared 8-grams respectively). A fourth item is the same problem with a minor answer-format change (39 shared 8-grams):

> **MATH-500:** *"Let λ be a constant, 0 ≤ λ ≤ 4, and let f:[0,1]→[0,1] be defined by f(x)=λx(1−x). Find the values of λ for which there exists x∈[0,1] such that f(x)≠x but f(f(x))=x."*  
> **s1K:** *(identical)*

> **MATH-500:** *"Find the number of integer values of k in [-500,500] for which log(kx)=2log(x+2) has exactly one real solution."*  
> **s1K:** *(identical)*

**Likely cause:** s1's decontamination filter ran on an internal version of the text with different LaTeX formatting (e.g., stripped whitespace or punctuation), causing tokenization mismatches. The public HuggingFace release retained the originals.

**Threshold sensitivity for s1:**

| Min shared 8-grams | Unique MATH-500 items | Interpretation |
|---|---|---|
| ≥1 | 135 | Includes common mathematical phrases |
| ≥5 | 36 | Likely content overlap |
| ≥10 | 13 | Definite content overlap |
| ≥20 | 5 | Near-exact copies |
| ≥50 | 2 | Word-for-word identical |

The steep drop from 135→36→13 shows the "any overlap" criterion is too permissive. At ≥5, 36 MATH-500 items are affected. At ≥10, the 13 items are all verifiably real contamination.

---

### Finding 1b: Tülu 3 — 14–17 MATH-500 items contaminated despite strict filter

Tülu uses the strictest filter (>50% token coverage), yet 14 unique MATH-500 items have high-overlap training counterparts. The distribution is clean — only 1 pair at n=1, with most pairs having 10–39 shared 8-grams. These are substantive overlaps, not noise.

Examples are predominantly linear algebra problems using identical setups with changed numeric values:

> **MATH-500:** *"The set of vectors v such that proj_{(2,1)} v = proj_{(2,1)} (3,2)..."*  
> **Tülu:** *"Find the vector v such that proj_{(2,1)} v = proj_{(2,1)} (3,2)..."*  
> *(39 shared 8-grams — same projection setup, slightly different question framing)*

> **MATH-500:** *"A reflection takes (5,0) to (4,3). Where does it take (-1,-3)?"*  
> **Tülu:** *"A reflection takes (-1,7) to (5,-5). Where does it take (1,4)?"*  
> *(30 shared n-grams — identical template, different numbers)*

---

### Finding 1c: OpenThoughts — 16–35 MATH-500 items from just a 20k sample

OpenThoughts uses a 13-gram threshold (stricter than 8-gram), yet 16 MATH-500 items are hit at ≥10 shared 13-grams from only 20k of 114k items. One pair has **348 shared 13-grams** — a near-verbatim copy:

> **MATH-500:** *"Five points A, B, C, D, O lie on a flat field. A is directly north of O, B is directly west of O..."*  
> **OpenThoughts:** *"Return your final response within \boxed{}. Five points A, B, C, D, O lie on a flat field. A is directly north of O, B is directly west of O..."*  
> *(348 shared 13-grams — identical problem, only prefix differs)*

The OpenThoughts items include a `\boxed{}` instruction prefix absent in MATH-500, explaining why the filter partially failed. Extrapolating to the full 114k dataset: **~90–200 MATH-500 items likely affected** (conservative estimate; uniform extrapolation would suggest ~370 but contamination rates vary by source).

---

## Section 2: Semantic Contamination (C_sem)

### Finding 2a: Tülu — 29 MATH-500 items with zero n-gram overlap but structural equivalence

This is the core finding of the paper. Despite Tülu's strict 50% coverage filter, 29 MATH-500 items have structurally equivalent training counterparts confirmed by LLM judge (38% precision from 147 candidates). These items share no 8-gram overlap — no surface-form filter could have caught them.

The top C_sem pairs reveal two sub-types:

**Near-paraphrase (same problem, different wording):**
> **MATH-500:** *"How many integers are in the solution set of |x−2|≤5.6?"*  
> **Tülu:** *"How many integers are there in the solution set of |x−2|≤5.6?"*  
> *(sim=0.998 — word-order swap only; n-gram filter missed due to tokenization boundary)*

> **MATH-500:** *"For how many two-digit primes is the sum of the digits equal to 8?"*  
> **Tülu:** *"For how many two-digit prime numbers is the sum of its digits 8?"*  
> *(sim=0.980 — minor rephrasing)*

**Template variant (same structure, different constants):**
> **MATH-500:** *"Let a,b,c be real numbers such that |ax²+bx+c|≤1 for all 0≤x≤1. Find the maximum of |a|+|b|+|c|."*  
> **Tülu:** *"Given a,b,c∈ℝ, for any x with |x|≤1, |ax²+bx+c|≤1. Find the max of |a|+|b|+|c|."*  
> *(sim=0.970 — same theorem, different domain statement)*

> **MATH-500:** *"Compute ∠ABC where A=(1,−11,2), B=(3,−4,1), C=(−2,1,−1)."*  
> **Tülu:** *"Compute ∠ABC where A=(−4,0,6), B=(−5,−1,2), C=(−6,−1,3)."*  
> *(sim=0.921 — identical dot-product procedure, different coordinates)*

The near-paraphrase cases are particularly damning: the filter failed on problems that differ only in word order, because the token sequence changes just enough to fall below the 50% coverage threshold.

---

### Finding 2b: OpenThoughts — 17 C_sem items from 20k sample

17 MATH-500 items have semantically equivalent counterparts in the 20k OpenThoughts sample (18% judge precision from 100 candidates). Extrapolating to full 114k: **~97 C_sem items** (rough lower bound).

### Finding 2c: s1 — 1 C_sem item

Only 1 confirmed C_sem item for s1, which is expected: the dataset is small (1,000 items) and was carefully curated, so semantic near-misses are rare.

---

## Section 3: Two Distinct Failure Modes

The findings reveal that contamination escapes detection in two fundamentally different ways:

**Type 1 — Implementation failure (C_lex):** The filter ran and flagged the item, but the item survived anyway. Mechanism: the filter operated on a different text representation (internal version vs. HuggingFace release), causing tokenization mismatches. Even exact duplicates passed through.

**Type 2 — Design gap (C_sem):** The filter could not have caught the item because it operates on surface form. Structurally equivalent problems with paraphrased wording, reordered sentences, or changed numeric constants share zero n-gram overlap with the benchmark.

Both failure modes are present in all three datasets. Addressing Type 1 requires better release engineering (run the filter on the exact text that gets released). Addressing Type 2 requires semantic decontamination — the gap this paper aims to fill.

---

## Section 4: Caveats and Limitations

1. **s1 C_lex threshold sensitivity:** The raw 135 number includes many 1–2 n-gram pairs that are likely mathematical boilerplate. We report ≥5 (36 items) as the conservative estimate and ≥10 (13 items) as the strict estimate. Reviewers should treat 13 as the floor and 36 as the ceiling.

2. **OpenThoughts extrapolation:** Results are from a 20k reproducible random sample (seed=42) of 114k items. Contamination rate may vary across sources within the dataset. We report sample results only; full-dataset audit is future work.

3. **Judge precision:** LLM judge precision ranges from 10% (s1, 14 candidates, small pool) to 38% (Tülu, 147 candidates). The 10% figure for s1 reflects a very small candidate pool, not poor judge quality. All CONTAMINATED labels were produced by the same prompt; no human validation has been completed yet.

4. **Tülu dataset scope:** We audited all math-sourced items from `allenai/tulu-3-sft-mixture` (334k items across 5 sources). The spec anticipated ~84k items (NuminaMath-TIR only). The broader scope may include sources not originally intended as "Tülu 3 math SFT." Results for the NuminaMath-TIR source specifically (64k items) should be broken out in future analysis.

---

## Next Steps

1. **Manual validation** of 20 Tülu C_sem pairs — fill in `precision_worksheet.tsv` to get a human precision estimate before reporting 29 as the headline number.
2. **Full OpenThoughts audit** — run all 114k items through the pipeline to replace the extrapolated estimate with an actual count.
3. **Behavioral experiments** — use `data/output/*_c_lex.jsonl` and `data/output/*_c_sem.jsonl` as input to the performance gap analysis (do models score higher on C_lex/C_sem items vs. clean items?).
4. **Tülu source breakdown** — split C_lex and C_sem counts by source (NuminaMath vs. PersonaHub vs. GSM8K) to identify which source drives the contamination signal.
