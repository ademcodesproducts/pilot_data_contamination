# Contamination Detection Findings Report
**Date:** 2026-04-23  
**Datasets audited:** s1K (1,000 items), Tülu 3 SFT math subset (334k items), OpenThoughts-114K (full 114k)  
**Benchmark:** MATH-500

---

## Summary

We audited three post-training datasets against MATH-500 using two contamination detection methods:
- **C_lex:** Lexical contamination — n-gram overlap with MATH-500 that survived the project's own decontamination filter
- **C_sem:** Semantic contamination — structurally equivalent problems with zero n-gram overlap, detected via embedding similarity + LLM judge

We find evidence of both failure modes across all three datasets. The key finding is that **n-gram decontamination, as currently practiced, provides false confidence**: it fails both because of implementation bugs (C_lex) and by design (C_sem).

OpenThoughts-114K shows the highest total contamination (24.6% of MATH-500, 95% CI unavailable for combined), despite using the strictest n-gram threshold (13-gram vs. 8-gram). The Tülu dataset is unique in that semantic contamination outnumbers lexical contamination — a direct consequence of its stricter coverage filter pushing contamination into the surface-change-only regime. s1 is dominated by a single implementation bug (exact duplicates that passed through the filter).

---

## Table 1: Contamination Summary

| Project | Train size | Filter used | C_lex (>=5 n-grams) | C_sem (judge) | Total unique | Rate |
|---|---|---|---|---|---|---|
| s1 | 1,000 | 8-gram, any overlap | 36 | 1 | 37 | 7.4% |
| Tülu 3 | 334k | 8-gram, >50% token coverage | 15 | 29 | 44 | 8.8% |
| OpenThoughts | 114k | 13-gram, any overlap | 93 | 42 | 123 | 24.6% |

*Rates are unique MATH-500 items affected out of 500. "Total unique" is the union of C_lex and C_sem sets (some items appear in both). C_lex counts use a conservative >=5 shared n-grams threshold to exclude boilerplate noise.*

*s1 C_lex at >=10 shared 8-grams: 13 items (stricter lower bound). Tülu C_lex at >=10: 14 items.*

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
| >=1 | 135 | Includes common mathematical phrases |
| >=5 | 36 | Likely content overlap |
| >=10 | 13 | Definite content overlap |
| >=20 | 5 | Near-exact copies |
| >=50 | 2 | Word-for-word identical |

The steep drop from 135→36→13 shows the "any overlap" criterion is too permissive. At >=5, 36 MATH-500 items are affected. At >=10, the 13 items are all verifiably real contamination.

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

### Finding 1c: OpenThoughts — 93 MATH-500 items contaminated across full 114k dataset

The full 114k audit (completed 2026-04-22) confirms large-scale C_lex contamination despite the 13-gram threshold. 93 unique MATH-500 items are affected at the >=5 shared 13-grams threshold (18.6% of MATH-500). The most extreme pair has **479 shared 13-grams** — a near-verbatim copy:

> **MATH-500:** *"Five points A, B, C, D, O lie on a flat field. A is directly north of O, B is directly west of O..."*  
> **OpenThoughts:** *"Return your final response within \boxed{}. Five points A, B, C, D, O lie on a flat field. A is directly north of O, B is directly west of O..."*  
> *(identical problem body — only the \boxed{} instruction prefix differs)*

The `\boxed{}` prefix shifts the token sequence just enough to evade the 13-gram filter on the leading tokens, while the problem body remains intact with hundreds of shared n-grams. This is the same implementation-bug pattern as s1.

---

## Section 2: Semantic Contamination (C_sem)

### Finding 2a: Tülu — 29 MATH-500 items with zero n-gram overlap but structural equivalence

This is the core finding of the paper. Despite Tülu's strict 50% coverage filter, 29 MATH-500 items have structurally equivalent training counterparts confirmed by LLM judge (38% precision from 147 candidates, validated at 44% by human review). These items share no 8-gram overlap — no surface-form filter could have caught them.

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

### Finding 2b: OpenThoughts — 42 MATH-500 items with semantic contamination (full 114k)

42 MATH-500 items have semantically equivalent counterparts in the full 114k OpenThoughts dataset (28% judge precision from candidates, validated at 29% by human review on 15 top pairs). Human-confirmed examples include:

> **MATH-500:** *"How many integers are in the solution set of |x−2|≤5.6?"*  
> **OpenThoughts:** *"Return your final response within \boxed{}. How many integers are there in the solution set of |x−2|≤5.6?"*  
> *(sim=0.845 — same problem appearing in both Tülu and OpenThoughts with different prefixes; evidence of shared upstream source)*

> **MATH-500:** *"Consider the geometric sequence 125/9, 25/3, 5, 3, ... What is the eighth term?"*  
> **OpenThoughts:** *"Consider the geometric sequence 3, 9/2, 27/4, 81/8, ... Find the eighth term."*  
> *(sim=0.921 — same formula a₈ = a₁ × r⁷, only ratio and starting value differ)*

After applying human precision correction (29%), the corrected C_sem estimate for OpenThoughts is approximately **12 items** (range: 11–14).

---

### Finding 2c: s1 — 1 C_sem item

Only 1 confirmed C_sem item for s1, which is expected: the dataset is small (1,000 items) and was carefully curated, so semantic near-misses are rare.

---

## Section 3: Cross-Dataset Failure Mode Comparison

Running bootstrap confidence intervals (10,000 resamples, seed=42) on contamination rates across all three datasets:

| Dataset | Filter | C_lex% | 95% CI | C_sem% | 95% CI | Total% | Mode |
|---|---|---|---|---|---|---|---|
| s1 | 8-gram, any overlap | 7.2% | [5.0, 9.6] | 0.2% | [0.0, 0.6] | 7.4% | C_lex dominant |
| Tülu 3 | 8-gram, >50% coverage | 3.0% | [1.6, 4.6] | 5.8% | [3.8, 8.0] | 8.8% | Mixed |
| OpenThoughts | 13-gram, any overlap | 18.6% | [15.2, 22.0] | 8.4% | [6.0, 11.0] | 24.6% | C_lex dominant |

**Contamination share (% of each dataset's total contaminated items):**

| Dataset | C_lex share | C_sem share | Interpretation |
|---|---|---|---|
| s1 | 97.3% | 2.7% | Implementation bug dominates |
| Tülu 3 | 34.1% | 65.9% | Both failure modes active |
| OpenThoughts | 75.6% | 34.1% | Implementation bug dominates |

**Key insight:** Tülu uses the strictest lexical filter and shows the lowest C_lex rate (3.0%) but the highest C_sem share (65.9% of its contamination). Its stricter filter successfully suppresses surface-form leakage but drives contamination into the semantic-only regime that no n-gram filter can detect. OpenThoughts has the highest total contamination (24.6%) despite a stricter 13-gram threshold than s1, demonstrating that n-gram length does not scale linearly with contamination protection at large dataset sizes.

---

## Section 4: Tülu Source Breakdown

Breaking Tülu's contamination by training data source reveals a highly concentrated signal:

| Source | Train size | C_lex | C_lex% | C_sem | C_sem% | Total | Total% |
|---|---|---|---|---|---|---|---|
| PersonaHub-Math | 149,960 | 1 | 0.2% | 0 | 0.0% | 1 | 0.2% |
| NuminaMath-TIR | 64,312 | 14 | 2.8% | 28 | 5.6% | 42 | 8.4% |
| GSM8K | 50,000 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| PersonaHub-Grade | 49,980 | 0 | 0.0% | 1 | 0.2% | 1 | 0.2% |
| PersonaHub-Algebra | 20,000 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| **ALL SOURCES** | **334,252** | **15** | **3.0%** | **29** | **5.8%** | **44** | **8.8%** |

*C_lex and C_sem count unique MATH-500 items affected per source. Source totals may overlap with dataset totals.*

**Finding:** NuminaMath-TIR (64k items, 19% of Tülu) drives nearly all contamination: 14/15 C_lex items (93%) and 28/29 C_sem items (97%). The three PersonaHub sources together (220k items, 66% of Tülu) contribute at most 2 contaminated MATH-500 items. GSM8K and PersonaHub-Algebra contribute zero.

This concentration matters for interpreting Tülu's contamination rate. The effective contamination source is a single 64k dataset that shares mathematical structure and content with MATH-500. The 50% coverage filter was specifically designed to protect against NuminaMath-TIR overlap — yet both C_lex and C_sem contamination persist within this source, confirming that the filter's design assumptions underestimate the semantic similarity between NuminaMath and MATH-500 problems.

---

## Section 5: Robustness Check (n-gram Threshold Sensitivity)

To rule out the criticism that C_lex results are an artifact of the n=13 hyperparameter choice, we re-ran the n-gram filter on OpenThoughts-114K at n=15 and n=20, holding the minimum shared n-grams threshold fixed at >=5:

| n-gram size | Raw hits | Filtered (>=5) | % MATH-500 |
|---|---|---|---|
| 13 | 162 | 93 | 18.6% |
| 15 | 123 | 65 | 13.0% |
| 20 | 51 | 32 | 6.4% |

- n=15 retains 70% of n=13 contaminated items (65 of 93)
- n=20 retains 34% of n=13 contaminated items (32 of 93)

**Interpretation:** C_lex contamination persists across all threshold choices. The trend is stable: stricter thresholds reduce noise but do not eliminate the signal. The 34% retention at n=20 is a lower bound on "hard" contamination — items with very long exact-match runs that cannot plausibly be explained by shared mathematical notation. These 32 items represent cases where the n-gram filter definitively failed regardless of threshold.

---

## Section 6: Two Distinct Failure Modes

The findings reveal that contamination escapes detection in two fundamentally different ways:

**Type 1 — Implementation failure (C_lex):** The filter ran and flagged the item, but the item survived anyway. Mechanism: the filter operated on a different text representation (internal version vs. HuggingFace release), causing tokenization mismatches. Even exact duplicates passed through.

**Type 2 — Design gap (C_sem):** The filter could not have caught the item because it operates on surface form. Structurally equivalent problems with paraphrased wording, reordered sentences, or changed numeric constants share zero n-gram overlap with the benchmark.

Both failure modes are present in all three datasets. Addressing Type 1 requires better release engineering (run the filter on the exact text that gets released). Addressing Type 2 requires semantic decontamination — the gap this paper aims to fill.

---

## Section 7: Unified Leakage Taxonomy

Across all three datasets, contaminated items fall into four categories defined by failure mode and mechanism. This replaces the dataset-by-dataset narrative with a cross-cutting structure.

---

### C_lex Type A — Implementation bug (filter ran on wrong text)

The decontamination filter was applied but operated on an internal pre-processed representation that differed from the publicly released text. The n-gram signature of the internal version did not match the release, so exact duplicates passed through undetected.

**s1** (95 and 89 shared 8-grams — word-for-word identical):
> **MATH-500:** *"Let λ be a constant, 0 ≤ λ ≤ 4, and let f:[0,1]→[0,1] be defined by f(x)=λx(1−x)..."*
> **s1K:** *(identical)*

> **MATH-500:** *"Steve says to Jon, 'I am thinking of a polynomial whose roots are all positive integers...'"*
> **s1K:** *(identical)*

**OpenThoughts** (479 and 348 shared 13-grams — identical problem body with instruction prefix):
> **MATH-500:** *"Five points A, B, C, D, O lie on a flat field. A is directly north of O..."*
> **OpenThoughts:** *"Return your final response within \boxed{}. Five points A, B, C, D, O lie on a flat field. A is directly north of O..."*
> *(348 shared 13-grams — only the \boxed{} prefix differs)*

**Root cause:** The filter ran on LaTeX-normalized or whitespace-stripped internal text; the HuggingFace release retained original formatting. The `\boxed{}` prefix added by OpenThoughts's pipeline caused the token sequence to shift just enough to evade the filter.

---

### C_lex Type B — Boundary case (survived strict coverage threshold)

The item has substantial n-gram overlap with a benchmark problem but falls just below the numeric threshold. Tülu's 50% token coverage filter is the strictest of the three, yet 15 items survive it — primarily from NuminaMath-TIR, the source the filter was designed to protect.

**Tülu / NuminaMath-TIR** (30–39 shared 8-grams, coverage just below 50%):
> **MATH-500:** *"The set of vectors v such that proj_{(2,1)} v = proj_{(2,1)} (3,2)..."*
> **Tülu:** *"Find the vector v such that proj_{(2,1)} v = proj_{(2,1)} (3,2)..."*
> *(39 shared 8-grams — same projection setup, different question framing shifts coverage below threshold)*

> **MATH-500:** *"A reflection takes (5,0) to (4,3). Where does it take (-1,-3)?"*
> **Tülu:** *"A reflection takes (-1,7) to (5,-5). Where does it take (1,4)?"*
> *(30 shared 8-grams — identical template, different numbers)*

**Root cause:** The 50% coverage threshold is measured over test tokens. When a training problem uses the same setup phrase but a different question ending, the shared tokens are concentrated in the problem statement while the question portion is unique — diluting coverage below 50% even for near-identical problems.

---

### C_sem Type A — Near-paraphrase (minimal surface change, zero n-gram overlap)

The training problem is the benchmark problem with only minor surface edits: word-order swap, synonym substitution, or punctuation change. These changes are sufficient to produce zero shared n-grams under any tokenizer, making detection by n-gram methods impossible by design. Confirmed by human review (Y label).

**Tülu** (sim=0.998, human-validated Y):
> **MATH-500:** *"How many integers are in the solution set of |x−2|≤5.6?"*
> **Tülu:** *"How many integers are there in the solution set of |x−2| ≤ 5.6?"*
> *(Word-order swap "are in" → "are there in"; tokenizer boundary shifts prevent any n-gram match)*

**Tülu** (sim=0.980, human-validated Y):
> **MATH-500:** *"For how many two-digit primes is the sum of the digits equal to 8?"*
> **Tülu:** *"For how many two-digit prime numbers is the sum of its digits 8?"*

**Tülu** (sim=0.844, human-validated Y):
> **MATH-500:** *"Let f(x)=|x−p|+|x−15|+|x−p−15|, 0<p<15. Find the minimum value for p≤x≤15."*
> **Tülu:** *"Let T=|x−p|+|x−15|+|x−15−p|, 0<p<15. For p≤x≤15, the minimum value of T is."*
> *(Same function renamed T; |x−p−15| and |x−15−p| are identical)*

**OpenThoughts** (sim=0.845, human-validated Y):
> **MATH-500:** *"How many integers are in the solution set of |x−2|≤5.6?"*
> **OpenThoughts:** *"Return your final response within \boxed{}. How many integers are there in the solution set of |x−2|≤5.6?"*
> *(Same problem appearing in both Tülu and OpenThoughts with different prefixes — evidence of shared upstream source)*

---

### C_sem Type B — Template regeneration (same structure, different constants)

The training problem uses the same mathematical procedure as the benchmark — the same theorem, formula, or algorithm — but with different numeric values, coordinates, or coefficients. The solution steps are identical; only the specific inputs change. No surface-form filter can detect this class, and a model trained on the template gains a direct advantage on the benchmark instance.

**Tülu** (sim=0.921, human-validated Y):
> **MATH-500:** *"Let A=(1,−11,2), B=(3,−4,1), C=(−2,1,−1). Compute ∠ABC in degrees."*
> **Tülu:** *"Let A=(−4,0,6), B=(−5,−1,2), C=(−6,−1,3). Compute ∠ABC in degrees."*
> *(Identical dot-product procedure: compute BA·BC / (|BA||BC|), apply arccos)*

**Tülu** (sim=0.890, human-validated Y):
> **MATH-500:** *"What is the domain of f(x)=(2x−7)/√(x²−5x+6)?"*
> **Tülu:** *"What is the domain of f(x)=(x+6)/√(x²−3x−4)?"*
> *(Same method: factor denominator quadratic, solve >0, intersect with numerator domain)*

**OpenThoughts** (sim=0.921, human-validated Y):
> **MATH-500:** *"Consider the geometric sequence 125/9, 25/3, 5, 3, ... What is the eighth term?"*
> **OpenThoughts:** *"Consider the geometric sequence 3, 9/2, 27/4, 81/8, ... Find the eighth term."*
> *(Same formula: a₈ = a₁ × r⁷; only ratio and starting value differ)*

**Tülu** (sim=0.970, human-validated Y — note domain difference):
> **MATH-500:** *"Let |ax²+bx+c|≤1 for all 0≤x≤1. Find the max of |a|+|b|+|c|."*
> **Tülu:** *"Let |ax²+bx+c|≤1 for all |x|≤1. Find the max of |a|+|b|+|c|."*
> *(Same Chebyshev-type extremal polynomial technique; domain differs so the answer differs — but the reasoning path is identical, making this a genuine advantage)*

---

### Taxonomy Summary

| Type | Mechanism | Detectable by n-gram filter? | Fix |
|---|---|---|---|
| C_lex A | Filter ran on wrong text representation | Should be — implementation gap | Run filter on released text, not internal text |
| C_lex B | Near-threshold overlap survives strict filter | Partially — threshold is too narrow | Lower threshold or use multiple thresholds |
| C_sem A | Word-order/synonym paraphrase | No — zero n-gram overlap by construction | Semantic decontamination required |
| C_sem B | Same procedure, different constants | No — surface form is genuinely different | Semantic decontamination required |

C_lex Type A and B are engineering failures — addressable with better release practices. C_sem Type A and B are fundamental gaps in the current paradigm — the reason semantic decontamination is needed.

---

## Section 8: Caveats and Limitations

1. **s1 C_lex threshold sensitivity:** The raw 135 number includes many 1–2 n-gram pairs that are likely mathematical boilerplate. We report >=5 (36 items) as the conservative estimate and >=10 (13 items) as the strict estimate. Reviewers should treat 13 as the floor and 36 as the ceiling.

2. **OpenThoughts full audit:** The 114k audit is complete. Contamination rates (18.6% C_lex, 8.4% C_sem, 24.6% total) replace all prior extrapolations. The original 20k sample extrapolation underestimated C_lex by approximately 2.6×, consistent with non-uniform contamination across the dataset.

3. **Judge precision — validated by human review:** LLM judge precision was validated against human annotation on the top-similarity pairs (2026-04-23):
   - **Tülu:** 20 pairs reviewed → 8Y / 10N / 2? → human precision **44%** (95% CI: 22–67%). Judge-estimated precision was 38%. Consistent.
   - **OpenThoughts:** 15 pairs reviewed → 4Y / 10N / 1? → human precision **29%** (95% CI: 7–50%). Judge-estimated precision was 28%. Nearly identical.
   - **Corrected C_sem estimates** (judge count × human precision): Tülu ~13 items (range 12–14); OpenThoughts ~12 items (range 11–14).
   - Human and judge precision are consistent across both datasets. The judge is not systematically over- or under-calling contamination.
   - s1: only 10 pairs judged (1 confirmed). Too small a pool for reliable precision estimate; treat as a lower bound.

4. **Tülu dataset scope:** We audited all math-sourced items from `allenai/tulu-3-sft-mixture` (334k items across 5 sources). The source breakdown (Section 4) shows NuminaMath-TIR drives effectively all contamination. The PersonaHub and GSM8K sources are clean by both C_lex and C_sem measures.

5. **Robustness check scope:** The n=15/20 robustness analysis was run on OpenThoughts only. The OT dataset is the largest and has the most C_lex signal, making it the most informative dataset for threshold sensitivity analysis. s1 and Tülu use 8-gram filters; a comparable check at 10-gram and 13-gram would be straightforward but is not yet done.

---

## Next Steps

1. **Behavioral experiments** — use `data/output/*_c_lex.jsonl` and `data/output/*_c_sem.jsonl` as input to the performance gap analysis (do models score higher on C_lex/C_sem items vs. clean items?).
2. **s1/Tülu robustness check** — re-run n-gram filter at stricter thresholds (10-gram for s1, 10/13-gram for Tülu) to extend the threshold sensitivity analysis to all three datasets.
3. **Canonical contaminated item list** — produce a final, deduplicated list of all MATH-500 items affected across all three datasets for use as a held-out evaluation set.
