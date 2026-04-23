"""
run_openthoughts_full.py

Downloads the full OpenThoughts-114K dataset and runs the contamination
detection pipeline (ngram -> embedding -> judge) on all 114k items.

Results go to *_full files so the 20k sample baseline is preserved:
  data/openthoughts_full.jsonl
  results/ngram_hits/openthoughts_full_ngram_hits.jsonl
  results/embedding_candidates/openthoughts_full_{train,math500}_embs.npy
  results/embedding_candidates/openthoughts_full_candidates.jsonl
  results/judge_results/openthoughts_full_judge_results.jsonl

Run 05_validate_and_report.py afterwards — it auto-detects the _full files.
"""

import os
import sys
import argparse
from pathlib import Path

# Run from the contamination_detection/ directory
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))


# ── Step 0: Download full dataset ────────────────────────────────────────────

def download_openthoughts_full():
    out_path = Path("data/openthoughts_full.jsonl")
    if out_path.exists():
        import jsonlines
        with jsonlines.open(out_path) as r:
            n = sum(1 for _ in r)
        print(f"Full OpenThoughts already cached: {n} items at {out_path}")
        return n

    print("Downloading OpenThoughts-114K (full)...")
    from datasets import load_dataset
    import jsonlines

    ds = None
    for dataset_id in ["open-thoughts/OpenThoughts-114k", "open-thoughts/OpenThoughts-114K"]:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split="train")
            print(f"  OK: {len(ds)} items")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load OpenThoughts-114K from HuggingFace.")

    items = []
    for i, row in enumerate(ds):
        messages = row.get("conversations") or row.get("messages", [])
        problem, solution = "", ""
        if messages:
            for msg in messages:
                role = msg.get("role") or msg.get("from", "")
                content = msg.get("content") or msg.get("value", "")
                if role in ("user", "human") and not problem:
                    problem = content
                elif role in ("assistant", "gpt") and not solution:
                    solution = content
        else:
            problem = row.get("problem") or row.get("question", "")
            solution = row.get("solution") or row.get("response", "")

        if not problem.strip():
            continue

        items.append({
            "train_id":       f"ot_{i:06d}",
            "problem":        problem,
            "solution":       solution[:1000],
            "source":         row.get("source", "openthoughts"),
            "dataset":        "openthoughts",
            "original_index": i,
        })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(ds)} items...")

    Path("data").mkdir(exist_ok=True)
    import jsonlines as jl
    with jl.open(out_path, mode="w") as w:
        w.write_all(items)
    print(f"  Saved {len(items)} items to {out_path}")
    return len(items)


# ── Step 1: N-gram filter ─────────────────────────────────────────────────────

def run_ngram(skip_if_exists=True):
    out_path = Path("results/ngram_hits/openthoughts_full_ngram_hits.jsonl")
    if skip_if_exists and out_path.exists():
        import jsonlines
        with jsonlines.open(out_path) as r:
            hits = list(r)
        print(f"N-gram hits already exist: {len(hits)} hits at {out_path}")
        return hits

    from scripts.ngram_runner import run_ngram_audit_standalone
    hits = run_ngram_audit_standalone(
        project_name="openthoughts_full",
        train_path="data/openthoughts_full.jsonl",
        math500_path="data/math500.jsonl",
        n=13,
        threshold_mode="any",
        output_path=str(out_path),
    )
    return hits


def run_ngram_inline(skip_if_exists=True):
    """Inline version — avoids import issues."""
    out_path = Path("results/ngram_hits/openthoughts_full_ngram_hits.jsonl")
    if skip_if_exists and out_path.exists():
        import jsonlines
        with jsonlines.open(out_path) as r:
            hits = list(r)
        print(f"N-gram hits already exist: {len(hits)} hits at {out_path}")
        return hits

    from transformers import AutoTokenizer
    import jsonlines
    from tqdm import tqdm

    print("\nLoading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    def get_ngrams(text, n):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    n = 13
    project_name = "openthoughts_full"

    print(f"\nRunning n-gram audit: {project_name}  (n={n}, threshold=any)")

    with jsonlines.open("data/math500.jsonl") as r:
        math500 = list(r)
    with jsonlines.open("data/openthoughts_full.jsonl") as r:
        train_data = list(r)

    print(f"  {len(train_data)} train items x {len(math500)} MATH-500 = {len(train_data)*len(math500):,} pairs")
    print("  Precomputing MATH-500 n-grams...")
    math500_ngrams = [(item, get_ngrams(item.get("problem",""), n), item.get("problem","")) for item in math500]

    hits = []
    for train_item in tqdm(train_data, desc="n-gram filter"):
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
            hits.append({
                "project":             project_name,
                "contamination_type":  "c_lex",
                "math500_id":          math_item["math500_id"],
                "math500_problem":     test_text,
                "math500_answer":      math_item.get("answer", ""),
                "math500_subject":     math_item.get("subject", ""),
                "math500_level":       math_item.get("level", -1),
                "train_id":            train_item["train_id"],
                "train_problem":       train_text,
                "n_shared_ngrams":     len(shared),
                "token_coverage":      0.0,
                "ngram_n":             n,
                "threshold_mode":      "any",
                "filter_should_have_caught": True,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(hits)

    unique = len({h["math500_id"] for h in hits})
    print(f"  Found {len(hits)} hits  ({unique} unique MATH-500 items)")
    print(f"  Saved to {out_path}")
    return hits


# ── Step 2: Embedding retrieval ───────────────────────────────────────────────

def run_embedding(skip_if_exists=True):
    out_path = Path("results/embedding_candidates/openthoughts_full_candidates.jsonl")
    if skip_if_exists and out_path.exists():
        import jsonlines
        with jsonlines.open(out_path) as r:
            cands = list(r)
        print(f"Embedding candidates already exist: {len(cands)} at {out_path}")
        return cands

    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import jsonlines
    from tqdm import tqdm
    import torch

    SIMILARITY_THRESHOLD = 0.70
    TOP_K = 5
    MODEL_NAME = "all-mpnet-base-v2"
    BATCH_SIZE = 16
    CHUNK_SIZE = 5000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning embedding retrieval: openthoughts_full  (device={device})")
    embed_model = SentenceTransformer(MODEL_NAME, device=device)

    with jsonlines.open("data/math500.jsonl") as r:
        math500 = list(r)
    with jsonlines.open("data/openthoughts_full.jsonl") as r:
        train_data = list(r)

    # Load known C_lex pairs to exclude
    c_lex_pairs = set()
    ngram_path = Path("results/ngram_hits/openthoughts_full_ngram_hits.jsonl")
    if ngram_path.exists():
        with jsonlines.open(ngram_path) as r:
            for item in r:
                c_lex_pairs.add((item["math500_id"], item["train_id"]))
    print(f"  Excluding {len(c_lex_pairs)} known C_lex pairs")

    valid_train   = [(i, item) for i, item in enumerate(train_data) if item.get("problem","").strip()]
    valid_math500 = [(i, item) for i, item in enumerate(math500)    if item.get("problem","").strip()]
    train_texts   = [item["problem"] for _, item in valid_train]
    m500_texts    = [item["problem"] for _, item in valid_math500]

    cache_dir = Path("results/embedding_candidates")

    def embed_chunked(texts, cache_stem):
        final = cache_dir / f"{cache_stem}.npy"
        if final.exists():
            arr = np.load(final)
            if arr.shape[0] == len(texts):
                print(f"  Loaded cached embeddings: {final}")
                return arr

        chunks, start, idx = [], 0, 0
        while start < len(texts):
            end = min(start + CHUNK_SIZE, len(texts))
            cf = cache_dir / f"{cache_stem}_chunk{idx:04d}.npy"
            if cf.exists():
                arr = np.load(cf)
                if arr.shape[0] == end - start:
                    chunks.append(arr)
                    start, idx = end, idx + 1
                    print(f"  Resuming chunk {idx} ({end}/{len(texts)})")
                    continue
            print(f"  Encoding {start}-{end} ({len(texts)} total)...")
            embs = embed_model.encode(
                texts[start:end], batch_size=BATCH_SIZE,
                show_progress_bar=True, normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            np.save(cf, embs)
            chunks.append(embs)
            if device == "cuda":
                torch.cuda.empty_cache()
            import time; time.sleep(10)
            start, idx = end, idx + 1

        all_embs = np.concatenate(chunks, axis=0)
        np.save(final, all_embs)
        for f in cache_dir.glob(f"{cache_stem}_chunk*.npy"):
            f.unlink()
        print(f"  Saved {final}")
        return all_embs

    print(f"  Embedding {len(train_texts)} training items...")
    train_embs = embed_chunked(train_texts, "openthoughts_full_train_embs")

    print("  Building FAISS index...")
    index = faiss.IndexFlatIP(train_embs.shape[1])
    index.add(train_embs)

    print(f"  Embedding {len(m500_texts)} MATH-500 items...")
    m500_embs = embed_chunked(m500_texts, "openthoughts_full_math500_embs")

    print("  Searching...")
    similarities, neighbor_idx = index.search(m500_embs, TOP_K)

    train_indices = [i for i, _ in valid_train]
    m500_indices  = [i for i, _ in valid_math500]

    candidates = []
    for m_pos, (sims, nbrs) in enumerate(zip(similarities, neighbor_idx)):
        math_item = math500[m500_indices[m_pos]]
        for sim, nbr_pos in zip(sims, nbrs):
            if sim < SIMILARITY_THRESHOLD:
                continue
            train_item = train_data[train_indices[nbr_pos]]
            if (math_item["math500_id"], train_item["train_id"]) in c_lex_pairs:
                continue
            candidates.append({
                "project":          "openthoughts_full",
                "contamination_type": "c_sem_candidate",
                "math500_id":       math_item["math500_id"],
                "math500_problem":  math_item["problem"],
                "math500_answer":   math_item.get("answer", ""),
                "math500_subject":  math_item.get("subject", ""),
                "math500_level":    math_item.get("level", -1),
                "train_id":         train_item["train_id"],
                "train_problem":    train_item.get("problem", ""),
                "train_solution":   train_item.get("solution", "")[:500],
                "similarity_score": float(sim),
                "ngram_overlap":    0,
            })

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(candidates)

    unique = len({c["math500_id"] for c in candidates})
    print(f"  {len(candidates)} candidates above {SIMILARITY_THRESHOLD} ({unique} unique MATH-500 items)")
    print(f"  Saved to {out_path}")
    return candidates


# ── Step 3: LLM judge ─────────────────────────────────────────────────────────

def run_judge(max_candidates=200, model="gpt-4o-mini", skip_if_complete=True):
    out_path = Path("results/judge_results/openthoughts_full_judge_results.jsonl")
    cands_path = Path("results/embedding_candidates/openthoughts_full_candidates.jsonl")

    import jsonlines

    with jsonlines.open(cands_path) as r:
        candidates = list(r)
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    if max_candidates:
        candidates = candidates[:max_candidates]

    already_judged = {}
    if out_path.exists():
        with jsonlines.open(out_path) as r:
            for item in r:
                already_judged[(item["math500_id"], item["train_id"])] = item

    if skip_if_complete and len(already_judged) >= len(candidates):
        confirmed = sum(1 for v in already_judged.values() if v.get("classification") == "CONTAMINATED")
        print(f"Judge already complete: {len(already_judged)} judged, {confirmed} CONTAMINATED")
        return list(already_judged.values())

    from openai import OpenAI
    from dotenv import load_dotenv
    import json, time, re
    from tqdm import tqdm

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    JUDGE_PROMPT = """\
You are an expert mathematician evaluating potential training data contamination in LLM benchmarks.

You will be shown two math problems: one from a training dataset and one from a test benchmark.
Your task is to classify whether the training problem contaminated the benchmark problem.

CLASSIFICATION CRITERIA:

CONTAMINATED: The two problems share the same core mathematical insight, technique, or approach.
A model trained on the training problem would have a meaningful advantage on the benchmark problem
beyond general mathematical knowledge. This includes:
- Same identity or theorem applied (e.g., both use arccos + arcsin = pi/2)
- Same algorithmic structure (e.g., both require the same DP recurrence)
- Template variants where only numbers/variables changed but identical reasoning is needed

RELATED: The problems are in the same mathematical area and share surface similarity,
but solving one does not provide meaningful advantage on the other.
Different specific techniques are required despite topic overlap.

CLEAN: The similarity is superficial. These are genuinely independent problems
that happen to use similar keywords or problem structures but require different reasoning.

IMPORTANT: "Same topic" (e.g., both about quadratic equations) is NOT sufficient for CONTAMINATED.
The specific reasoning steps must be shared.

Training problem:
{train_problem}

Benchmark problem:
{math500_problem}

Respond with a JSON object only, no other text:
{{"classification": "CONTAMINATED" or "RELATED" or "CLEAN", "confidence": "HIGH" or "MEDIUM" or "LOW", "reasoning": "one sentence explaining the classification", "shared_insight": "if CONTAMINATED, describe the specific shared mathematical insight; else null"}}"""

    def judge_pair(train_problem, math500_problem):
        prompt = JUDGE_PROMPT.format(
            train_problem=train_problem[:1500],
            math500_problem=math500_problem[:1500],
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=300, timeout=30,
                )
                raw = resp.choices[0].message.content or ""
                raw_clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
                try:
                    return json.loads(raw_clean)
                except json.JSONDecodeError:
                    pass
                m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
                if m:
                    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', m.group())
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                cls_m = re.search(r'"classification"\s*:\s*"(CONTAMINATED|RELATED|CLEAN)"', raw, re.IGNORECASE)
                if cls_m:
                    return {"classification": cls_m.group(1).upper(), "confidence": "LOW",
                            "reasoning": "extracted via regex", "shared_insight": None}
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return {"classification": "ERROR", "confidence": "LOW", "reasoning": "judge failed", "shared_insight": None}

    results = list(already_judged.values())
    to_judge = [c for c in candidates if (c["math500_id"], c["train_id"]) not in already_judged]
    print(f"\nRunning LLM judge: openthoughts_full")
    print(f"  {len(candidates)} candidates, {len(already_judged)} already done, {len(to_judge)} remaining")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for cand in tqdm(to_judge, desc="Judging openthoughts_full"):
        j = judge_pair(cand["train_problem"], cand["math500_problem"])
        result = {
            **cand, **j,
            "contamination_type": (
                "c_sem"           if j["classification"] == "CONTAMINATED"
                else "related"    if j["classification"] == "RELATED"
                else "clean_candidate"
            ),
        }
        results.append(result)
        with jsonlines.open(out_path, mode="w") as w:
            w.write_all(results)
        time.sleep(0.1)

    confirmed = [r for r in results if r.get("classification") == "CONTAMINATED"]
    total = len([r for r in results if r.get("classification") != "ERROR"])
    print(f"\n  CONTAMINATED (C_sem): {len(confirmed)}")
    print(f"  Judge precision:      {len(confirmed)/max(total,1)*100:.1f}%")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full OpenThoughts-114K contamination pipeline")
    parser.add_argument("--skip-download",   action="store_true")
    parser.add_argument("--skip-ngram",      action="store_true")
    parser.add_argument("--skip-embedding",  action="store_true")
    parser.add_argument("--skip-judge",      action="store_true")
    parser.add_argument("--max-judge",       type=int, default=200,
                        help="Max embedding candidates to send to judge (default: 200)")
    parser.add_argument("--judge-model",     default="gpt-4o-mini")
    parser.add_argument("--no-resume",       action="store_true",
                        help="Ignore existing intermediate files and recompute")
    args = parser.parse_args()

    resume = not args.no_resume

    if not args.skip_download:
        download_openthoughts_full()

    if not args.skip_ngram:
        run_ngram_inline(skip_if_exists=resume)

    if not args.skip_embedding:
        run_embedding(skip_if_exists=resume)

    if not args.skip_judge:
        run_judge(max_candidates=args.max_judge, model=args.judge_model,
                  skip_if_complete=resume)

    print("\nDone. Run 05_validate_and_report.py --no-spotcheck to update the summary.")
