"""
03_llm_judge.py

LLM-as-judge verification of embedding candidates.
Uses GPT-4o-mini for cost efficiency.

Processes candidates in order of similarity score (highest first).
Saves results incrementally so you can resume if interrupted.

Cost estimate: ~$0.01 per 100 pairs at GPT-4o-mini pricing.

Output: results/judge_results/{project}_judge_results.jsonl
"""

from openai import OpenAI
import jsonlines
import os
import json
import time
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


JUDGE_PROMPT = """\
You are an expert mathematician evaluating potential training data contamination in LLM benchmarks.

You will be shown two math problems: one from a training dataset and one from a test benchmark.
Your task is to classify whether the training problem contaminated the benchmark problem.

CLASSIFICATION CRITERIA:

CONTAMINATED: The two problems share the same core mathematical insight, technique, or approach.
A model trained on the training problem would have a meaningful advantage on the benchmark problem
beyond general mathematical knowledge. This includes:
- Same identity or theorem applied (e.g., both use arccos + arcsin = π/2)
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


def make_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or set the environment variable.")
    return OpenAI(api_key=api_key)


def judge_pair(client, train_problem, math500_problem, model="gpt-4o-mini", max_retries=3):
    prompt = JUDGE_PROMPT.format(
        train_problem=train_problem[:1500],
        math500_problem=math500_problem[:1500],
    )
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=30,
            )
            raw = resp.choices[0].message.content or ""

            # Strip markdown code fences
            raw_clean = re.sub(r"```(?:json)?\s*", "", raw).strip()

            # Try direct parse
            try:
                return json.loads(raw_clean)
            except json.JSONDecodeError:
                pass

            # Extract outermost {...} block (handles text before/after JSON)
            m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if m:
                candidate = m.group()
                # Replace unescaped backslashes in string values (LaTeX)
                # Only fix backslashes not already part of valid JSON escapes
                candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

            # Last resort: extract just the classification field
            cls_match = re.search(r'"classification"\s*:\s*"(CONTAMINATED|RELATED|CLEAN)"', raw, re.IGNORECASE)
            conf_match = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', raw, re.IGNORECASE)
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
            if cls_match:
                return {
                    "classification": cls_match.group(1).upper(),
                    "confidence": conf_match.group(1).upper() if conf_match else "LOW",
                    "reasoning": reason_match.group(1) if reason_match else "extracted via regex",
                    "shared_insight": None,
                }

            last_err = f"no JSON in: {raw[:100]}"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print(f"\n  [WARN] Judge failed: {last_err}")
    return {"classification": "ERROR", "confidence": "LOW",
            "reasoning": last_err[:200], "shared_insight": None}


def run_judge(project_name, candidates_path, output_path,
              max_candidates=None, resume=True, model="gpt-4o-mini"):
    print(f"\n{'='*60}")
    print(f"Running LLM judge: {project_name}")

    client = make_client()

    with jsonlines.open(candidates_path) as r:
        candidates = list(r)

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    if max_candidates:
        candidates = candidates[:max_candidates]

    print(f"  Candidates to judge: {len(candidates)}")

    already_judged = {}
    if resume and Path(output_path).exists():
        with jsonlines.open(output_path) as r:
            for item in r:
                already_judged[(item["math500_id"], item["train_id"])] = item
        print(f"  Resuming: {len(already_judged)} already judged")

    results = list(already_judged.values())
    to_judge = [c for c in candidates
                if (c["math500_id"], c["train_id"]) not in already_judged]
    print(f"  Remaining to judge: {len(to_judge)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for candidate in tqdm(to_judge, desc=f"Judging {project_name}"):
        judgment = judge_pair(
            client,
            candidate["train_problem"],
            candidate["math500_problem"],
            model=model,
        )
        result = {
            **candidate,
            **judgment,
            "contamination_type": (
                "c_sem"           if judgment["classification"] == "CONTAMINATED"
                else "related"    if judgment["classification"] == "RELATED"
                else "clean_candidate"
            ),
        }
        results.append(result)

        # Incremental save — safe to interrupt
        with jsonlines.open(output_path, mode="w") as w:
            w.write_all(results)

        time.sleep(0.1)

    confirmed = [r for r in results if r["classification"] == "CONTAMINATED"]
    related   = [r for r in results if r["classification"] == "RELATED"]
    clean     = [r for r in results if r["classification"] == "CLEAN"]
    total     = len([r for r in results if r["classification"] != "ERROR"])

    print(f"\n  Results for {project_name}:")
    print(f"    CONTAMINATED (C_sem): {len(confirmed)}")
    print(f"    RELATED:              {len(related)}")
    print(f"    CLEAN:                {len(clean)}")
    print(f"    Judge precision:      {len(confirmed)/max(total,1)*100:.1f}%")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-s1",   type=int, default=100)
    parser.add_argument("--max-tulu", type=int, default=150)
    parser.add_argument("--max-ot",   type=int, default=100)
    parser.add_argument("--model",    default="gpt-4o-mini")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    resume = not args.no_resume

    run_judge("s1",
              "results/embedding_candidates/s1_candidates.jsonl",
              "results/judge_results/s1_judge_results.jsonl",
              max_candidates=args.max_s1, resume=resume, model=args.model)

    run_judge("tulu",
              "results/embedding_candidates/tulu_candidates.jsonl",
              "results/judge_results/tulu_judge_results.jsonl",
              max_candidates=args.max_tulu, resume=resume, model=args.model)

    run_judge("openthoughts",
              "results/embedding_candidates/openthoughts_candidates.jsonl",
              "results/judge_results/openthoughts_judge_results.jsonl",
              max_candidates=args.max_ot, resume=resume, model=args.model)
