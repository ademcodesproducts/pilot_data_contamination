"""
02_embedding_retrieval.py

For each MATH-500 problem, retrieve the top-k most similar training
items using sentence embeddings and FAISS approximate nearest neighbor.

Uses: all-mpnet-base-v2 (best quality for math problems)
Threshold: 0.70 (retrieves more candidates; LLM judge will filter)

Excludes: any pairs already identified as C_lex in Stage 1

Output: results/embedding_candidates/{project}_candidates.jsonl
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import jsonlines
import os
from pathlib import Path
from tqdm import tqdm

SIMILARITY_THRESHOLD = 0.70
TOP_K = 5
MODEL_NAME = "all-mpnet-base-v2"

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model: {MODEL_NAME}  (device={device})")
embed_model = SentenceTransformer(MODEL_NAME, device=device)

# Smaller batch size avoids GPU OOM on large datasets
BATCH_SIZE = 32


def embed_texts_chunked(texts, cache_path, batch_size=BATCH_SIZE, chunk_size=5000):
    """
    Embed texts in chunks, saving each chunk to disk.
    Resumes from last completed chunk if interrupted.
    Returns a memory-mapped numpy array.
    """
    cache_path = Path(cache_path)
    n = len(texts)
    dim = 768  # all-mpnet-base-v2 output dim

    # Final output file
    out_file = cache_path.with_suffix(".npy")

    # Check if fully cached
    if out_file.exists():
        arr = np.load(out_file)
        if arr.shape[0] == n:
            print(f"  Loaded cached embeddings from {out_file}")
            return arr

    # Chunk checkpoint files
    chunks = []
    start = 0
    chunk_idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_file = cache_path.parent / f"{cache_path.stem}_chunk{chunk_idx:04d}.npy"
        if chunk_file.exists():
            arr = np.load(chunk_file)
            if arr.shape[0] == end - start:
                chunks.append(arr)
                start = end
                chunk_idx += 1
                print(f"  Resuming: chunk {chunk_idx} already done ({end}/{n})")
                continue

        print(f"  Encoding items {start}–{end} ({end}/{n})...")
        batch_texts = texts[start:end]
        embs = embed_model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        np.save(chunk_file, embs)
        chunks.append(embs)
        # Free GPU memory and pause briefly between chunks
        if device == "cuda":
            torch.cuda.empty_cache()
        import time; time.sleep(3)
        start = end
        chunk_idx += 1

    # Concatenate and save final
    all_embs = np.concatenate(chunks, axis=0)
    np.save(out_file, all_embs)
    # Clean up chunk files
    for f in cache_path.parent.glob(f"{cache_path.stem}_chunk*.npy"):
        f.unlink()
    print(f"  Saved full embeddings to {out_file}")
    return all_embs


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def load_ngram_hits(ngram_hits_path):
    try:
        with jsonlines.open(ngram_hits_path) as r:
            hits = list(r)
        return {(h["math500_id"], h["train_id"]) for h in hits}
    except FileNotFoundError:
        return set()


def run_embedding_retrieval(project_name, train_path, math500_path,
                            ngram_hits_path, output_path):
    print(f"\n{'='*60}")
    print(f"Running embedding retrieval: {project_name}")

    with jsonlines.open(math500_path) as r:
        math500 = list(r)
    with jsonlines.open(train_path) as r:
        train_data = list(r)

    c_lex_pairs = load_ngram_hits(ngram_hits_path)
    print(f"  Excluding {len(c_lex_pairs)} known C_lex pairs")

    # Filter empty texts, keep index mapping
    valid_train   = [(i, item) for i, item in enumerate(train_data)   if item.get("problem", "").strip()]
    valid_math500 = [(i, item) for i, item in enumerate(math500)      if item.get("problem", "").strip()]

    print(f"  Train items: {len(valid_train)}, MATH-500 items: {len(valid_math500)}")

    train_indices  = [i for i, _ in valid_train]
    train_texts    = [item["problem"] for _, item in valid_train]
    m500_indices   = [i for i, _ in valid_math500]
    m500_texts     = [item["problem"] for _, item in valid_math500]

    cache_dir = Path("results/embedding_candidates")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("  Embedding training data (chunked, resumable)...")
    train_embs = embed_texts_chunked(
        train_texts,
        cache_path=cache_dir / f"{project_name}_train_embs",
    )

    print("  Building FAISS index...")
    index = build_faiss_index(train_embs)

    print("  Embedding MATH-500 and searching...")
    m500_embs = embed_texts_chunked(
        m500_texts,
        cache_path=cache_dir / f"{project_name}_math500_embs",
    )
    similarities, neighbor_idx = index.search(m500_embs, TOP_K)

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
                "project": project_name,
                "contamination_type": "c_sem_candidate",
                "math500_id": math_item["math500_id"],
                "math500_problem": math_item["problem"],
                "math500_answer": math_item.get("answer", ""),
                "math500_subject": math_item.get("subject", ""),
                "math500_level": math_item.get("level", -1),
                "train_id": train_item["train_id"],
                "train_problem": train_item.get("problem", ""),
                "train_solution": train_item.get("solution", "")[:500],
                "similarity_score": float(sim),
                "ngram_overlap": 0,
            })

    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as w:
        w.write_all(candidates)

    unique_math500 = len({c["math500_id"] for c in candidates})
    print(f"  Found {len(candidates)} candidates above threshold {SIMILARITY_THRESHOLD}")
    print(f"  Unique MATH-500 items: {unique_math500}")
    print(f"  Saved to {output_path}")
    return candidates


if __name__ == "__main__":
    s1_candidates = run_embedding_retrieval(
        project_name="s1",
        train_path="data/s1k.jsonl",
        math500_path="data/math500.jsonl",
        ngram_hits_path="results/ngram_hits/s1_ngram_hits.jsonl",
        output_path="results/embedding_candidates/s1_candidates.jsonl",
    )

    tulu_candidates = run_embedding_retrieval(
        project_name="tulu",
        train_path="data/tulu_math.jsonl",
        math500_path="data/math500.jsonl",
        ngram_hits_path="results/ngram_hits/tulu_ngram_hits.jsonl",
        output_path="results/embedding_candidates/tulu_candidates.jsonl",
    )

    ot_candidates = run_embedding_retrieval(
        project_name="openthoughts",
        train_path="data/openthoughts.jsonl",
        math500_path="data/math500.jsonl",
        ngram_hits_path="results/ngram_hits/openthoughts_ngram_hits.jsonl",
        output_path="results/embedding_candidates/openthoughts_candidates.jsonl",
    )

    print("\n=== EMBEDDING RETRIEVAL COMPLETE ===")
    print(f"s1:           {len(s1_candidates)} candidates")
    print(f"Tülu 3:       {len(tulu_candidates)} candidates")
    print(f"OpenThoughts: {len(ot_candidates)} candidates")
