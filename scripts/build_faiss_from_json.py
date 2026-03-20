import os
import json
import pickle
import argparse
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of persona records.")

    return data


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().split())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def build_chunk_records(
    personas: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100,
    include_assistant: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract transcript dialogue from each persona.
    By default, only user turns are indexed, since those are the interview answers.
    """
    records = []

    for persona in personas:
        persona_id = persona.get("persona_id", "")
        name = persona.get("name", "")
        transcript = persona.get("transcript", [])

        if not isinstance(transcript, list):
            continue

        for turn_idx, turn in enumerate(transcript):
            role = turn.get("role", "")
            content = clean_text(turn.get("content", ""))

            if not content:
                continue

            if not include_assistant and role != "user":
                continue

            if content == "[INTERVIEW COMPLETE]":
                continue

            chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

            for chunk_idx, chunk in enumerate(chunks):
                records.append(
                    {
                        "persona_id": persona_id,
                        "name": name,
                        "turn_idx": turn_idx,
                        "chunk_idx": chunk_idx,
                        "role": role,
                        "text": chunk,
                    }
                )

    return records


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True, help="Path to batch_run_results.json")
    parser.add_argument("--index-path", default="persona_transcripts.index")
    parser.add_argument("--meta-path", default="persona_transcripts_metadata.pkl")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="Include assistant questions/prompts in the index as well"
    )
    args = parser.parse_args()

    print(f"Loading JSON from: {args.json_path}")
    personas = load_json(args.json_path)
    print(f"Loaded {len(personas)} persona records")

    records = build_chunk_records(
        personas=personas,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        include_assistant=args.include_assistant,
    )

    if not records:
        raise ValueError("No transcript chunks were extracted from the JSON.")

    texts = [r["text"] for r in records]
    print(f"Created {len(texts)} text chunks")

    print(f"Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    faiss.write_index(index, args.index_path)
    with open(args.meta_path, "wb") as f:
        pickle.dump(records, f)

    print(f"Saved index to: {args.index_path}")
    print(f"Saved metadata to: {args.meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()