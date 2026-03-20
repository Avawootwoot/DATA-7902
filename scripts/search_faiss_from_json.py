import pickle
import argparse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--index-path", default="persona_transcripts.index")
    parser.add_argument("--meta-path", default="persona_transcripts_metadata.pkl")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading model...")
    model = SentenceTransformer(args.model_name)

    print("Loading FAISS index...")
    index = faiss.read_index(args.index_path)

    print("Loading metadata...")
    with open(args.meta_path, "rb") as f:
        records = pickle.load(f)

    query_vec = model.encode(
        [args.query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_vec, args.top_k)

    print(f"\nQuery: {args.query}")
    print("\nTop matches:\n")

    for rank, idx in enumerate(indices[0], start=1):
        record = records[idx]
        print(f"Rank: {rank}")
        print(f"Persona ID: {record['persona_id']}")
        print(f"Name: {record['name']}")
        print(f"Role: {record['role']}")
        print(f"Turn Index: {record['turn_idx']}")
        print(f"Chunk Index: {record['chunk_idx']}")
        print(f"Distance: {distances[0][rank - 1]:.4f}")
        print(f"Text: {record['text']}")
        print("-" * 70)


if __name__ == "__main__":
    main()