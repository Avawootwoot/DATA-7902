import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON = BASE_DIR / 'data' / 'raw' / 'batch_run_results.json'
DEFAULT_INDEX = BASE_DIR / 'data' / 'processed' / 'persona_transcripts.index'
DEFAULT_META = BASE_DIR / 'data' / 'processed' / 'persona_transcripts_metadata.pkl'


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Expected the JSON root to be a list of persona records.')
    return data


def clean_text(text: str) -> str:
    if not text:
        return ''
    return ' '.join(str(text).strip().split())


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


def build_chunk_records(personas: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 100, include_assistant: bool = False) -> List[Dict[str, Any]]:
    records = []
    for persona in personas:
        persona_id = persona.get('persona_id', '')
        name = persona.get('name', '')
        transcript = persona.get('transcript', [])
        if not isinstance(transcript, list):
            continue
        for turn_idx, turn in enumerate(transcript):
            role = turn.get('role', '')
            content = clean_text(turn.get('content', ''))
            if not content or content == '[INTERVIEW COMPLETE]':
                continue
            if not include_assistant and role != 'user':
                continue
            for chunk_idx, chunk in enumerate(chunk_text(content, chunk_size=chunk_size, overlap=overlap)):
                records.append({
                    'persona_id': persona_id,
                    'name': name,
                    'turn_idx': turn_idx,
                    'chunk_idx': chunk_idx,
                    'role': role,
                    'text': chunk,
                })
    return records


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', default=str(DEFAULT_JSON))
    parser.add_argument('--index-path', default=str(DEFAULT_INDEX))
    parser.add_argument('--meta-path', default=str(DEFAULT_META))
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--include-assistant', action='store_true')
    args = parser.parse_args()

    json_path = Path(args.json_path)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading JSON from: {json_path}')
    personas = load_json(json_path)
    print(f'Loaded {len(personas)} persona records')

    records = build_chunk_records(
        personas=personas,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        include_assistant=args.include_assistant,
    )
    if not records:
        raise ValueError('No transcript chunks were extracted from the JSON.')

    texts = [r['text'] for r in records]
    print(f'Created {len(texts)} text chunks')

    print(f'Loading embedding model: {args.model_name}')
    model = SentenceTransformer(args.model_name)

    print('Generating embeddings...')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype('float32')
    print(f'Embeddings shape: {embeddings.shape}')

    print('Building FAISS index...')
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_path))
    with open(meta_path, 'wb') as f:
        pickle.dump(records, f)

    print(f'Saved index to: {index_path}')
    print(f'Saved metadata to: {meta_path}')
    print('Done.')


if __name__ == '__main__':
    main()
