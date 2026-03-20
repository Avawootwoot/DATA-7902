import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = BASE_DIR / 'data' / 'raw' / 'batch_run_results.json'
DEFAULT_BIOS = BASE_DIR / 'outputs' / 'biographies.json'
DEFAULT_OUT_JSON = BASE_DIR / 'outputs' / 'biography_evaluation.json'
DEFAULT_OUT_CSV = BASE_DIR / 'outputs' / 'biography_evaluation_summary.csv'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text: str) -> str:
    if text is None:
        return ''
    return ' '.join(str(text).strip().split())


def split_into_statements(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [clean_text(p) for p in parts if clean_text(p)]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.matmul(a_norm, b_norm.T)


def build_evidence_texts(persona: Dict[str, Any], include_assistant: bool = False) -> List[str]:
    evidence = []
    facts = persona.get('facts', {})
    if isinstance(facts, dict):
        for k, v in facts.items():
            k = clean_text(k)
            v = clean_text(v)
            if k and v:
                evidence.append(f'{k}: {v}')
            elif v:
                evidence.append(v)
    timeline = persona.get('timeline', [])
    if isinstance(timeline, list):
        for item in timeline:
            if not isinstance(item, dict):
                continue
            year = item.get('year', None)
            event = clean_text(item.get('event', ''))
            location = clean_text(item.get('location', ''))
            parts = []
            if year is not None:
                parts.append(f'Year: {year}')
            if event:
                parts.append(f'Event: {event}')
            if location:
                parts.append(f'Location: {location}')
            if parts:
                evidence.append(' | '.join(parts))
    transcript = persona.get('transcript', [])
    if isinstance(transcript, list):
        for turn in transcript:
            if not isinstance(turn, dict):
                continue
            role = clean_text(turn.get('role', ''))
            content = clean_text(turn.get('content', ''))
            if not content or content == '[INTERVIEW COMPLETE]':
                continue
            if include_assistant or role == 'user':
                evidence.append(content)
    return evidence


def build_fact_units(persona: Dict[str, Any]) -> List[str]:
    units = []
    facts = persona.get('facts', {})
    if isinstance(facts, dict):
        for k, v in facts.items():
            k = clean_text(k)
            v = clean_text(v)
            if k and v:
                units.append(f'{k}: {v}')
            elif v:
                units.append(v)
    timeline = persona.get('timeline', [])
    if isinstance(timeline, list):
        for item in timeline:
            if not isinstance(item, dict):
                continue
            year = item.get('year', None)
            event = clean_text(item.get('event', ''))
            location = clean_text(item.get('location', ''))
            parts = []
            if year is not None:
                parts.append(f'Year: {year}')
            if event:
                parts.append(f'Event: {event}')
            if location:
                parts.append(f'Location: {location}')
            if parts:
                units.append(' | '.join(parts))
    return units


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def evaluate_persona(biography_text: str, persona: Dict[str, Any], model: SentenceTransformer, support_threshold: float, coverage_threshold: float, include_assistant: bool = False) -> Dict[str, Any]:
    biography_text = clean_text(biography_text)
    bio_statements = split_into_statements(biography_text)
    evidence_texts = build_evidence_texts(persona, include_assistant=include_assistant)
    fact_units = build_fact_units(persona)

    if not bio_statements:
        return {'tp': 0, 'fp': 0, 'fn': len(fact_units), 'precision': 0.0, 'recall': 0.0, 'hallucination_rate': 0.0, 'num_statements': 0, 'num_facts': len(fact_units), 'supported_statements': [], 'unsupported_statements': [], 'covered_facts': [], 'missed_facts': fact_units}
    if not evidence_texts:
        return {'tp': 0, 'fp': len(bio_statements), 'fn': len(fact_units), 'precision': 0.0, 'recall': 0.0, 'hallucination_rate': 1.0 if bio_statements else 0.0, 'num_statements': len(bio_statements), 'num_facts': len(fact_units), 'supported_statements': [], 'unsupported_statements': bio_statements, 'covered_facts': [], 'missed_facts': fact_units}

    bio_emb = model.encode(bio_statements, convert_to_numpy=True).astype('float32')
    evidence_emb = model.encode(evidence_texts, convert_to_numpy=True).astype('float32')
    bio_vs_evidence = cosine_similarity_matrix(bio_emb, evidence_emb)
    best_support_scores = bio_vs_evidence.max(axis=1)

    supported_statements = []
    unsupported_statements = []
    tp = 0
    fp = 0
    for stmt, score in zip(bio_statements, best_support_scores):
        if score >= support_threshold:
            tp += 1
            supported_statements.append({'statement': stmt, 'score': float(score)})
        else:
            fp += 1
            unsupported_statements.append({'statement': stmt, 'score': float(score)})

    covered_facts = []
    missed_facts = []
    if fact_units:
        fact_emb = model.encode(fact_units, convert_to_numpy=True).astype('float32')
        fact_vs_bio = cosine_similarity_matrix(fact_emb, bio_emb)
        best_fact_scores = fact_vs_bio.max(axis=1)
        for fact, score in zip(fact_units, best_fact_scores):
            if score >= coverage_threshold:
                covered_facts.append({'fact': fact, 'score': float(score)})
            else:
                missed_facts.append({'fact': fact, 'score': float(score)})

    fn = len(missed_facts)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    hallucination_rate = safe_div(fp, tp + fp)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'hallucination_rate': hallucination_rate,
        'num_statements': len(bio_statements),
        'num_facts': len(fact_units),
        'supported_statements': supported_statements,
        'unsupported_statements': unsupported_statements,
        'covered_facts': covered_facts,
        'missed_facts': missed_facts,
    }


def find_biography_record(biography_records: List[Dict[str, Any]], persona_id: str) -> Optional[Dict[str, Any]]:
    for record in biography_records:
        if record.get('persona_id') == persona_id:
            return record
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-json', default=str(DEFAULT_RESULTS))
    parser.add_argument('--biographies-json', default=str(DEFAULT_BIOS))
    parser.add_argument('--output-json', default=str(DEFAULT_OUT_JSON))
    parser.add_argument('--output-csv', default=str(DEFAULT_OUT_CSV))
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2')
    parser.add_argument('--support-threshold', type=float, default=0.62)
    parser.add_argument('--coverage-threshold', type=float, default=0.58)
    parser.add_argument('--include-assistant', action='store_true')
    args = parser.parse_args()

    results_path = Path(args.results_json)
    bios_path = Path(args.biographies_json)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    personas = load_json(results_path)
    biography_records = load_json(bios_path)
    if not isinstance(personas, list):
        raise ValueError('results-json must contain a list of persona records.')
    if not isinstance(biography_records, list):
        raise ValueError('biographies-json must contain a list of biography records.')

    model = SentenceTransformer(args.model_name)
    detailed_results = []
    summary_rows = []

    for persona in personas:
        persona_id = persona.get('persona_id', '')
        name = persona.get('name', '')
        bio_record = find_biography_record(biography_records, persona_id)
        if not bio_record:
            summary_rows.append({'persona_id': persona_id, 'name': name, 'status': 'missing_biography', 'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'hallucination_rate': 0.0})
            continue
        biography_text = bio_record.get('biography', '')
        metrics = evaluate_persona(biography_text, persona, model, args.support_threshold, args.coverage_threshold, include_assistant=args.include_assistant)
        detailed_results.append({'persona_id': persona_id, 'name': name, 'biography': biography_text, 'metrics': metrics})
        summary_rows.append({'persona_id': persona_id, 'name': name, 'status': 'ok', 'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'], 'precision': metrics['precision'], 'recall': metrics['recall'], 'hallucination_rate': metrics['hallucination_rate'], 'num_statements': metrics['num_statements'], 'num_facts': metrics['num_facts']})

    df = pd.DataFrame(summary_rows)
    ok_df = df[df['status'] == 'ok'].copy()
    overall = {
        'personas_evaluated': int(len(ok_df)),
        'avg_precision': float(ok_df['precision'].mean()) if len(ok_df) else 0.0,
        'avg_recall': float(ok_df['recall'].mean()) if len(ok_df) else 0.0,
        'avg_hallucination_rate': float(ok_df['hallucination_rate'].mean()) if len(ok_df) else 0.0,
        'total_tp': int(ok_df['tp'].sum()) if len(ok_df) else 0,
        'total_fp': int(ok_df['fp'].sum()) if len(ok_df) else 0,
        'total_fn': int(ok_df['fn'].sum()) if len(ok_df) else 0,
        'micro_precision': safe_div(ok_df['tp'].sum(), ok_df['tp'].sum() + ok_df['fp'].sum()) if len(ok_df) else 0.0,
        'micro_recall': safe_div(ok_df['tp'].sum(), ok_df['tp'].sum() + ok_df['fn'].sum()) if len(ok_df) else 0.0,
        'micro_hallucination_rate': safe_div(ok_df['fp'].sum(), ok_df['tp'].sum() + ok_df['fp'].sum()) if len(ok_df) else 0.0,
    }

    payload = {'settings': {'results_json': str(results_path), 'biographies_json': str(bios_path), 'model_name': args.model_name, 'support_threshold': args.support_threshold, 'coverage_threshold': args.coverage_threshold, 'include_assistant': args.include_assistant}, 'overall': overall, 'per_persona': detailed_results}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print('\nEvaluation complete.')
    print(f'Saved detailed results to: {out_json}')
    print(f'Saved summary CSV to: {out_csv}')
    print('\nOverall metrics:')
    for k, v in overall.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
