#!/usr/bin/env python3
"""Evaluate 7-class MELD utterance predictions from a CSV file.

Required columns: sample_id, true_label, pred_label and an independent reference.
Invalid/empty model predictions are counted as errors, not discarded.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
INVALID = "__INVALID__"


def evaluate_meld(csv_path, out_dir, expected_count=None, reference_path=None, reference_rows=None):
    if (reference_path is None) == (reference_rows is None):
        raise ValueError("Provide exactly one independent reference_path or reference_rows")
    data = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    required = {"sample_id", "true_label", "pred_label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(data) == 0:
        raise ValueError("Prediction file is empty")
    if expected_count is not None and len(data) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, got {len(data)}; check coverage")
    data["sample_id"] = data["sample_id"].astype(str).str.strip()
    if (data["sample_id"] == "").any():
        raise ValueError("Empty sample_id in predictions")
    if data["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id; check utterance alignment")

    if reference_path is not None:
        reference_path = Path(reference_path)
        if reference_path.suffix.lower() == '.jsonl':
            with reference_path.open(encoding='utf-8') as f:
                reference_rows = [json.loads(line) for line in f if line.strip()]
        else:
            reference_rows = pd.read_csv(reference_path, keep_default_na=False, dtype=str).to_dict('records')
    ref = pd.DataFrame(reference_rows)
    if 'sample_id' not in ref and 'id' in ref:
        ref = ref.rename(columns={'id': 'sample_id'})
    if 'true_label' not in ref and 'label' in ref:
        ref = ref.rename(columns={'label': 'true_label'})
    if not {'sample_id', 'true_label'}.issubset(ref.columns) or ref.empty:
        raise ValueError('Reference must contain nonempty sample_id and true_label (or id and label)')
    if ref['sample_id'].isna().any() or ref['true_label'].isna().any():
        raise ValueError('Reference contains missing IDs or labels')
    ref['sample_id'] = ref['sample_id'].astype(str).str.strip()
    ref['true_label'] = ref['true_label'].astype(str).str.strip().str.lower()
    if (ref['sample_id'] == '').any() or ref['sample_id'].duplicated().any():
        raise ValueError('Empty or duplicate sample_id in reference')
    if (~ref['true_label'].isin(LABELS)).any():
        raise ValueError('Unknown true label in reference')
    pred_ids, ref_ids = set(data['sample_id']), set(ref['sample_id'])
    if pred_ids != ref_ids:
        raise ValueError(f'ID mismatch: missing={len(ref_ids-pred_ids)}, extra={len(pred_ids-ref_ids)}')
    expected_gold = ref.set_index('sample_id')['true_label']
    actual_gold = data.set_index('sample_id')['true_label'].astype(str).str.strip().str.lower()
    incorrect = actual_gold != expected_gold.reindex(actual_gold.index)
    if incorrect.any():
        raise ValueError(f'True label mismatch for {int(incorrect.sum())} IDs: {actual_gold.index[incorrect].tolist()[:5]}')

    gold = data["true_label"].astype(str).str.strip().str.lower()
    pred_raw = data["pred_label"].astype(str).str.strip().str.lower()
    if (~gold.isin(LABELS)).any():
        examples = sorted(gold[~gold.isin(LABELS)].unique().tolist())
        raise ValueError(f"Unknown true labels: {examples}")
    pred = pred_raw.where(pred_raw.isin(LABELS), INVALID)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, zero_division=0
    )
    per_class = pd.DataFrame({
        "emotion": LABELS, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })
    per_class.to_csv(out_dir / "per_class.csv", index=False, float_format="%.6f")

    cm_labels = LABELS + [INVALID]  # extra column keeps invalid responses visible
    cm = confusion_matrix(gold, pred, labels=cm_labels)
    cm_df = pd.DataFrame(cm[:len(LABELS), :], index=LABELS, columns=cm_labels)
    cm_df.index.name = "true\\pred"
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    metrics = {
        "num_samples": int(len(data)),
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(f1_score(gold, pred, labels=LABELS, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)),
        "invalid_count": int((pred == INVALID).sum()),
        "invalid_rate": float((pred == INVALID).mean()),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if (pred == INVALID).any():
        data.loc[pred == INVALID].to_csv(out_dir / "invalid_predictions.csv", index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MELD 7-class utterance predictions")
    parser.add_argument("--input", required=True, help="CSV with true_label and pred_label")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics and per-class reports")
    parser.add_argument("--expected_count", type=int, default=None, help="Optional required row count (Dev=1109, Test=2610)")
    parser.add_argument("--reference", required=True, help="Independent gold CSV or JSONL with sample IDs and labels")
    args = parser.parse_args()
    evaluate_meld(args.input, args.output_dir, args.expected_count, reference_path=args.reference)
