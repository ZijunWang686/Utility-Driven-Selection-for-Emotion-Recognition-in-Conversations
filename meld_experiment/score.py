"""Strict ID alignment; invalid generated labels count as errors, never get dropped."""
import argparse
import json
from pathlib import Path
from common import LABELS, keyed, read_jsonl


def evaluate(gold_rows, prediction_rows):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
    gold, predictions = keyed(gold_rows), keyed(prediction_rows)
    if not gold or gold.keys() != predictions.keys():
        raise ValueError(f'ID mismatch or empty reference: missing={len(gold.keys() - predictions.keys())}, extra={len(predictions.keys() - gold.keys())}')
    y_true = [gold[k]['label'] for k in gold]
    if set(y_true) - set(LABELS):
        raise ValueError('Unknown gold label')
    if any('true_label' in predictions[k] and predictions[k]['true_label'] != gold[k]['label'] for k in gold):
        raise ValueError('Prediction true_label disagrees with reference')
    y_pred = [predictions[k]['prediction'] if predictions[k]['prediction'] in LABELS else '__invalid__' for k in gold]
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), zero_division=0)
    return {'samples': len(gold), 'accuracy': 100 * accuracy_score(y_true, y_pred),
            'weighted_f1': 100 * f1_score(y_true, y_pred, labels=list(LABELS), average='weighted', zero_division=0),
            'macro_f1': 100 * f1_score(y_true, y_pred, labels=list(LABELS), average='macro', zero_division=0),
            'invalid_predictions': y_pred.count('__invalid__'), 'score_scale': '0-100',
            'per_class': {label: {'precision': float(p[i] * 100), 'recall': float(r[i] * 100), 'f1': float(f[i] * 100), 'support': int(n[i])} for i, label in enumerate(LABELS)},
            'confusion_labels': [*LABELS, '__invalid__'],
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[*LABELS, '__invalid__']).tolist()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gold', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
