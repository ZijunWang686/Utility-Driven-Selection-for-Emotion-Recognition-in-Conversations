"""Prepare causal inputs and separate gold labels from the official MELD CSVs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from common import LABELS, write_jsonl


def prepare(csv_path, split, window=12):
    if window < 0:
        raise ValueError('window must be nonnegative')
    dialogues = defaultdict(list)
    seen = set()
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['Dialogue_ID']), int(row['Utterance_ID']))
            if key in seen:
                raise ValueError(f'Duplicate utterance {key}')
            seen.add(key)
            if row['Emotion'] not in LABELS or not row['Utterance'].strip():
                raise ValueError(f'Invalid label or empty text at {key}')
            dialogues[key[0]].append((key[1], row))
    inputs, refs, train = [], [], []
    for did, utterances in sorted(dialogues.items()):
        utterances.sort(key=lambda pair: pair[0])
        history = []
        for uid, row in utterances:
            current = {'utterance_id': uid, 'speaker': row['Speaker'], 'text': row['Utterance']}
            item = {'id': f'{split}/dia{did}_utt{uid}', 'split': split, 'dialogue_id': did,
                    'history': history[-window:] if window else [], 'current': current}
            inputs.append(item)
            refs.append({'id': item['id'], 'label': row['Emotion']})
            if split == 'train':
                train.append({**item, 'label': row['Emotion']})
            history.append(current)
    return inputs, refs, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--window', type=int, default=12)
    args = p.parse_args()
    counts = {}
    for split in ('train', 'dev', 'test'):
        inputs, refs, train = prepare(args.raw_dir / f'{split}_sent_emo.csv', split, args.window)
        write_jsonl(args.output_dir / f'{split}.inputs.jsonl', inputs)
        write_jsonl(args.output_dir / f'{split}.gold.jsonl', refs)
        if train:
            write_jsonl(args.output_dir / 'train.sft.jsonl', train)
        counts[split] = len(inputs)
    metadata = {'window': args.window, 'labels': LABELS, 'counts': counts,
                'protocol': 'official split, earlier utterances only, gold labels separated from inference inputs'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
