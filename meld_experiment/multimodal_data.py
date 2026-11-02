"""Strict MELD alignment and leakage-free prompts for shared T/A/V classification."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from common import LABELS, keyed, messages, read_jsonl

MODES = ('T', 'T+A', 'T+V', 'T+A+V')
WEIGHTS = (0.25, 0.20, 0.20, 0.35)
MEDIA_FILES = {
    'train': ('caption_train_audio_v1/descriptions.jsonl', 'caption_train_video_2fps_v1/descriptions_repaired.jsonl'),
    'dev': ('caption_dev_audio_v4/descriptions.jsonl', 'caption_dev_video_2fps_v1/descriptions_repaired.jsonl'),
    'test': ('caption_test_audio_v1/descriptions.jsonl', 'caption_test_video_2fps_v1/descriptions_final.jsonl'),
}
MISSING = '[unavailable]'


def description_index(path, split, modality):
    result = {}
    for record in read_jsonl(path):
        sample_id = record['sample_id']
        if sample_id in result:
            raise ValueError(f'duplicate {modality} sample ID: {sample_id}')
        if record.get('split') != split or record.get('modality') != modality:
            raise ValueError(f'incorrect split or modality: {sample_id}')
        status = record.get('status')
        value = record.get('description')
        if status == 'ok':
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'empty description marked ok: {sample_id}')
            value = value.strip()
        elif status in ('missing_media', 'target_unavailable', 'error'):
            if value is not None:
                raise ValueError(f'unavailable record contains description: {sample_id}')
            value = None
        else:
            raise ValueError(f'unknown status {status!r}: {sample_id}')
        result[sample_id] = {'description': value, 'status': status}
    return result


def load_split(data_dir, media_root, split, include_labels=False):
    data_dir, media_root = Path(data_dir), Path(media_root)
    rows = read_jsonl(data_dir / f'{split}.inputs.jsonl')
    expected = keyed(rows)
    audio, video = (description_index(media_root / 'runs' / p, split, modality)
                    for p, modality in zip(MEDIA_FILES[split], ('audio', 'video')))
    for modality, entries in [('audio', audio), ('video', video)]:
        if entries.keys() != expected.keys():
            raise ValueError(f'{split} {modality} ID mismatch: missing={len(expected.keys()-entries.keys())}, extra={len(entries.keys()-expected.keys())}')
    refs = keyed(read_jsonl(data_dir / f'{split}.gold.jsonl')) if include_labels else None
    if refs is not None and refs.keys() != expected.keys():
        raise ValueError(f'{split} label ID mismatch')
    combined = []
    for row in rows:
        key = row['id']
        if not key.startswith(split + '/') or 'label' in row:
            raise ValueError(f'invalid inference input: {key}')
        item = dict(row, audio_description=audio[key]['description'], audio_status=audio[key]['status'],
                    video_description=video[key]['description'], video_status=video[key]['status'])
        if refs is not None:
            label = refs[key]['label']
            if label not in LABELS:
                raise ValueError(f'invalid label: {key}')
            item['label'] = label
        combined.append(item)
    return combined


def mode_for_sample(seed, epoch, sample_id):
    digest = hashlib.sha256(f'{seed}|{epoch}|{sample_id}'.encode()).digest()
    val = int.from_bytes(digest[:8], 'big') / 2**64
    running = 0.0
    for mode, weight in zip(MODES, WEIGHTS):
        running += weight
        if val < running:
            return mode
    return MODES[-1]


def multimodal_messages(row, mode, history=None):
    if mode not in MODES:
        raise ValueError(f'unknown modality mode: {mode}')
    prompt = messages(row, history=history)[0]['content']
    suffix = '\n\nOutput only the emotion label in lowercase.'
    if not prompt.endswith(suffix):
        raise ValueError('Unexpected base prompt format')
    extras = []
    if mode in ('T+A', 'T+A+V'):
        extras.append('Audio Description: ' + (row.get('audio_description') or MISSING))
    if mode in ('T+V', 'T+A+V'):
        extras.append('Visual Description: ' + (row.get('video_description') or MISSING))
    if extras:
        prompt = prompt[:-len(suffix)] + '\n\n' + '\n'.join(extras) + suffix
    return [{'role': 'user', 'content': prompt}]


def encode_multimodal_prompt(tokenizer, row, mode, max_tokens):
    history = list(row['history'])
    while True:
        result = tokenizer.apply_chat_template(multimodal_messages(row, mode, history), tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        if len(result) <= max_tokens:
            return result, len(row['history']) - len(history)
        if not history:
            raise ValueError(f"{row['id']}: target and selected descriptions exceed token budget {max_tokens}")
        history.pop(0)


def encode_training_row(tokenizer, row, mode, max_length, answer_budget):
    target = tokenizer(row['label'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    if len(target) > answer_budget:
        raise ValueError('increase max_new_tokens for label and EOS')
    prompt, dropped = encode_multimodal_prompt(tokenizer, row, mode, max_length - answer_budget)
    return {'input_ids': prompt + target, 'labels': [-100] * len(prompt) + target}


def audit_splits(data_dir, media_root):
    report = {}
    for split in ('train', 'dev', 'test'):
        rows = load_split(data_dir, media_root, split, include_labels=True)
        report[split] = {'samples': len(rows), 'audio_status': dict(Counter(r['audio_status'] for r in rows)),
                         'video_status': dict(Counter(r['video_status'] for r in rows))}
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Validate the entire MELD media alignment on CPU')
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--media-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit_splits(args.data_dir, args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
