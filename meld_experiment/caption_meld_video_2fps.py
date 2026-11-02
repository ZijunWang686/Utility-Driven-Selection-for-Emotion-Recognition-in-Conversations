#!/usr/bin/env python3
"""Generate MELD video descriptions with Qwen3-VL video input at 2 fps."""
import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / 'prompts'
VIDEO_PROMPT = (PROMPT_DIR / 'video_caption.txt').read_text(encoding='utf-8').strip()


def read_rows(csv_path, split):
    seen, rows = set(), []
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did, uid = int(r['Dialogue_ID']), int(r['Utterance_ID'])
            sample_id = f'{split}/dia{did}_utt{uid}'
            if sample_id in seen:
                raise ValueError(f'duplicate ID {sample_id}')
            seen.add(sample_id)
            rows.append({'sample_id': sample_id, 'dialogue_id': did, 'utterance_id': uid,
                         'speaker': r['Speaker'].strip(),
                         'filename': f'dia{did}_utt{uid}.mp4'})
    return sorted(rows, key=lambda r: (r['dialogue_id'], r['utterance_id']))


def media_index(root):
    index = {}
    for p in root.rglob('*.mp4'):
        index.setdefault(p.name, []).append(p.resolve())
    return index


def parse_visual_response(text):
    """Parse the model's visual target decision without treating it as verified."""
    raw = text.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {'target_status', 'description'}:
        raise ValueError('Visual response must have exactly target_status and description')
    status, description = result['target_status'], result['description']
    if status not in ('visible', 'not_visible', 'uncertain') or not isinstance(description, str):
        raise ValueError('Invalid visual target status or description')
    description = description.strip()
    if status == 'visible' and not description:
        raise ValueError('Visible target requires a nonempty description')
    if status != 'visible' and description:
        raise ValueError('Uncertain or invisible target must not have a description')
    return status, description


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--media-dir', type=Path, required=True, help='Directory for exactly one split; recursively searched for mp4')
    p.add_argument('--split', choices=('train', 'dev', 'test'), required=True)
    p.add_argument('--modality', choices=('audio', 'video'), required=True)
    p.add_argument('--model', required=True, help='Local model directory')
    p.add_argument('--output-dir', type=Path, required=True, help='New, unique run directory')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--max-new-tokens', type=int, default=None)
    p.add_argument('--audit-only', action='store_true', help='Check CSV/media mapping without GPU')
    p.add_argument('--allow-missing', action='store_true', help='Record missing media as missing without dropping the sample')
    p.add_argument('--resume', action='store_true', help='Continue an interrupted run with identical settings')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        p.error('--max-new-tokens must be positive')
    if args.modality != 'video':
        p.error('This 2 fps script is for video only; use caption_meld.py for audio')
    max_new_tokens = args.max_new_tokens or 160
    rows = read_rows(args.csv, args.split)
    if args.limit:
        rows = rows[:args.limit]
    index = media_index(args.media_dir)
    missing = [r['sample_id'] for r in rows if not index.get(r['filename'])]
    duplicates = [r['sample_id'] for r in rows if len(index.get(r['filename'], [])) > 1]
    if args.resume:
        if args.audit_only or not args.output_dir.is_dir():
            p.error('--resume requires an existing generation run directory')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(args.output_dir / 'run.log'), logging.StreamHandler()])
    audit = {'split': args.split, 'modality': args.modality, 'rows': len(rows), 'missing': missing,
             'duplicate_media': duplicates, 'csv': str(args.csv.resolve()), 'media_dir': str(args.media_dir.resolve())}
    if not args.resume:
        (args.output_dir / 'alignment.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    logging.info('Alignment: %d rows, %d missing, %d ambiguous', len(rows), len(missing), len(duplicates))
    if args.audit_only:
        return
    if duplicates or (missing and not args.allow_missing):
        raise SystemExit('Fix missing or ambiguous media before generation; see alignment.json')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=512*512)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                             device_map={'': 0}, attn_implementation='sdpa')
    model.eval()
    config = {'model': str(Path(args.model).resolve()), 'split': args.split, 'modality': args.modality, 'limit': args.limit,
              'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              'media_dir': str(args.media_dir.resolve()), 'allow_missing': args.allow_missing,
              'prompt': VIDEO_PROMPT,
              'max_new_tokens': max_new_tokens, 'dtype': 'bfloat16', 'torch': torch.__version__,
              'transformers': __import__('transformers').__version__, 'gpu': torch.cuda.get_device_name(0),
              'video_input': 'native_video', 'requested_fps': 2.0,
              'visual_speaker_identification': 'model-estimated, not independently verified'}
    config_path = args.output_dir / 'run_config.json'
    output_path = args.output_dir / 'descriptions.jsonl'
    completed = {}
    if args.resume:
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('Run configuration changed; create a new run instead of resuming')
        if output_path.exists():
            with output_path.open(encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    key = record['sample_id']
                    if key in completed:
                        raise ValueError(f'Duplicate existing output: {key}')
                    completed[key] = record
        expected = {row['sample_id'] for row in rows}
        if completed.keys() - expected:
            raise ValueError('Existing output has unexpected IDs')
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    failures = sum(r['status'] == 'error' for r in completed.values())
    missing_count = sum(r['status'] == 'missing_media' for r in completed.values())
    consecutive_errors = 0
    with output_path.open('a' if args.resume else 'w', encoding='utf-8') as out:
        for i, row in enumerate(rows, 1):
            if row['sample_id'] in completed:
                continue
            start = time.perf_counter()
            matches = index.get(row['filename'], [])
            record = {**row, 'split': args.split, 'modality': args.modality,
                      'media_path': str(matches[0]) if matches else None}
            if not matches:
                missing_count += 1
                record.update({'status': 'missing_media', 'description': None, 'latency_seconds': 0})
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                logging.warning('%d/%d %s missing media', i, len(rows), row['sample_id'])
                continue
            path = matches[0]
            try:
                record.update({'requested_fps': 2.0, 'speaker_verified': False})
                if not row['speaker']:
                    raise ValueError('Missing target speaker')
                video_prompt = VIDEO_PROMPT.replace('{speaker}', row['speaker'])
                messages = [{'role': 'user', 'content': [{'type': 'video', 'video': str(path)}, {'type': 'text', 'text': video_prompt}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt', fps=2.0).to('cuda')
                if 'video_grid_thw' in inputs:
                    record['video_grid_thw'] = inputs['video_grid_thw'].tolist()
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                new_ids = ids[0][inputs.input_ids.shape[1]:]
                description = processor.batch_decode([new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                if not description:
                    raise ValueError('empty model output')
                if args.modality == 'video':
                    record['raw_output'] = description
                    target_status, candidate = parse_visual_response(description)
                    record.update({'status': 'ok' if target_status == 'visible' else 'target_unavailable',
                                   'target_status_model': target_status, 'candidate_description': candidate or None,
                                   'description': candidate if target_status == 'visible' else None, 'raw_output': description,
                                   'word_count': len(candidate.split())})
                else:
                    record.update({'status': 'ok', 'description': description, 'word_count': len(description.split())})
                record.update({'new_tokens': len(new_ids), 'hit_max_new_tokens': len(new_ids) >= max_new_tokens})
                consecutive_errors = 0
            except Exception as exc:
                failures += 1
                consecutive_errors += 1
                record.update({'status': 'error', 'description': None, 'error': f'{type(exc).__name__}: {exc}'})
                logging.exception('Failed %s', row['sample_id'])
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            record['latency_seconds'] = time.perf_counter() - start
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            logging.info('%d/%d %s %s', i, len(rows), row['sample_id'], record['status'])
            if consecutive_errors >= 5:
                logging.error('Stopping after five consecutive errors; inspect log and fix the environment before resuming')
                raise SystemExit('Stopped after five consecutive errors; existing records are retained')
    summary = {'samples': len(rows), 'missing_media': missing_count, 'failures': failures,
               'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
    if args.modality == 'video':
        with output_path.open(encoding='utf-8') as f:
            statuses = [json.loads(line)['status'] for line in f]
        summary.update({'descriptions_available': statuses.count('ok'),
                        'target_unavailable': statuses.count('target_unavailable')})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logging.info('Summary: %s', summary)
    if failures:
        raise SystemExit('Some samples failed; inspect run.log and descriptions.jsonl')


if __name__ == '__main__':
    main()
