"""Qwen3-8B text-only zero-shot MELD evaluation on the development split."""
import argparse
import csv
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

from common import LABELS, encode_prompt, parse_prediction
from prepare_text import prepare
from score import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PROMPT_VERSION = 'meld_plan_v5_prompt_a_constrained_v1'


def label_constraint(tokenizer, prompt_length, max_new_tokens):
    """Constrain decoding to complete label token sequences followed by EOS."""
    sequences = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(not seq for seq in sequences) or len({tuple(seq) for seq in sequences}) != len(sequences):
        raise ValueError('Labels must have distinct, nonempty token sequences')
    if max(map(len, sequences)) + 1 > max_new_tokens:
        raise ValueError('max_new_tokens is too small for a label and EOS')
    def allowed(_batch_id, input_ids):
        prefix = input_ids[prompt_length:].tolist()
        next_tokens = set()
        for seq in sequences:
            if seq[:len(prefix)] == prefix:
                next_tokens.add(seq[len(prefix)] if len(prefix) < len(seq) else tokenizer.eos_token_id)
        if not next_tokens:
            raise ValueError('Generated tokens do not match any label')
        return sorted(next_tokens)
    return allowed


def final_answer(raw, mode):
    """Never parse a thinking trace as an emotion prediction."""
    if mode == 'think':
        if '</think>' not in raw:
            return '', '__invalid__'
        raw = raw.rsplit('</think>', 1)[1]
    for token in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
        raw = raw.replace(token, '')
    answer = raw.strip()
    return answer, parse_prediction(answer)


def find_csv(root, split):
    root = Path(root).expanduser()
    candidates = [root / f'{split}_sent_emo.csv',
                  root / 'raw' / f'{split}_sent_emo.csv',
                  root / 'MELD.Raw' / f'{split}_sent_emo.csv',
                  root / 'MELD' / f'{split}_sent_emo.csv']
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('MELD CSV not found. Checked: ' + ', '.join(map(str, candidates)))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split', choices=['dev'], default='dev',
                   help='This baseline scores Dev only; Test is reserved for final evaluation.')
    p.add_argument('--mode', choices=['think', 'no_think'], required=True)
    p.add_argument('--model', default='Qwen/Qwen3-8B', help='Hub ID or local model directory')
    p.add_argument('--output-root', type=Path, default=Path('meld_experiment/runs/zero_shot'))
    p.add_argument('--limit', type=int, help='First N Dev samples, for a smoke run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window', type=int, default=12)
    p.add_argument('--max-input-tokens', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=None)
    return p.parse_args()


def main():
    args = arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.window < 0 or args.max_input_tokens < 1:
        raise ValueError('--window must be nonnegative and --max-input-tokens positive')
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (1024 if args.mode == 'think' else 16)
    if max_new_tokens < 1:
        raise ValueError('--max-new-tokens must be positive')
    csv_path = find_csv(args.data_root, args.split)
    inputs, gold, _ = prepare(csv_path, args.split, window=args.window)
    if len(inputs) != 1109:
        raise ValueError(f'Expected 1109 Dev samples, found {len(inputs)}')
    if args.limit is not None:
        inputs, gold = inputs[:args.limit], gold[:args.limit]
    if not inputs:
        raise ValueError('No Dev samples found')
    run_dir = args.output_root.expanduser() / (
        f"{args.split}_{args.mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(run_dir / 'run.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    log = logging.getLogger(__name__)
    generation = {'do_sample': False, 'max_new_tokens': max_new_tokens, 'batch_size': 1}
    config = {'model': args.model, 'mode': args.mode, 'enable_thinking': args.mode == 'think',
              'prompt_version': PROMPT_VERSION, 'seed': args.seed, 'generation': generation,
              'output_constraint': 'seven_label_token_trie' if args.mode == 'no_think' else 'none',
              'data_csv': str(csv_path), 'split': args.split, 'samples': len(inputs),
              'limit': args.limit, 'window': args.window, 'max_input_tokens': args.max_input_tokens,
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_dir': str(run_dir.resolve())}
    config_path = run_dir / 'run_config.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info('Run directory: %s', run_dir)
    run_started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is required; run this experiment on the Linux server.')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={'': 0},
            attn_implementation='sdpa').eval()
        model.config.use_cache = True
        config.update({'torch_version': torch.__version__,
                       'transformers_version': __import__('transformers').__version__,
                       'model_revision': getattr(model.config, '_commit_hash', None),
                       'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16'})
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        torch.cuda.reset_peak_memory_stats()
        predictions = []
        with (run_dir / 'predictions.jsonl').open('w', encoding='utf-8') as f:
            for index, (row, ref) in enumerate(zip(inputs, gold), 1):
                # Only row (text and speaker) is passed to the model. ref is used after generation.
                prompt_ids, dropped = encode_prompt(tokenizer, row, args.max_input_tokens,
                                                     enable_thinking=args.mode == 'think')
                tensor = torch.tensor([prompt_ids], device='cuda')
                torch.cuda.synchronize()
                start = time.perf_counter()
                error = None
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=tensor, attention_mask=torch.ones_like(tensor),
                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                            prefix_allowed_tokens_fn=(label_constraint(tokenizer, len(prompt_ids), max_new_tokens)
                                                      if args.mode == 'no_think' else None),
                            **{k: v for k, v in generation.items() if k != 'batch_size'})
                    new_ids = output[0, len(prompt_ids):]
                    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
                    answer, prediction = final_answer(raw, args.mode)
                    length = len(new_ids)
                except (RuntimeError, ValueError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    log.exception('Generation failed for %s', row['id'])
                    raw, answer, prediction, length = '', '', '__invalid__', 0
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                record = {'id': row['id'], 'true_label': ref['label'],
                          'prediction': prediction, 'final_text': answer,
                          'raw_output': raw, 'mode': args.mode, 'new_tokens': length,
                          'input_tokens': len(prompt_ids), 'dropped_history': dropped,
                          'latency_seconds': elapsed, 'generation_error': error,
                          'hit_max_new_tokens': length >= max_new_tokens}
                predictions.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                if index % 100 == 0 or index == len(inputs):
                    log.info('Completed %s/%s', index, len(inputs))
        metrics = evaluate(gold, predictions)
        csv_path = run_dir / 'predictions.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'true_label', 'pred_label'])
            writer.writeheader()
            writer.writerows({'sample_id': r['id'], 'true_label': r['true_label'],
                              'pred_label': r['prediction']} for r in predictions)
        from evaluate_meld import evaluate_meld
        evaluate_meld(csv_path, run_dir / 'evaluation', expected_count=len(inputs), reference_rows=gold)
        metrics.update({'invalid_rate': metrics['invalid_predictions'] / len(inputs),
                        'average_latency_seconds': sum(r['latency_seconds'] for r in predictions) / len(inputs),
                        'examples_per_second': len(inputs) / sum(r['latency_seconds'] for r in predictions),
                        'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                        'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                        'generation_errors': sum(r['generation_error'] is not None for r in predictions),
                        'max_length_hits': sum(r['hit_max_new_tokens'] for r in predictions)})
        (run_dir / 'metrics.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        config['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        config['runtime_seconds'] = time.perf_counter() - run_started
        config['status'] = 'completed'
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        log.info('Metrics: %s', json.dumps(metrics, ensure_ascii=False))
    except Exception:
        config['status'] = 'failed'
        config['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        config['runtime_seconds'] = time.perf_counter() - run_started
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        log.exception('Run failed')
        raise


if __name__ == '__main__':
    main()
