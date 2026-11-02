"""Predict without loading gold labels. Greedy decoding, one utterance at a time."""
import argparse
import json
from pathlib import Path
from common import read_jsonl, encode_prompt, parse_prediction
from zero_shot_meld import label_constraint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--model', help='Override base model location after moving a run')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Smoke testing only; omit for final scoring')
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise SystemExit('Prediction configuration requires CUDA.')
    if args.output.exists():
        raise SystemExit(f'Output exists; choose a new filename: {args.output}')
    cfg = json.loads((args.adapter / 'run_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    opts = {'torch_dtype': torch.bfloat16, 'device_map': {'': 0}, 'attn_implementation': 'sdpa'}
    if cfg['use_4bit']:
        opts['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model or cfg['model_name'], **opts), args.adapter)
    model.eval()
    model.config.use_cache = True
    rows = read_jsonl(args.inputs)
    if args.limit:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for index, row in enumerate(rows, 1):
            if 'label' in row:
                raise ValueError('Use *.inputs.jsonl, not training records or gold files')
            tokens, dropped = encode_prompt(tokenizer, row, cfg['max_length'] - cfg['max_new_tokens'], enable_thinking=False)
            ids = torch.tensor([tokens], device='cuda')
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=cfg['max_new_tokens'], pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=label_constraint(tokenizer, len(tokens), cfg['max_new_tokens']))
            text = tokenizer.decode(out[0, len(tokens):], skip_special_tokens=True)
            f.write(json.dumps({'id': row['id'], 'prediction': parse_prediction(text), 'raw_output': text,
                                'dropped_history': dropped}, ensure_ascii=False) + '\n')
            f.flush()
            if index % 100 == 0:
                print(f'{index}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
