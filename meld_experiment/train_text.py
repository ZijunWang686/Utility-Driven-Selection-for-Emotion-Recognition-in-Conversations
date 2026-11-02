"""Train only emotion-label tokens. CUDA required; no model download at import time."""
import argparse
import json
import gc
import shutil
import subprocess
import sys
from pathlib import Path
from common import read_jsonl, keyed, encode_supervised
from score import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/text_lora_qwen3_8b.json')
    p.add_argument('--model', help='Local model directory or Hugging Face identifier')
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/text_qlora'))
    p.add_argument('--smoke', action='store_true', help='Two training steps; output must use a separate directory')
    p.add_argument('--lora', action='store_true', help='Use BF16 LoRA without 4-bit quantization')
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU. Preparation/downloads work in CPU mode; training requires enabling the GPU.')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('This configuration requires a BF16-capable GPU such as RTX 4090D.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f'Refusing to overwrite nonempty run directory: {args.output_dir}')
    cfg = json.loads(Path(args.config).read_text())
    if args.model:
        cfg['model_name'] = args.model
    if args.lora:
        cfg['use_4bit'] = False
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = read_jsonl(args.data_dir / 'train.sft.jsonl')
    gold = keyed(read_jsonl(args.data_dir / 'dev.gold.jsonl'))
    dev_inputs = read_jsonl(args.data_dir / 'dev.inputs.jsonl')
    if set(keyed(dev_inputs)) != set(gold):
        raise ValueError('Validation inputs and labels have different IDs')
    dev_rows = [{**row, 'label': gold[row['id']]['label']} for row in dev_inputs]
    if args.smoke:
        train_rows, dev_rows = train_rows[:32], dev_rows[:16]

    def encode(row):
        return encode_supervised(tokenizer, row, cfg['max_length'], cfg['max_new_tokens'])

    train_data, dev_data = [encode(r) for r in train_rows], [encode(r) for r in dev_rows]

    def collate(rows):
        width = max(len(r['input_ids']) for r in rows)
        return {
            'input_ids': torch.tensor([r['input_ids'] + [tokenizer.pad_token_id] * (width - len(r['input_ids'])) for r in rows]),
            'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (width - len(r['input_ids'])) for r in rows]),
            'labels': torch.tensor([r['labels'] + [-100] * (width - len(r['labels'])) for r in rows]),
        }

    load_args = {'torch_dtype': torch.bfloat16, 'device_map': {'': 0}, 'attn_implementation': 'sdpa'}
    if cfg['use_4bit']:
        load_args['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], **load_args)
    model.config.use_cache = False
    if cfg['use_4bit']:
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir),
        num_train_epochs=cfg['epochs'], max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=2 if args.smoke else cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'], warmup_ratio=0.03, weight_decay=0.01,
        lr_scheduler_type='cosine', bf16=True, tf32=True, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps' if args.smoke else 'epoch', save_strategy='steps' if args.smoke else 'epoch',
        eval_steps=2 if args.smoke else None, save_steps=2 if args.smoke else 500,
        prediction_loss_only=True,
        load_best_model_at_end=args.smoke, metric_for_best_model='eval_loss', greater_is_better=False,
        save_total_limit=2 if args.smoke else None, logging_steps=1 if args.smoke else 10, report_to='none',
        seed=cfg['seed'], data_seed=cfg['seed'], dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=dev_data,
                      data_collator=collate, processing_class=tokenizer)
    trainer.train()
    trainer.save_state()
    final = args.output_dir / 'adapter'
    if args.smoke:
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))
        cfg.update({'smoke': True, 'best_checkpoint': trainer.state.best_model_checkpoint,
                    'checkpoint_selection': 'smoke only: minimum validation loss'})
    else:
        checkpoints = sorted(args.output_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        if not checkpoints:
            raise RuntimeError('No epoch checkpoints saved')
        # Free the training model before starting separate evaluation processes.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        results = []
        for checkpoint in checkpoints:
            (checkpoint / 'run_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            predictions = checkpoint / 'dev_predictions.jsonl'
            subprocess.run([sys.executable, str(Path(__file__).with_name('predict_text.py')),
                            '--adapter', str(checkpoint), '--model', cfg['model_name'],
                            '--inputs', str(args.data_dir / 'dev.inputs.jsonl'),
                            '--output', str(predictions)], check=True)
            metrics = evaluate(list(gold.values()), read_jsonl(predictions))
            (checkpoint / 'dev_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            results.append({'checkpoint': str(checkpoint), 'weighted_f1': metrics['weighted_f1'],
                            'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']})
        best = max(results, key=lambda item: item['weighted_f1'])
        best_dir = Path(best['checkpoint'])
        final.mkdir(parents=True, exist_ok=False)
        for name in ('adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin'):
            if (best_dir / name).is_file():
                shutil.copy2(best_dir / name, final / name)
        tokenizer.save_pretrained(str(final))
        cfg.update({'smoke': False, 'best_checkpoint': str(best_dir),
                    'checkpoint_selection': 'maximum Dev Weighted-F1; earliest checkpoint breaks ties',
                    'checkpoint_results': results})
        (args.output_dir / 'checkpoint_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print('Selected checkpoint:', json.dumps(best), flush=True)
    (final / 'run_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
