# Utility-Driven Selection for Emotion Recognition in Conversations

MELD experiments with Qwen3-8B: text baselines, a shared multimodal LoRA classifier, dialogue-disjoint out-of-fold (OOF) utility estimation, and a lightweight text-based modality selector.

The four classifier inputs are **T**, **T+A**, **T+V**, and **T+A+V**. Audio and visual evidence are generated descriptions. The selector uses dialogue text to choose which evidence to include.

This release contains the core research code. The current selector is a TF-IDF/logistic-regression proxy; a Qwen/GRPO selector is not included. Experiment results are recorded in [RESULT.md](RESULT.md).

## Code map

| Stage | Files in `meld_experiment/` |
| --- | --- |
| Data preparation | `prepare_text.py`, `download_meld.py`, `audit_meld.py` |
| Text baseline | `zero_shot_meld.py`, `train_text.py`, `predict_text.py` |
| Media descriptions | `caption_meld.py`, `caption_meld_video_2fps.py`, `prompts/` |
| Shared classifier | `multimodal_data.py`, `train_multimodal.py`, `predict_multimodal.py` |
| OOF utilities | `prepare_oof_folds.py`, `predict_oof_fold.py`, `merge_oof_probabilities.py` |
| Selection | `select_utility_grid.py`; utility rules in `diagnose_oof.py` |
| Evaluation | `score.py`, root-level `evaluate_meld.py`, `tests/` |

Datasets, media, model weights, adapters, predictions, and local server scripts are excluded. Obtain MELD separately from its [official repository](https://github.com/declare-lab/MELD). Keep the official train/dev/test splits.

## Setup and CPU checks

Use Python 3.10/3.11. Install a CUDA-compatible PyTorch build separately on the GPU server, then install classifier dependencies:

```bash
python -m pip install -r meld_experiment/requirements.txt
python -m unittest discover -s meld_experiment/tests -v
```

For CPU-only data preparation and tests, only these packages are needed:

```bash
python -m pip install 'pandas>=2,<3' 'scikit-learn==1.6.1'
python -m unittest discover -s meld_experiment/tests -v
```

The pinned classifier environment uses Transformers 4.51.3. Media captioning needs a **separate environment** with support for `Qwen3VLForConditionalGeneration` / `Qwen2_5OmniForConditionalGeneration`, video decoding, and audio processing; those GPU paths are not covered by CPU tests.

## Text baseline

Run the zero-shot wrapper from the repository root; replace `/path/to/MELD` with the directory containing the official CSV files:

```bash
bash scripts/check_server.sh --data-root /path/to/MELD
bash scripts/run_meld_baseline.sh --data-root /path/to/MELD --split dev --mode no_think
```

For training and the remaining examples, work inside `meld_experiment/`:

```bash
cd meld_experiment
python prepare_text.py --raw-dir /path/to/MELD --output-dir data/processed
python train_text.py --config configs/text_lora_qwen3_8b.json --output-dir runs/text_lora
python predict_text.py --adapter runs/text_lora/adapter --inputs data/processed/dev.inputs.jsonl --output runs/text_lora/dev_predictions.jsonl
python score.py --gold data/processed/dev.gold.jsonl --predictions runs/text_lora/dev_predictions.jsonl --output runs/text_lora/dev_metrics.json
```

Preparation preserves sample IDs and keeps inference inputs separate from gold labels. Prompts use the current utterance and up to 12 preceding utterances from the same dialogue.

## Multimodal classification and utility selection

See [the pipeline guide](meld_experiment/README.md) for media cache layout, shared classifier commands, OOF training, and the proxy selector.

CPU checks validate preprocessing, prompt construction, label constraints, evaluation alignment, and modality handling. Full training, caption generation, and OOF experiments require the corresponding datasets, model weights, and GPU environments.
