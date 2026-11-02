# Multimodal and utility-selection pipeline

Run these commands inside `meld_experiment/`, after preparing the text data as described in the root README. Use a new output directory for each experiment.

## 1. Generate media descriptions

Caption scripts take the official CSV and the media directory for a single split. They provide `--audit-only`, `--limit`, and `--resume`; use `--help` for all options. Example audio and video generation:

```bash
python caption_meld.py --csv /path/to/MELD/train_sent_emo.csv --media-dir /path/to/train_media --split train --modality audio --model /path/to/Qwen2.5-Omni-3B --output-dir data/media_cache/runs/caption_train_audio_v1
python caption_meld_video_2fps.py --csv /path/to/MELD/train_sent_emo.csv --media-dir /path/to/train_media --split train --modality video --model /path/to/Qwen3-VL-4B-Instruct --output-dir data/media_cache/runs/caption_train_video_2fps_v1
```

`multimodal_data.py` expects the following files below `--media-root`:

```text
runs/
  caption_train_audio_v1/descriptions.jsonl
  caption_train_video_2fps_v1/descriptions_repaired.jsonl
  caption_dev_audio_v4/descriptions.jsonl
  caption_dev_video_2fps_v1/descriptions_repaired.jsonl
  caption_test_audio_v1/descriptions.jsonl
  caption_test_video_2fps_v1/descriptions_final.jsonl
```

These names identify the cache versions used by the experiments. Generate each required split separately. The caption scripts produce `descriptions.jsonl`; `repair_visual_jsonl.py --input ... --output ...` creates a separate repaired visual file and preserves repair provenance. Use `accept_visual_candidates.py` only for older `needs_review` records; it retains the fact that the speaker was not independently verified. The loader requires exact sample-ID coverage. Missing or unavailable media are represented explicitly, rather than dropping samples. Training and Dev evaluation need train/dev caches; test caches are not required for those commands.

## 2. Train and evaluate the shared classifier

```bash
python train_multimodal.py --model /path/to/Qwen3-8B --media-root data/media_cache --output-dir runs/shared_weighted --imbalance weighted
python predict_multimodal.py --model /path/to/Qwen3-8B --adapter runs/shared_weighted/adapter --media-root data/media_cache --mode T --output-dir runs/shared_weighted/dev_modes/T
```

Repeat prediction for `T+A`, `T+V`, and `T+A+V`, using output directories `T_A`, `T_V`, and `T_A_V`. The same adapter is used for all four modes. Training mixes the four modes with probabilities 25%, 20%, 20%, and 35%. The final checkpoint is selected by Dev T+A+V weighted F1. `--smoke` is available for a short training check.

## 3. Produce dialogue-disjoint OOF predictions

```bash
python prepare_oof_folds.py --output-dir runs/oof_folds --seed 42
```

For each `k` from 0 to 4, train a separate weighted teacher on that fold's training IDs, then predict only its held-out IDs (the example below is fold 0):

```bash
python train_multimodal.py --model /path/to/Qwen3-8B --media-root data/media_cache --output-dir runs/oof/fold_0/train --imbalance weighted --train-id-file runs/oof_folds/fold_0_train_ids.txt
python predict_oof_fold.py --model /path/to/Qwen3-8B --adapter runs/oof/fold_0/train/adapter --media-root data/media_cache --train-ids runs/oof_folds/fold_0_train_ids.txt --heldout-ids runs/oof_folds/fold_0_heldout_ids.txt --output-dir runs/oof/fold_0/oof_predictions
```

After all five folds finish:

```bash
python merge_oof_probabilities.py --fold-dir runs/oof_folds --run-root runs/oof --output-dir runs/oof/merged
```

The merge validates fold coverage, data hashes, and label probabilities, then computes each modality's gold-label log-likelihood gain relative to T.

## 4. Train and evaluate the lightweight selector

```bash
python select_utility_grid.py --oof-root runs/oof --dev-predictions runs/shared_weighted/dev_modes --output-dir runs/proxy_selector
```

The selector learns from OOF utility labels using text TF-IDF features and logistic regression. Its utility is `log_gain_vs_T - lambda * cost`, with costs T=0, T+A=0.33, T+V=0.33, T+A+V=1. The epsilon rule can favor a cheaper correct action close to the best utility. Lambda and epsilon are chosen on Dev; among grid points within one weighted-F1 percentage point of the best, selection favors lower mean cost.

Outputs include grid metrics, selected training actions, and Dev selections. This script evaluates a proxy selector and does not serialize a deployable model. Dev is used for checkpoint and grid selection; the reported Dev scores are not held-out test estimates. The cost values describe modality use, not measured end-to-end latency; caption-generation cost is separate.

`diagnose_oof.py` also contains descriptive analyses tied to the original local report layout; its utility and CSV-validation functions are shared by the selector. The diagnostic CLI is optional for the main pipeline.
