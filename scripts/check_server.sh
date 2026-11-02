#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --data-root PATH [--model HUB_ID_OR_PATH] [--python PYTHON]" >&2
}

DATA_ROOT=""
MODEL="Qwen/Qwen3-8B"
PYTHON="python3"
while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="${2:?missing data root}"; shift 2 ;;
    --model) MODEL="${2:?missing model}"; shift 2 ;;
    --python) PYTHON="${2:?missing Python executable}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$DATA_ROOT" ]]; then usage; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project: $PROJECT_DIR"
echo "Python executable: $PYTHON"
"$PYTHON" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi unavailable" >&2
fi
"$PYTHON" - "$DATA_ROOT" "$MODEL" "$PROJECT_DIR" <<'PY'
import importlib
import pathlib
import sys

data_root, model_name, project = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(project) / 'meld_experiment'))
from zero_shot_meld import find_csv

print('Data root:', pathlib.Path(data_root).expanduser())
for split in ('train', 'dev', 'test'):
    try:
        print(f'{split} CSV:', find_csv(data_root, split))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
for name in ('torch', 'transformers', 'accelerate', 'sklearn'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'unknown'))
torch = importlib.import_module('torch')
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check server GPU and CUDA PyTorch installation.')
print('GPU:', torch.cuda.get_device_name(0))
print('BF16 supported:', torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit('This baseline requires BF16 support.')
if pathlib.Path(model_name).expanduser().exists():
    print('Local model:', pathlib.Path(model_name).expanduser().resolve())
else:
    print('Model Hub ID:', model_name, '(weights checked when baseline starts)')
PY
