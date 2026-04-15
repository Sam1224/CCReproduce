#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"

python -m orca.train_probe --out_dir outputs/probe
python -m orca.eval --ckpt outputs/probe/probe.pt --delta 0.10
