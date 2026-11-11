#!/usr/bin/env bash
# Full experiment run on Google Colab (Runtime -> Change runtime type -> T4 GPU).
#   !git clone https://github.com/<your-username>/task-aware-injection-detector
#   %cd task-aware-injection-detector
#   !bash scripts/run_colab.sh
set -euo pipefail

pip install -q -r requirements.txt
pip install -q -e .

# 1. Training data: SQuAD passages + synthetic injections + hard negatives
python -m tapid.build_dataset --source squad --max-passages 20000 --out data/processed

# 2. Proposed model and the context-free ablation (B1)
python -m tapid.train --config configs/default.yaml --out outputs/proposed
python -m tapid.train --config configs/default.yaml --no-task --out outputs/b1_context_free

# 3. Evaluate both on every test set that exists (E1-E3)
TESTS=$(ls data/processed/test.jsonl data/processed/*/test.jsonl 2>/dev/null || true)
python -m tapid.evaluate --model outputs/proposed --test $TESTS
python -m tapid.evaluate --model outputs/b1_context_free --test $TESTS

# 4. Off-the-shelf baselines (B2, B4; B3 needs Hugging Face licence acceptance)
python -m tapid.baselines --baseline protectai --test $TESTS
python -m tapid.baselines --baseline piguard --test $TESTS
