#!/usr/bin/env bash
# Offline sanity check on a laptop CPU: runs the tests and the sample pipeline.
set -euo pipefail
pip install -q -r requirements.txt
pip install -q -e .
pytest -q
python -m tapid.build_dataset --source sample --copies 25 --out data/processed
python -m tapid.rag_demo --llm mock
