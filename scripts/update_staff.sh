#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/update_staff.sh [key_file] [base_url]
# Defaults: nva_api_keys_test.json, https://api.test.nva.aws.unit.no

KEY_FILE=${1:-nva_api_keys_test.json}
BASE_URL=${2:-https://api.test.nva.aws.unit.no}

echo "[1/4] Activating venv"
source .venv/bin/activate

echo "[2/4] Refreshing staff metadata from staff.csv -> data/staff.yaml + data/staff_records.jsonl"
timeout 900 python -m app.index.refresh_staff --staff-csv staff.csv --output data/staff.yaml --records-output data/staff_records.jsonl --skip-index

echo "[3/4] Syncing NVA publications from ${BASE_URL}"
timeout 900 python -m app.index.nva.sync --key-file "$KEY_FILE" --base-url "$BASE_URL" --output data/nva/results.jsonl

echo "[4/4] Rebuilding retrieval index (GPU if available, CPU fallback)"
set +e
timeout 900 python -m app.index.build --nva-results data/nva/results.jsonl
BUILD_STATUS=$?
if [ $BUILD_STATUS -ne 0 ]; then
  echo "[fallback] GPU build failed (status $BUILD_STATUS). Retrying on CPU..."
  CUDA_VISIBLE_DEVICES="" timeout 900 python -m app.index.build --nva-results data/nva/results.jsonl
fi
set -e

echo "Update complete."
