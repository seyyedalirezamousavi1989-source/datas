#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f "out/04_multitimeframe/XAUUSD_M15_mtf.csv" || ! -f "out/01_normalized/XAUUSD_M5_normalized.csv" ]]; then
  echo "Missing Stage 4 inputs. Extract/copy Stage 4 outputs into xauusd_stage5/out/ before rebuilding." >&2
  echo "Required: out/04_multitimeframe/XAUUSD_M15_mtf.csv and out/01_normalized/XAUUSD_M5_normalized.csv" >&2
  exit 2
fi
python3 -m stage5.labeling
python3 -m stage5.probabilities
python3 -m stage5.feature_report
python3 -m stage5.evaluate
python3 -m stage5.build_datasets
python3 -m stage5.audits
python3 -m unittest discover -s tests
python3 -m stage5.package_sft
python3 out/09_sft/join_and_verify.py
python3 -m stage5.preflight_sft
