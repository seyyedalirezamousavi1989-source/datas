#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
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
