#!/usr/bin/env bash
# Download the three public datasets used in the experiments into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/nhanes

# CDISC/PHUSE pilot ADaM (PHUSE Test Data Factory)
TDF=https://github.com/phuse-org/TestDataFactory/raw/main/Updated/TDF_ADaM
for f in adsl adlbc adlbh; do curl -fsSL -o "data/raw/$f.xpt" "$TDF/$f.xpt"; done

# UCI HCV data (Hoffmann et al. 2018, CC BY 4.0)
curl -fsSL -o data/raw/hcv.zip https://archive.ics.uci.edu/static/public/571/hcv+data.zip
unzip -o -q data/raw/hcv.zip -d data/raw && rm data/raw/hcv.zip

# NHANES 2017-March 2020 pre-pandemic (CDC/NCHS, public domain)
NH=https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles
for f in P_DEMO P_BIOPRO P_CBC P_GHB P_HDL P_HSCRP P_ALB_CR P_DIQ P_MCQ P_KIQ_U; do
  curl -fsSL -o "data/raw/nhanes/$f.xpt" "$NH/$f.xpt"
done
echo "done: $(ls data/raw | wc -l | tr -d ' ') files in data/raw"
