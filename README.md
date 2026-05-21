# HBAC Forecast Pipeline

This repository contains the final forecasting pipeline used to generate the
selected HBAC submissions. The code is organized as a reproducible
workflow with shared data loading, feature generation, model training, artifact
rebuild, and submission writing in one entry point.

## Data

Place the competition files in:

```text
dataset/train.csv
dataset/sample_submission.csv
```

You can also pass another folder with the same two files via `--data-dir`.

Generated files are written under `outputs/`, which is intentionally ignored by
git.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The Chronos rebuild path also requires the local environment to have the Chronos
runtime and its model dependencies available.

## Reproduce Final Submissions

Run:

```powershell
python run_pipeline.py `
  --build-final-artifacts `
  --rebuild-chronos-anchor `
  --final-recipe both_final `
  --output-dir outputs\clean_slate_final_recipes
```

This writes the two selected submission CSV files under the configured output
directory, together with audit and diagnostic files.

## Pipeline Overview

The pipeline performs these steps:

- loads and validates the competition data;
- builds a robust statistical demand backbone from recent SKU history;
- trains a direct gradient-boosted demand model for high-impact SKUs;
- applies volume matching so model shape is kept while SKU-level total demand
  remains stable;
- rebuilds the public-window adjustment artifacts from model outputs;
- runs the Chronos-based public-shape rebuild stage;
- combines the public and evaluation-window forecasts into the two selected
  final submissions;
- writes diagnostics and audit CSVs next to the generated submissions.

No dataset files or generated submission artifacts are stored in this
repository.
