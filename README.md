# Clean Slate Forecast Pipeline

This folder contains the maintained forecasting pipeline and final-recipe
rebuild code. The minimal data-loading, scoring, and submission helpers live in
`core.py`, so the normal clean-slate model path no longer depends on the old
experimental `src/hbac_forecast.py`.

Data requirement:

This repository intentionally contains source code only. To reproduce a submission, place the competition data files at:

```text
dataset/train.csv
dataset/sample_submission.csv
```

Alternatively, pass another folder with the same two files via `--data-dir`.

Final selected submissions:

```powershell
python run_pipeline.py `
  --build-final-artifacts `
  --rebuild-chronos-anchor `
  --final-recipe both_final `
  --output-dir outputs\clean_slate_final_recipes
```

This writes the two selected submissions:

```text
outputs/clean_slate_final_recipes/submission_private_skuratio_g120_h50_clip0p75_1p8.csv
outputs/clean_slate_final_recipes/submission_breakthrough_k20_private_target150.csv
```

With `--build-final-artifacts --rebuild-chronos-anchor`, the pipeline first
creates the intermediate artifacts under
`outputs/clean_slate_final_recipes/artifacts/`:

- rebuilds the public sliced anchor from the configured `current_best` and
  direction source submissions using the active-mid-high uncertainty gate,
  non-Sunday public horizons, lower-only direction, and 3% per-cell cap;
- reruns Chronos-Bolt on the top active profit-weight SKUs to produce the
  failed-direction probe;
- builds the inverse-Chronos public k10.75 and k20 shapes;
- trains the XGBVM evaluation shape from `train.csv`;
- writes the private-source, baseline, and final selected submissions.

No CSV values are embedded in the code. The public anchor step is deterministic
and writes its own audit/metadata files. If the anchor has already been rebuilt,
`--rebuild-chronos-anchor` can be omitted and `--chronos-anchor-source` can point
to that file.

`g120_h50` uses the generated k10.75 public block, generated XGBVM evaluation shape, and per-SKU historical ratio calibration. `k20_target150` uses the generated k20 public block, generated XGBVM evaluation shape, and scales the selected evaluation block to a 1.50 private/public weighted ratio.

Source-only final command with risk-managed private calibration:

```powershell
python run_pipeline.py `
  --output-dir outputs\clean_slate_source_only_global160 `
  --submission-name submission_clean_slate_source_only_global160.csv `
  --objective tweedie `
  --volume-matching `
  --closure-rebound-strength 0.811 `
  --median-weight 0.65 `
  --alpha-grid 0.70 `
  --xgb-scale-grid 1.0 `
  --validation-scale-grid 1.0 `
  --evaluation-scale-grid 1.0 `
  --top-n 500 `
  --cv-folds 3 `
  --train-stride-days 28 `
  --max-train-rows 700000 `
  --tweedie-power 1.5 `
  --min-child-weight 10 `
  --recency-halflife 365 `
  --n-estimators 260 `
  --learning-rate 0.035 `
  --private-calibration global_ratio `
  --private-calibration-pool-top-n 500 `
  --private-calibration-top-k 92 `
  --private-calibration-score active_weight `
  --private-active-min-days 5 `
  --private-global-target-ratio 1.60
```

This is a self-contained forecasting recipe: it trains XGBoost from `dataset/train.csv`, checks the configuration with rolling cross-validation, anchors SKU totals with volume matching against a robust statistical backbone, and strengthens the first forecast block after a recent non-Sunday closure. The private/evaluation calibration is also source-only: it selects the 92 high-impact still-active SKUs from training data and scales their evaluation block to a fixed 1.60 private/public ratio, motivated by historical September-to-October demand behavior in the training period. It does not read any previous submission file from `outputs/`.

Stat-only reference command:

```powershell
python run_pipeline.py `
  --output-dir outputs\clean_slate_closure_rebound_stat_s0p54_mw0p65 `
  --submission-name submission_clean_slate_closure_rebound_stat_s0p54_mw0p65.csv `
  --closure-rebound-strength 0.54 `
  --median-weight 0.65 `
  --alpha-grid 0 `
  --validation-scale-grid 1.0 `
  --evaluation-scale-grid 1.0 `
  --top-n 500 `
  --cv-folds 3
```

Rationale:

- `closure_rebound` detects a recent non-Sunday global closure in the training tail.
- It builds a standard `starter_dow_blend` forecast and an open-business-day forecast that excludes global zero-transaction days from the rolling base.
- For the first 28 forecast days only, it blends non-Sunday forecasts toward the open-business-day forecast.
- Sundays keep the standard forecast, preserving the observed near-zero Sunday structure.
- No previous submission file or external target signal is used.
