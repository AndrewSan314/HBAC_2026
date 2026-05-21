# Clean Slate Forecast Pipeline

This repository root contains a reproducible forecasting pipeline that does not read previous submission outputs. The minimal data-loading, scoring, and submission helpers live in `core.py`, so the clean slate pipeline no longer depends on the old experimental `src/hbac_forecast.py`.

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

This is the clean replacement for the leaderboard-derived final artifact: it trains XGBoost from `dataset/train.csv`, uses volume matching for stable SKU totals, strengthens the public block rebound after the late non-Sunday closure, then selects the 92 high-impact still-active SKUs from training data only and scales their evaluation/private block to a 1.60 private/public ratio. It does not read any previous submission file from `outputs/`.

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
- No previous submission file or leaderboard-derived prediction is used.
