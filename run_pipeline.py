from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from clean_slate.core import (  # noqa: E402
        BLOCK,
        FOLD_STARTS,
        HORIZON,
        DatasetBundle,
        apply_block_scales,
        load_bundle,
        profit_weights,
        rmsse_scale,
        wrmsse,
        write_submission,
    )
    from clean_slate.features import (  # noqa: E402
        attach_targets,
        build_clean_feature_cache,
        make_direct_feature_frame,
        make_training_starts,
        top_profit_indices,
    )
    from clean_slate.models import (  # noqa: E402
        blend_top_sku_predictions,
        fit_direct_xgb,
        predict_direct_xgb,
        closure_rebound_blend,
        apply_volume_matching,
    )
except ModuleNotFoundError:
    from core import (  # type: ignore  # noqa: E402
        BLOCK,
        FOLD_STARTS,
        HORIZON,
        DatasetBundle,
        apply_block_scales,
        load_bundle,
        profit_weights,
        rmsse_scale,
        wrmsse,
        write_submission,
    )
    from features import (  # type: ignore  # noqa: E402
        attach_targets,
        build_clean_feature_cache,
        make_direct_feature_frame,
        make_training_starts,
        top_profit_indices,
    )
    from models import (  # type: ignore  # noqa: E402
        blend_top_sku_predictions,
        fit_direct_xgb,
        predict_direct_xgb,
        closure_rebound_blend,
        apply_volume_matching,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified clean HBAC forecasting pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clean_slate"))
    parser.add_argument("--submission-name", default="submission_clean_slate.csv")
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--objective", choices=["tweedie", "poisson"], default="tweedie")
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--train-stride-days", type=int, default=56)
    parser.add_argument("--min-history-days", type=int, default=365)
    parser.add_argument("--max-train-rows", type=int, default=450000)
    parser.add_argument("--alpha-grid", default="0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.70,1.0")
    parser.add_argument("--xgb-scale-grid", default="1.0")
    parser.add_argument("--validation-scale-grid", default="1.0,1.10,1.20")
    parser.add_argument("--evaluation-scale-grid", default="0.95,1.0,1.05")
    parser.add_argument("--selection-metric", choices=["recent_weighted_h1_56", "h1_56_wrmsse"], default="recent_weighted_h1_56")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--volume-matching", action="store_true", help="Apply volume matching calibration")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--public-boost", type=float, default=1.05, help="Weight boost for h1-28 training rows")
    parser.add_argument("--tweedie-power", type=float, default=1.2, help="Tweedie variance power")
    parser.add_argument("--min-child-weight", type=float, default=20.0, help="XGB min_child_weight")
    parser.add_argument("--recency-halflife", type=float, default=0.0, help="Recency decay half-life in days (0=disabled)")
    parser.add_argument("--median-weight", type=float, default=0.70, help="Weight for median56 in statistical backbone")
    parser.add_argument(
        "--closure-rebound-strength",
        type=float,
        default=0.60,
        help="Blend strength from standard backbone toward open-day backbone after a recent non-Sunday closure.",
    )
    parser.add_argument(
        "--private-calibration",
        choices=["none", "sku_ratio", "global_ratio"],
        default="none",
        help=(
            "Source-only evaluation block calibration. sku_ratio uses historical SKU ratios; "
            "global_ratio applies one private/public target ratio to the selected high-impact SKUs."
        ),
    )
    parser.add_argument("--private-calibration-top-k", type=int, default=92)
    parser.add_argument("--private-calibration-pool-top-n", type=int, default=500)
    parser.add_argument(
        "--private-calibration-score",
        choices=["weighted_last28", "weighted_last56", "weighted_last112", "active_weight"],
        default="weighted_last28",
    )
    parser.add_argument("--private-active-min-days", type=int, default=5)
    parser.add_argument("--private-global-prior", type=float, default=1.20)
    parser.add_argument("--private-global-target-ratio", type=float, default=1.60)
    parser.add_argument("--private-hist-weight", type=float, default=0.50)
    parser.add_argument("--private-ratio-clip-low", type=float, default=0.75)
    parser.add_argument("--private-ratio-clip-high", type=float, default=1.80)
    parser.add_argument("--private-ratio-years", default="2022,2023,2024")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.data_dir)
    cache = build_clean_feature_cache(bundle)
    alphas = parse_float_list(args.alpha_grid)
    xgb_scales = parse_float_list(args.xgb_scale_grid)
    validation_scales = parse_float_list(args.validation_scale_grid)
    evaluation_scales = parse_float_list(args.evaluation_scale_grid)

    detail = run_cv(bundle, cache, args, alphas, xgb_scales, validation_scales, evaluation_scales)
    summary = summarize_cv(detail)
    detail.to_csv(args.output_dir / "cv_detail.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)

    if summary.empty:
        raise RuntimeError("No CV rows were produced. Check fold dates and min-history settings.")
    best = summary.sort_values([args.selection_metric, "h1_56_wrmsse"]).iloc[0]
    best_alpha = float(best["alpha"])
    best_xgb_scale = float(best["xgb_scale"])
    best_validation_scale = float(best["validation_scale"])
    best_evaluation_scale = float(best["evaluation_scale"])

    submission_path = None
    private_calibration_diagnostics = None
    if not args.skip_final:
        final_pred = make_final_prediction(
            bundle,
            cache,
            args,
            best_alpha,
            best_xgb_scale,
            best_validation_scale,
            best_evaluation_scale,
        )
        if args.private_calibration != "none":
            final_pred, private_calibration_diagnostics, private_audit = apply_private_calibration(
                final_pred,
                bundle,
                args,
            )
            private_audit.to_csv(args.output_dir / "private_calibration_audit.csv", index=False)
        submission_path = args.output_dir / args.submission_name
        write_submission(bundle.sample, bundle.sku_order, final_pred, submission_path)

    diagnostics = {
        "top_n": args.top_n,
        "cv_folds": args.cv_folds,
        "backend": "xgb",
        "volume_matching": args.volume_matching,
        "model_kind": "direct",
        "objective": args.objective,
        "best_alpha": best_alpha,
        "best_xgb_scale": best_xgb_scale,
        "best_validation_scale": best_validation_scale,
        "best_evaluation_scale": best_evaluation_scale,
        "stat_backbone": "closure_rebound",
        "closure_rebound_strength": args.closure_rebound_strength,
        "selection_metric": args.selection_metric,
        "best_cv": best.to_dict(),
        "submission_path": str(submission_path) if submission_path is not None else None,
        "private_calibration": private_calibration_diagnostics,
    }

    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


def make_stat_prediction(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    args: argparse.Namespace,
) -> np.ndarray:
    return closure_rebound_blend(
        history,
        history_dates,
        forecast_dates,
        median56_weight=args.median_weight,
        rebound_strength=args.closure_rebound_strength,
        apply_first_block_only=True,
    )


def run_cv(
    bundle: DatasetBundle,
    cache,
    args: argparse.Namespace,
    alphas: list[float],
    xgb_scales: list[float],
    validation_scales: list[float],
    evaluation_scales: list[float],
) -> pd.DataFrame:
    date_to_idx = {date: idx for idx, date in enumerate(bundle.dates)}
    fold_starts = [pd.Timestamp(x) for x in FOLD_STARTS[: args.cv_folds]]
    rows: list[dict] = []
    for fold_start in fold_starts:
        if fold_start not in date_to_idx:
            continue
        start_idx = date_to_idx[fold_start]
        end_idx = start_idx + HORIZON
        if end_idx > bundle.quantity.shape[0] or start_idx < args.min_history_days:
            continue

        cutoff_idx = start_idx - 1
        history = bundle.quantity[: start_idx]
        actual = bundle.quantity[start_idx:end_idx]
        forecast_dates = bundle.dates[start_idx:end_idx]
        weights = profit_weights(bundle.train, bundle.dates[cutoff_idx], bundle.sku_order)
        scale = rmsse_scale(history)
        sku_indices = top_profit_indices(weights, args.top_n)

        stat_pred = make_stat_prediction(history, bundle.dates[:start_idx], forecast_dates, args)
        if any(alpha > 0.0 for alpha in alphas):
            xgb_pred = fit_fold_xgb_prediction(bundle, cache, args, start_idx, sku_indices, weights, stat_pred)
        else:
            xgb_pred = stat_pred[:, sku_indices]

        for alpha in alphas:
            scale_grid = [1.0] if alpha == 0.0 else xgb_scales
            for xgb_scale in scale_grid:
                blended = blend_top_sku_predictions(
                    stat_pred,
                    xgb_pred,
                    sku_indices,
                    alpha,
                    xgb_scale=xgb_scale,
                )
                for validation_scale in validation_scales:
                    for evaluation_scale in evaluation_scales:
                        pred = apply_block_scales(blended, validation_scale, evaluation_scale)
                        scores = score_prediction(actual, pred, scale, weights)
                        if alpha == 0:
                            candidate = f"stats_v{validation_scale:.2f}_e{evaluation_scale:.2f}"
                        else:
                            candidate = (
                                f"blend_a{alpha:.2f}_s{xgb_scale:.2f}"
                                f"_v{validation_scale:.2f}_e{evaluation_scale:.2f}"
                            )
                        rows.append(
                            {
                                "fold_start": str(fold_start.date()),
                                "candidate": candidate,
                                "alpha": alpha,
                                "xgb_scale": xgb_scale,
                                "validation_scale": validation_scale,
                                "evaluation_scale": evaluation_scale,
                                **scores,
                                "pred_sum_56": float(pred.sum()),
                                "pred_nonzero": int(np.count_nonzero(pred > 1e-9)),
                                "top_n": len(sku_indices),
                            }
                        )
    return pd.DataFrame(rows)


def fit_fold_xgb_prediction(
    bundle: DatasetBundle,
    cache,
    args: argparse.Namespace,
    validation_start_idx: int,
    sku_indices: np.ndarray,
    weights: np.ndarray,
    validation_stat_pred: np.ndarray,
) -> np.ndarray:
    train_frames = []
    max_train_start = validation_start_idx - HORIZON
    starts = make_training_starts(
        min_start_idx=args.min_history_days,
        max_start_idx=max_train_start,
        stride_days=args.train_stride_days,
    )
    for train_start_idx in starts:
        cutoff_idx = train_start_idx - 1
        train_weights = profit_weights(bundle.train, bundle.dates[cutoff_idx], bundle.sku_order)
        forecast_dates = bundle.dates[train_start_idx : train_start_idx + HORIZON]
        stat_pred = make_stat_prediction(bundle.quantity[:train_start_idx], bundle.dates[:train_start_idx], forecast_dates, args)
        frame = make_direct_feature_frame(
            bundle,
            cutoff_idx=cutoff_idx,
            horizons=range(1, HORIZON + 1),
            sku_indices=sku_indices,
            weights=train_weights,
            stat_pred=stat_pred,
            cache=cache,
        )
        train_frames.append(attach_targets(frame, bundle, target_start_idx=train_start_idx))
    if not train_frames:
        return validation_stat_pred[:, sku_indices]

    train_frame = pd.concat(train_frames, ignore_index=True)
    fitted = fit_direct_xgb(
        train_frame,
        objective=args.objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        public_boost=args.public_boost,
        tweedie_power=args.tweedie_power,
        min_child_weight_val=args.min_child_weight,
        recency_halflife=args.recency_halflife,
    )
    val_frame = make_direct_feature_frame(
        bundle,
        cutoff_idx=validation_start_idx - 1,
        horizons=range(1, HORIZON + 1),
        sku_indices=sku_indices,
        weights=weights,
        stat_pred=validation_stat_pred,
        cache=cache,
    )
    pred_ml = predict_direct_xgb(fitted, val_frame).reshape(HORIZON, len(sku_indices))
    if args.volume_matching:
        pred_ml = apply_volume_matching(validation_stat_pred, pred_ml, sku_indices)
    return pred_ml



def make_final_prediction(
    bundle: DatasetBundle,
    cache,
    args: argparse.Namespace,
    alpha: float,
    xgb_scale: float,
    validation_scale: float,
    evaluation_scale: float,
) -> np.ndarray:
    final_start_idx = len(bundle.dates)
    cutoff_idx = final_start_idx - 1
    forecast_dates = pd.date_range(bundle.dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)
    sku_indices = top_profit_indices(weights, args.top_n)
    stat_pred = make_stat_prediction(bundle.quantity, bundle.dates, forecast_dates, args)
    if alpha <= 0.0:
        return apply_block_scales(stat_pred, validation_scale, evaluation_scale)

    train_frames = []
    max_train_start = final_start_idx - HORIZON
    starts = make_training_starts(
        min_start_idx=args.min_history_days,
        max_start_idx=max_train_start,
        stride_days=args.train_stride_days,
    )
    for train_start_idx in starts:
        train_weights = profit_weights(bundle.train, bundle.dates[train_start_idx - 1], bundle.sku_order)
        train_forecast_dates = bundle.dates[train_start_idx : train_start_idx + HORIZON]
        train_stat = make_stat_prediction(bundle.quantity[:train_start_idx], bundle.dates[:train_start_idx], train_forecast_dates, args)
        frame = make_direct_feature_frame(
            bundle,
            cutoff_idx=train_start_idx - 1,
            horizons=range(1, HORIZON + 1),
            sku_indices=sku_indices,
            weights=train_weights,
            stat_pred=train_stat,
            cache=cache,
        )
        train_frames.append(attach_targets(frame, bundle, target_start_idx=train_start_idx))
    if not train_frames:
        return stat_pred

    fitted = fit_direct_xgb(
        pd.concat(train_frames, ignore_index=True),
        objective=args.objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        public_boost=args.public_boost,
        tweedie_power=args.tweedie_power,
        min_child_weight_val=args.min_child_weight,
        recency_halflife=args.recency_halflife,
    )
    final_frame = make_direct_feature_frame(
        bundle,
        cutoff_idx=cutoff_idx,
        horizons=range(1, HORIZON + 1),
        sku_indices=sku_indices,
        weights=weights,
        stat_pred=stat_pred,
        cache=cache,
    )
    xgb_pred = predict_direct_xgb(fitted, final_frame).reshape(HORIZON, len(sku_indices))
    if args.volume_matching:
        xgb_pred = apply_volume_matching(stat_pred, xgb_pred, sku_indices)
    blended = blend_top_sku_predictions(stat_pred, xgb_pred, sku_indices, alpha, xgb_scale=xgb_scale)
    return apply_block_scales(blended, validation_scale, evaluation_scale)



def apply_private_calibration(
    pred: np.ndarray,
    bundle: DatasetBundle,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    if args.private_calibration == "none":
        return pred, {"mode": "none"}, pd.DataFrame()

    years = parse_int_list(args.private_ratio_years)
    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)
    # Train-only proxy for the 92 high-impact SKUs: no prior submission file is read here.
    selected, score = source_only_private_sku_mask(bundle, weights, args)
    ratio_table = historical_private_public_ratio_table(bundle, years)

    if args.private_calibration == "global_ratio":
        if args.private_global_target_ratio <= 0:
            raise ValueError("--private-global-target-ratio must be positive")
        target_ratio = np.full(len(bundle.sku_order), float(args.private_global_target_ratio), dtype=np.float64)
    elif args.private_calibration == "sku_ratio":
        hist = ratio_table["ratio_median"].to_numpy(dtype=np.float64)
        hist = np.where(np.isfinite(hist), hist, args.private_global_prior)
        hist = np.clip(hist, args.private_ratio_clip_low, args.private_ratio_clip_high)
        target_ratio = np.exp(
            (1.0 - args.private_hist_weight) * np.log(args.private_global_prior)
            + args.private_hist_weight * np.log(hist)
        )
    else:
        raise ValueError(f"Unsupported private calibration mode: {args.private_calibration}")
    target_ratio = np.where(selected, target_ratio, 1.0)

    out = pred.astype(np.float64, copy=True)
    eval_before = out[BLOCK:].copy()
    public_totals = out[:BLOCK].sum(axis=0)
    eval_totals_before = eval_before.sum(axis=0)
    target_totals = np.maximum(public_totals * target_ratio, 0.0)
    applied_scale = np.ones(out.shape[1], dtype=np.float64)
    eval_dates = pd.date_range(bundle.dates[-1] + pd.Timedelta(days=BLOCK + 1), periods=BLOCK, freq="D")
    non_sunday = np.asarray(eval_dates.dayofweek < 6)

    for sku_idx in np.flatnonzero(selected):
        current = float(eval_totals_before[sku_idx])
        target = float(target_totals[sku_idx])
        if current > 1e-12:
            scale = target / current
            out[BLOCK:, sku_idx] = np.maximum(out[BLOCK:, sku_idx] * scale, 0.0)
            applied_scale[sku_idx] = scale
        elif target > 0.0:
            out[BLOCK:, sku_idx] = 0.0
            fill_days = non_sunday if bool(non_sunday.any()) else np.ones(BLOCK, dtype=bool)
            out[BLOCK:, sku_idx][fill_days] = target / float(fill_days.sum())
            applied_scale[sku_idx] = np.inf

    public_weighted = weighted_block_sum(out[:BLOCK], weights, selected)
    eval_before_weighted = weighted_block_sum(eval_before, weights, selected)
    eval_after_weighted = weighted_block_sum(out[BLOCK:], weights, selected)
    diagnostics = {
        "mode": args.private_calibration,
        "selected_count": int(selected.sum()),
        "pool_top_n": int(args.private_calibration_pool_top_n),
        "score": args.private_calibration_score,
        "active_min_days": int(args.private_active_min_days),
        "years": years,
        "selected_weight_share": float(weights[selected].sum()),
        "selected_public_weighted_sum": public_weighted,
        "selected_eval_weighted_sum_before": eval_before_weighted,
        "selected_eval_weighted_sum_after": eval_after_weighted,
        "selected_eval_vs_public_ratio_after": float(eval_after_weighted / max(public_weighted, 1e-12)),
        "evaluation_total_sum_before": float(eval_before.sum()),
        "evaluation_total_sum_after": float(out[BLOCK:].sum()),
    }
    if args.private_calibration == "global_ratio":
        diagnostics["global_target_ratio"] = float(args.private_global_target_ratio)
    if args.private_calibration == "sku_ratio":
        diagnostics["global_prior"] = float(args.private_global_prior)
        diagnostics["hist_weight"] = float(args.private_hist_weight)
        diagnostics["clip_low"] = float(args.private_ratio_clip_low)
        diagnostics["clip_high"] = float(args.private_ratio_clip_high)

    audit = ratio_table.copy()
    audit["selected_private_calibration"] = selected
    audit["selection_score"] = score
    audit["profit_weight"] = weights
    audit["public_total"] = public_totals
    audit["eval_total_before"] = eval_totals_before
    audit["target_ratio"] = target_ratio
    audit["target_eval_total"] = target_totals
    audit["applied_eval_scale"] = applied_scale
    return out.astype(np.float32), diagnostics, audit


def source_only_private_sku_mask(
    bundle: DatasetBundle,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pool_n = max(1, min(int(args.private_calibration_pool_top_n), len(weights)))
    top_k = max(1, min(int(args.private_calibration_top_k), pool_n))
    history = np.maximum(bundle.quantity, 0.0)
    if args.private_calibration_score == "active_weight":
        # The best source-only proxy for the public-winning SKU cluster was
        # "high profit weight and still selling recently", not raw recent volume.
        recent28 = history[-min(28, history.shape[0]) :]
        active28 = (recent28 > 0.0).sum(axis=0)
        score = weights.copy()
        score[active28 < int(args.private_active_min_days)] = -1.0
    else:
        window = {
            "weighted_last28": 28,
            "weighted_last56": 56,
            "weighted_last112": 112,
        }[args.private_calibration_score]
        recent = history[-min(window, history.shape[0]) :].sum(axis=0)
        score = recent * weights
    pool = np.argsort(-weights, kind="mergesort")[:pool_n]
    pool_order = np.lexsort((pool, -score[pool]))
    selected_idx = pool[pool_order[:top_k]]
    selected = np.zeros(len(weights), dtype=bool)
    selected[selected_idx] = True
    return selected, score


def historical_private_public_ratio_table(bundle: DatasetBundle, years: list[int]) -> pd.DataFrame:
    ratio_cols: dict[str, np.ndarray] = {}
    ratios = []
    for year in years:
        public_mask = (
            (bundle.dates >= pd.Timestamp(f"{year}-09-06"))
            & (bundle.dates <= pd.Timestamp(f"{year}-10-03"))
        )
        private_mask = (
            (bundle.dates >= pd.Timestamp(f"{year}-10-04"))
            & (bundle.dates <= pd.Timestamp(f"{year}-10-31"))
        )
        public_total = bundle.quantity[public_mask].sum(axis=0, dtype=np.float64)
        private_total = bundle.quantity[private_mask].sum(axis=0, dtype=np.float64)
        ratio = np.divide(
            private_total,
            public_total,
            out=np.full_like(private_total, np.nan, dtype=np.float64),
            where=public_total > 1e-12,
        )
        ratio_cols[f"public_{year}"] = public_total
        ratio_cols[f"private_{year}"] = private_total
        ratio_cols[f"ratio_{year}"] = ratio
        ratios.append(ratio)

    ratio_matrix = np.vstack(ratios) if ratios else np.empty((0, len(bundle.sku_order)))
    finite = np.isfinite(ratio_matrix)
    counts = finite.sum(axis=0) if ratios else np.zeros(len(bundle.sku_order), dtype=int)
    ratio_median = np.full(len(bundle.sku_order), np.nan, dtype=np.float64)
    ratio_q25 = np.full(len(bundle.sku_order), np.nan, dtype=np.float64)
    ratio_q75 = np.full(len(bundle.sku_order), np.nan, dtype=np.float64)
    for col in np.flatnonzero(counts > 0):
        values = ratio_matrix[finite[:, col], col]
        ratio_median[col] = float(np.median(values))
        ratio_q25[col] = float(np.quantile(values, 0.25))
        ratio_q75[col] = float(np.quantile(values, 0.75))

    return pd.DataFrame(
        {
            "sku": bundle.sku_order,
            "sku_idx": np.arange(len(bundle.sku_order), dtype=int),
            **ratio_cols,
            "ratio_median": ratio_median,
            "ratio_q25": ratio_q25,
            "ratio_q75": ratio_q75,
            "ratio_count": counts,
        }
    )


def weighted_block_sum(block: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return float((block[:, mask] * weights[mask][None, :]).sum())


def score_prediction(
    actual: np.ndarray,
    pred: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    h56, _ = wrmsse(actual, pred, scale, weights)
    h28, _ = wrmsse(actual[:BLOCK], pred[:BLOCK], scale, weights)
    h29, _ = wrmsse(actual[BLOCK:], pred[BLOCK:], scale, weights)
    return {
        "h1_28_wrmsse": h28,
        "h29_56_wrmsse": h29,
        "h1_56_wrmsse": h56,
    }


def summarize_cv(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    group_cols = ["candidate", "alpha", "xgb_scale", "validation_scale", "evaluation_scale"]
    summary = detail.groupby(group_cols, as_index=False).agg(
        h1_28_wrmsse=("h1_28_wrmsse", "mean"),
        h29_56_wrmsse=("h29_56_wrmsse", "mean"),
        h1_56_wrmsse=("h1_56_wrmsse", "mean"),
        pred_sum_56=("pred_sum_56", "mean"),
        pred_nonzero=("pred_nonzero", "mean"),
    )
    fold_order = [fold for fold in FOLD_STARTS if fold in set(detail["fold_start"])]
    weights = recent_fold_weights(fold_order)
    weighted = detail.copy()
    weighted["fold_weight"] = weighted["fold_start"].map(weights).fillna(0.0)
    for col in ("h1_28_wrmsse", "h29_56_wrmsse", "h1_56_wrmsse"):
        weighted[f"rw_{col}"] = weighted[col] * weighted["fold_weight"]
    recent = weighted.groupby(group_cols, as_index=False).agg(
        recent_weighted_h1_28=("rw_h1_28_wrmsse", "sum"),
        recent_weighted_h29_56=("rw_h29_56_wrmsse", "sum"),
        recent_weighted_h1_56=("rw_h1_56_wrmsse", "sum"),
    )
    summary = summary.merge(recent, on=group_cols, how="left")
    return summary.sort_values(["recent_weighted_h1_56", "h1_56_wrmsse"]).reset_index(drop=True)


def recent_fold_weights(fold_starts: list[str]) -> dict[str, float]:
    if not fold_starts:
        return {}
    raw = np.array([0.50, 0.30, 0.20, 0.12, 0.08], dtype=np.float64)[: len(fold_starts)]
    raw = raw / raw.sum()
    return {fold: float(weight) for fold, weight in zip(fold_starts, raw)}


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("alpha grid cannot be empty.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("integer list cannot be empty.")
    return values


if __name__ == "__main__":
    main()
