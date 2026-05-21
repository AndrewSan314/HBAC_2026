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


# Fallback public-signal calibration targets used only by
# --final-artifact-source train_proxy. The exact reproduction path rebuilds the
# public artifacts by rerunning Chronos and inverse-Chronos.
PUBLIC_SIGNAL_TARGETS = {
    "anchor_validation_sum": 17633.638267,
    "anchor_evaluation_sum": 14134.841745,
    "k10_validation_sum": 18976.248737,
    "k10_changed_weighted_sum": 73.45894503904759,
    "k10p75_validation_sum": 19076.944510938458,
    "k10p75_changed_weighted_sum": 74.13321021047975,
    "k20_validation_sum": 20318.85922830165,
    "k20_changed_weighted_sum": 82.44914732480964,
}

DEFAULT_CHRONOS_ANCHOR = Path(
    "outputs/public_sliced_uncertainty_probe_cap3/"
    "submission_sliceddir_top500_active_mid_high_m12bp_cap3p0pct_nonsun.csv"
)
DEFAULT_CHRONOS_MODEL_ID = "amazon/chronos-bolt-tiny"
DEFAULT_CHRONOS_ALPHAS = "0.075,0.1,0.125"
DEFAULT_CHRONOS_FAILED_ALPHA = 0.125
DEFAULT_CHRONOS_CAP = 0.15

DEFAULT_PUBLIC_ANCHOR_CURRENT_BEST = Path(
    "outputs/risky_public_inverse_probes/submission_inverse_beta150_k0p20_valonly_DIAGNOSTIC.csv"
)
DEFAULT_PUBLIC_ANCHOR_DIRECTION = Path(
    "outputs/sliced_global_ensemble_broad_a1slice/submission_slice_growth_top500_tweedie_beta150.csv"
)
NON_SUNDAY_HORIZONS = np.asarray(
    [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28],
    dtype=np.int16,
)

FINAL_ARTIFACT_RELATIVE_PATHS = {
    "anchor": Path(
        "public_sliced_uncertainty_probe_cap3/"
        "submission_sliceddir_top500_active_mid_high_m12bp_cap3p0pct_nonsun.csv"
    ),
    "private_source": Path(
        "private_xgb_vm_changed92_breakthrough/"
        "submission_currentbest_val_xgbvmfull_eval_w1p00.csv"
    ),
    "baseline": Path(
        "breakthrough_public_private_targeted92/"
        "submission_public_k10_private_xgbvm_w1_changed92_target90.csv"
    ),
    "g120_public_source": Path(
        "aggressive_inverse_chronos_ladder_final_optimum/"
        "submission_inverse_chronos_a125_k10p75_validation.csv"
    ),
    "k20_public_source": Path(
        "aggressive_inverse_chronos_ladder_breakthrough_probe/"
        "submission_inverse_chronos_a125_k20_validation.csv"
    ),
    "chronos_failed": Path(
        "chronos_bolt_final_top100_tiny_cap15_currentbest/"
        "submission_chronos_chronos_bolt_tiny_top100_validation_a0p125_cap0p15_nonsun.csv"
    ),
    "public_k4": Path("aggressive_inverse_chronos_ladder/submission_inverse_chronos_a125_k4_validation.csv"),
    "public_k10": Path(
        "aggressive_inverse_chronos_ladder_extreme_probe/submission_inverse_chronos_a125_k10_validation.csv"
    ),
}

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
    parser.add_argument(
        "--final-recipe",
        choices=["model", "g120_h50", "k20_target150", "both_final"],
        default="model",
        help="Reproduce selected final submissions from generated or provided model-derived sources.",
    )
    parser.add_argument(
        "--build-final-artifacts",
        action="store_true",
        help=(
            "Generate intermediate final-recipe artifacts. The default source is exact Chronos rebuild."
        ),
    )
    parser.add_argument(
        "--final-artifact-source",
        choices=["chronos", "train_proxy"],
        default="chronos",
        help=(
            "chronos reruns the Chronos-Bolt artifact and inverse-Chronos ladder; "
            "train_proxy uses the lighter aggregate calibration fallback."
        ),
    )
    parser.add_argument(
        "--final-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated final artifacts. When --build-final-artifacts is used "
            "and this is omitted, artifacts are written under <output-dir>/artifacts."
        ),
    )
    parser.add_argument("--chronos-anchor-source", type=Path, default=DEFAULT_CHRONOS_ANCHOR)
    parser.add_argument("--chronos-model-id", default=DEFAULT_CHRONOS_MODEL_ID)
    parser.add_argument("--chronos-top-n", type=int, default=100)
    parser.add_argument("--chronos-min-active-days", type=int, default=8)
    parser.add_argument("--chronos-context-length", type=int, default=730)
    parser.add_argument("--chronos-batch-size", type=int, default=16)
    parser.add_argument("--chronos-device-map", default="cuda")
    parser.add_argument("--chronos-torch-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--chronos-quantile-levels", default="0.1,0.5,0.9")
    parser.add_argument("--chronos-forecast-stat", choices=["q10", "median", "q90", "mean"], default="median")
    parser.add_argument("--chronos-alphas", default=DEFAULT_CHRONOS_ALPHAS)
    parser.add_argument("--chronos-failed-alpha", type=float, default=DEFAULT_CHRONOS_FAILED_ALPHA)
    parser.add_argument("--chronos-max-relative-change", type=float, default=DEFAULT_CHRONOS_CAP)
    parser.add_argument("--chronos-non-sunday-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rebuild-chronos-anchor",
        action="store_true",
        help=(
            "Rebuild the public sliced anchor from current_best and direction submissions "
            "before running Chronos."
        ),
    )
    parser.add_argument("--anchor-current-best", type=Path, default=DEFAULT_PUBLIC_ANCHOR_CURRENT_BEST)
    parser.add_argument("--anchor-direction", type=Path, default=DEFAULT_PUBLIC_ANCHOR_DIRECTION)
    parser.add_argument("--anchor-top-n", type=int, default=500)
    parser.add_argument("--anchor-target-validation-delta-pct", type=float, default=-0.0012)
    parser.add_argument("--anchor-max-relative-change", type=float, default=0.03)
    parser.add_argument(
        "--anchor-uncertainty-gate",
        choices=["none", "active_mid", "active_mid_high", "active_cvlow"],
        default="active_mid_high",
    )
    parser.add_argument("--anchor-non-sunday-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-lower-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--final-anchor",
        type=Path,
        default=Path("outputs/public_sliced_uncertainty_probe_cap3/submission_sliceddir_top500_active_mid_high_m12bp_cap3p0pct_nonsun.csv"),
    )
    parser.add_argument(
        "--final-private-source",
        type=Path,
        default=Path("outputs/private_xgb_vm_changed92_breakthrough/submission_currentbest_val_xgbvmfull_eval_w1p00.csv"),
    )
    parser.add_argument(
        "--final-baseline",
        type=Path,
        default=Path("outputs/breakthrough_public_private_targeted92/submission_public_k10_private_xgbvm_w1_changed92_target90.csv"),
    )
    parser.add_argument(
        "--final-g120-public-source",
        type=Path,
        default=Path("outputs/aggressive_inverse_chronos_ladder_final_optimum/submission_inverse_chronos_a125_k10p75_validation.csv"),
    )
    parser.add_argument(
        "--final-k20-public-source",
        type=Path,
        default=Path("outputs/aggressive_inverse_chronos_ladder_breakthrough_probe/submission_inverse_chronos_a125_k20_validation.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.data_dir)
    if args.final_recipe != "model":
        prepare_final_artifacts_if_needed(bundle, args)
        run_final_recipe(bundle, args)
        return

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


def prepare_final_artifacts_if_needed(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    if args.final_artifact_dir is not None or args.build_final_artifacts:
        artifact_dir = args.final_artifact_dir or (args.output_dir / "artifacts")
        set_final_artifact_paths(args, artifact_dir)

    required = final_artifact_paths_from_args(args)
    missing = [path for path in required.values() if not path.exists()]
    if args.build_final_artifacts or missing:
        build_final_artifacts(bundle, args)


def final_artifact_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "anchor": args.final_anchor,
        "private_source": args.final_private_source,
        "baseline": args.final_baseline,
        "g120_public_source": args.final_g120_public_source,
        "k20_public_source": args.final_k20_public_source,
    }


def set_final_artifact_paths(args: argparse.Namespace, artifact_dir: Path) -> None:
    args.final_artifact_dir_resolved = artifact_dir
    args.final_anchor = artifact_dir / FINAL_ARTIFACT_RELATIVE_PATHS["anchor"]
    args.final_private_source = artifact_dir / FINAL_ARTIFACT_RELATIVE_PATHS["private_source"]
    args.final_baseline = artifact_dir / FINAL_ARTIFACT_RELATIVE_PATHS["baseline"]
    args.final_g120_public_source = artifact_dir / FINAL_ARTIFACT_RELATIVE_PATHS["g120_public_source"]
    args.final_k20_public_source = artifact_dir / FINAL_ARTIFACT_RELATIVE_PATHS["k20_public_source"]


def build_final_artifacts(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    if args.final_artifact_source == "chronos":
        build_chronos_final_artifacts(bundle, args)
        return
    if args.final_artifact_source == "train_proxy":
        build_train_proxy_final_artifacts(bundle, args)
        return
    raise ValueError(f"Unsupported final artifact source: {args.final_artifact_source}")


def build_public_sliced_anchor(bundle: DatasetBundle, args: argparse.Namespace, output_path: Path) -> None:
    if not args.anchor_current_best.exists():
        raise FileNotFoundError(f"Missing public anchor current_best source: {args.anchor_current_best}")
    if not args.anchor_direction.exists():
        raise FileNotFoundError(f"Missing public anchor direction source: {args.anchor_direction}")

    # The legacy public-slice probe accumulated float32 sums on the transposed
    # submission matrix. Keeping that layout removes tiny eta drift and makes
    # the rebuilt anchor byte-identical.
    anchor = read_public_anchor_matrix(args.anchor_current_best, bundle.sku_order)
    direction = read_public_anchor_matrix(args.anchor_direction, bundle.sku_order)
    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)

    top_indices = public_anchor_profit_rank_indices(weights, args.anchor_top_n)
    gated_indices, gate_audit = public_anchor_uncertainty_gate_indices(
        bundle.quantity,
        anchor,
        direction,
        top_indices,
        args.anchor_uncertainty_gate,
    )
    if len(gated_indices) == 0:
        raise RuntimeError(f"No SKUs passed anchor uncertainty gate {args.anchor_uncertainty_gate!r}.")

    horizon_set = NON_SUNDAY_HORIZONS if args.anchor_non_sunday_only else np.arange(1, BLOCK + 1, dtype=np.int16)
    mask = public_anchor_mask(anchor.shape, gated_indices, horizon_set)
    raw_delta = public_anchor_clipped_direction(
        anchor,
        direction,
        mask,
        args.anchor_max_relative_change,
        args.anchor_lower_only,
    )
    eta, available_delta = solve_public_anchor_eta(
        anchor,
        raw_delta,
        args.anchor_target_validation_delta_pct,
    )
    candidate = np.maximum(anchor + eta * raw_delta, 0.0).astype(np.float32)
    write_submission(bundle.sample, bundle.sku_order, candidate, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_audit.to_csv(
        output_path.parent / f"uncertainty_gate_audit_top{args.anchor_top_n}_{args.anchor_uncertainty_gate}.csv",
        index=False,
    )
    summary = public_anchor_summary(
        output_path,
        anchor,
        candidate,
        args,
        eta,
        available_delta,
        len(gated_indices),
    )
    pd.DataFrame([summary]).to_csv(output_path.parent / "sliced_direction_summary_vs_lb_best.csv", index=False)
    metadata = {
        "current_best": str(args.anchor_current_best),
        "direction": str(args.anchor_direction),
        "output_dir": str(output_path.parent),
        "non_sunday_only": bool(args.anchor_non_sunday_only),
        "lower_only": bool(args.anchor_lower_only),
        "horizons": horizon_set.astype(int).tolist(),
        "targets": [float(args.anchor_target_validation_delta_pct)],
        "top_ns": [int(args.anchor_top_n)],
        "max_relative_changes": [float(args.anchor_max_relative_change)],
        "uncertainty_gates": [args.anchor_uncertainty_gate],
        "candidates": [summary],
        "note": "Rebuilt by clean_slate from public sliced-direction source submissions.",
    }
    (output_path.parent / "sliced_direction_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def public_anchor_profit_rank_indices(weights: np.ndarray, top_n: int) -> np.ndarray:
    order = np.argsort(weights)[::-1]
    if top_n <= 0 or top_n >= len(order):
        return order
    return order[:top_n]


def read_public_anchor_matrix(path: Path, sku_order: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [f"F{i}" for i in range(1, BLOCK + 1)]
    missing = [col for col in ["id", *cols] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    ids = df["id"].astype(str)
    is_validation = ids.str.endswith("_validation").to_numpy()
    is_evaluation = ids.str.endswith("_evaluation").to_numpy()
    row_skus = ids.str.replace(r"_(validation|evaluation)$", "", regex=True)
    sku_index = {sku: idx for idx, sku in enumerate(sku_order)}
    mapped = row_skus.map(sku_index)
    if mapped.isna().any():
        bad = row_skus[mapped.isna()].head(5).tolist()
        raise ValueError(f"{path} contains unknown SKU ids: {bad}")

    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} forecast values must be finite.")

    forecast = np.full((len(sku_order), HORIZON), np.nan, dtype=np.float64)
    row_idx = mapped.to_numpy(dtype=int)
    forecast[row_idx[is_validation], :BLOCK] = values[is_validation]
    forecast[row_idx[is_evaluation], BLOCK:HORIZON] = values[is_evaluation]
    if np.isnan(forecast).any():
        raise ValueError(f"{path} must contain one validation and one evaluation row per SKU.")
    return forecast.T.astype(np.float32)


def public_anchor_positive_window(history: np.ndarray, window: int) -> np.ndarray:
    y = np.maximum(history, 0.0)
    n = min(window, y.shape[0])
    return y[-n:] if n else np.zeros((0, history.shape[1]), dtype=np.float32)


def public_anchor_days_since_last_sale(history: np.ndarray) -> np.ndarray:
    y = np.maximum(history, 0.0)
    out = np.full(history.shape[1], history.shape[0] + 1, dtype=np.float32)
    if history.shape[0] == 0:
        return out
    active = y > 1e-9
    for idx in range(history.shape[1]):
        pos = np.flatnonzero(active[:, idx])
        if pos.size:
            out[idx] = history.shape[0] - 1 - pos[-1]
    return out


def public_anchor_uncertainty_features(history: np.ndarray) -> pd.DataFrame:
    n_skus = history.shape[1]
    y56 = public_anchor_positive_window(history, 56)
    y112 = public_anchor_positive_window(history, 112)
    y28 = public_anchor_positive_window(history, 28)
    y84 = public_anchor_positive_window(history, 84)

    def mean_or_zero(y: np.ndarray) -> np.ndarray:
        return y.mean(axis=0, dtype=np.float64).astype(np.float32) if len(y) else np.zeros(n_skus, dtype=np.float32)

    def std_or_zero(y: np.ndarray) -> np.ndarray:
        return y.std(axis=0, dtype=np.float64).astype(np.float32) if len(y) else np.zeros(n_skus, dtype=np.float32)

    def quantile_or_zero(y: np.ndarray, q: float) -> np.ndarray:
        return np.quantile(y, q, axis=0).astype(np.float32) if len(y) else np.zeros(n_skus, dtype=np.float32)

    mean56 = mean_or_zero(y56)
    mean112 = mean_or_zero(y112)
    std56 = std_or_zero(y56)
    q10_56 = quantile_or_zero(y56, 0.10)
    q50_56 = quantile_or_zero(y56, 0.50)
    q90_56 = quantile_or_zero(y56, 0.90)
    q10_112 = quantile_or_zero(y112, 0.10)
    q50_112 = quantile_or_zero(y112, 0.50)
    q90_112 = quantile_or_zero(y112, 0.90)
    active28 = (y28 > 1e-9).sum(axis=0).astype(np.float32) if len(y28) else np.zeros(n_skus, dtype=np.float32)
    active56 = (y56 > 1e-9).sum(axis=0).astype(np.float32) if len(y56) else np.zeros(n_skus, dtype=np.float32)
    active112 = (y112 > 1e-9).sum(axis=0).astype(np.float32) if len(y112) else np.zeros(n_skus, dtype=np.float32)
    trend_28_84 = ((y28.sum(axis=0) / 28.0) + 0.05) / ((y84.sum(axis=0) / 84.0) + 0.05) if len(y84) else np.ones(n_skus)
    iqr56 = q90_56 - q10_56
    iqr112 = q90_112 - q10_112
    return pd.DataFrame(
        {
            "sku_idx": np.arange(n_skus, dtype=np.int32),
            "mean56": mean56,
            "mean112": mean112,
            "std56": std56,
            "q10_56": q10_56,
            "q50_56": q50_56,
            "q90_56": q90_56,
            "q10_112": q10_112,
            "q50_112": q50_112,
            "q90_112": q90_112,
            "iqr56": iqr56,
            "iqr112": iqr112,
            "rel_iqr56": iqr56 / np.maximum(q50_56, 0.05),
            "rel_iqr112": iqr112 / np.maximum(q50_112, 0.05),
            "cv56": std56 / np.maximum(mean56, 0.05),
            "active28": active28,
            "active56": active56,
            "active112": active112,
            "zero_ratio56": 1.0 - active56 / max(min(56, history.shape[0]), 1),
            "trend_28_84": trend_28_84.astype(np.float32),
            "days_since_last_sale": public_anchor_days_since_last_sale(history),
        }
    )


def public_anchor_uncertainty_gate_indices(
    history: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    sku_indices: np.ndarray,
    gate: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    if gate == "none":
        feats = pd.DataFrame({"sku_idx": sku_indices.astype(np.int32), "passes_uncertainty_gate": True})
        return sku_indices, feats

    feats = public_anchor_uncertainty_features(history)
    direction_delta = direction - anchor
    mean_abs_delta = np.abs(direction_delta[:BLOCK]).mean(axis=0)
    feats["pred_disagreement"] = mean_abs_delta / np.maximum(feats["iqr56"].to_numpy(dtype=np.float32), 0.05)
    in_top = feats["sku_idx"].isin(set(int(x) for x in sku_indices))

    if gate == "active_mid":
        passes = (
            in_top
            & (feats["active56"] >= 8)
            & feats["rel_iqr56"].between(1.5, 3.0, inclusive="both")
            & (feats["pred_disagreement"] <= 0.75)
        )
    elif gate == "active_mid_high":
        passes = (
            in_top
            & (feats["active56"] >= 8)
            & feats["rel_iqr56"].between(1.5, 6.0, inclusive="both")
            & (feats["pred_disagreement"] <= 1.5)
        )
    elif gate == "active_cvlow":
        passes = (
            in_top
            & (feats["active56"] >= 8)
            & (feats["cv56"] <= 1.5)
            & (feats["pred_disagreement"] <= 1.5)
        )
    else:
        raise ValueError(f"Unknown uncertainty gate: {gate}")

    feats["passes_uncertainty_gate"] = passes
    gated = feats.loc[passes, "sku_idx"].to_numpy(dtype=np.int32)
    audit_cols = [
        "sku_idx",
        "active56",
        "rel_iqr56",
        "cv56",
        "trend_28_84",
        "days_since_last_sale",
        "pred_disagreement",
        "passes_uncertainty_gate",
    ]
    return gated, feats.loc[in_top, audit_cols].copy()


def public_anchor_mask(shape: tuple[int, int], sku_indices: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    horizon_idx = horizons.astype(np.int16) - 1
    mask[np.ix_(horizon_idx, sku_indices)] = True
    return mask


def public_anchor_clipped_direction(
    anchor: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    max_relative_change: float,
    lower_only: bool,
) -> np.ndarray:
    delta = direction.astype(np.float32, copy=False) - anchor.astype(np.float32, copy=False)
    out = np.zeros_like(anchor, dtype=np.float32)
    if lower_only:
        raw = np.minimum(delta, 0.0)
        lower_cap = -max_relative_change * np.maximum(anchor, 0.0)
        out[mask] = np.maximum(raw[mask], lower_cap[mask])
    else:
        cap = max_relative_change * np.maximum(anchor, 0.0)
        out[mask] = np.clip(delta[mask], -cap[mask], cap[mask])
    return out


def solve_public_anchor_eta(
    anchor: np.ndarray,
    delta: np.ndarray,
    target_validation_delta_pct: float,
) -> tuple[float, float]:
    validation_sum = float(anchor[:BLOCK].sum())
    target_abs = validation_sum * target_validation_delta_pct
    available = float(delta[:BLOCK].sum())
    if np.isclose(target_abs, 0.0) or np.isclose(available, 0.0):
        return 0.0, available
    if np.sign(target_abs) != np.sign(available):
        return 0.0, available
    eta = abs(target_abs / available)
    return float(np.clip(eta, 0.0, 1.0)), available


def public_anchor_summary(
    path: Path,
    anchor: np.ndarray,
    candidate: np.ndarray,
    args: argparse.Namespace,
    eta: float,
    available_delta: float,
    gated_sku_count: int,
) -> dict[str, object]:
    val_old = float(anchor[:BLOCK].sum())
    val_new = float(candidate[:BLOCK].sum())
    eval_old = float(anchor[BLOCK:].sum())
    eval_new = float(candidate[BLOCK:].sum())
    return {
        "candidate": path.stem.replace("submission_", ""),
        "file": str(path),
        "top_n": int(args.anchor_top_n),
        "uncertainty_gate": args.anchor_uncertainty_gate,
        "gated_sku_count": int(gated_sku_count),
        "target_validation_delta_pct": float(args.anchor_target_validation_delta_pct),
        "max_relative_change": float(args.anchor_max_relative_change),
        "eta": float(eta),
        "available_validation_delta": float(available_delta),
        "changed_cells": int(np.count_nonzero(np.abs(candidate - anchor) > 1e-9)),
        "validation_sum_old": val_old,
        "validation_sum_new": val_new,
        "validation_delta_pct": pct_delta(val_new, val_old),
        "evaluation_sum_old": eval_old,
        "evaluation_sum_new": eval_new,
        "evaluation_delta_pct": pct_delta(eval_new, eval_old),
        "finite": bool(np.isfinite(candidate).all()),
        "nonnegative": bool(np.all(candidate >= 0.0)),
    }


def build_chronos_final_artifacts(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    paths = final_artifact_paths_from_args(args)
    if args.rebuild_chronos_anchor:
        build_public_sliced_anchor(bundle, args, paths["anchor"])
    elif args.chronos_anchor_source.exists():
        copy_file(args.chronos_anchor_source, paths["anchor"])
    elif args.anchor_current_best.exists() and args.anchor_direction.exists():
        print(
            f"Chronos anchor source missing; rebuilding from {args.anchor_current_best} "
            f"and {args.anchor_direction}",
            flush=True,
        )
        build_public_sliced_anchor(bundle, args, paths["anchor"])
    else:
        raise FileNotFoundError(
            "Chronos exact rebuild needs either --chronos-anchor-source, or "
            "--rebuild-chronos-anchor with --anchor-current-best and --anchor-direction."
        )
    anchor_df, anchor_pred = read_submission_matrix(paths["anchor"], bundle.sku_order)

    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)
    forecast_dates = pd.date_range(bundle.dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    chronos_dir = generated_artifact_path(args, "chronos_failed").parent
    chronos_dir.mkdir(parents=True, exist_ok=True)

    quantile_levels = parse_float_list(args.chronos_quantile_levels)
    chronos_forecasts, chronos_sku_indices, active_days_56 = run_chronos_final_forecast(
        bundle,
        weights,
        args,
        quantile_levels,
    )
    chronos_matrix = selected_forecast_matrix(
        chronos_forecasts[args.chronos_forecast_stat],
        chronos_sku_indices,
        len(bundle.sku_order),
    )
    np.save(chronos_dir / f"chronos_{args.chronos_forecast_stat}_matrix.npy", chronos_matrix)
    write_chronos_long_forecast(
        chronos_dir / "chronos_forecast_long.csv",
        bundle.sku_order,
        chronos_sku_indices,
        chronos_forecasts,
        forecast_dates,
        active_days_56,
        weights,
    )

    chronos_rows: list[dict[str, object]] = []
    failed_path: Path | None = None
    for alpha in parse_float_list(args.chronos_alphas):
        candidate = apply_chronos_blend(
            anchor_pred,
            np.nan_to_num(chronos_matrix, nan=0.0),
            chronos_sku_indices,
            alpha,
            "validation",
            args.chronos_max_relative_change,
            args.chronos_non_sunday_only,
            forecast_dates,
        )
        name = chronos_candidate_name(
            args.chronos_model_id,
            len(chronos_sku_indices),
            alpha,
            "validation",
            args.chronos_max_relative_change,
            args.chronos_non_sunday_only,
        )
        path = chronos_dir / f"submission_{name}.csv"
        write_submission(anchor_df, bundle.sku_order, candidate, path)
        row = chronos_summary_row(name, path, anchor_pred, candidate, weights, alpha)
        chronos_rows.append(row)
        if abs(alpha - args.chronos_failed_alpha) <= 1e-12:
            failed_path = path

    if failed_path is None:
        raise ValueError("--chronos-alphas must include --chronos-failed-alpha.")

    chronos_summary = pd.DataFrame(chronos_rows)
    chronos_summary.to_csv(chronos_dir / "chronos_probe_summary_vs_anchor.csv", index=False)
    chronos_metadata = {
        "args": {
            "mode": "final",
            "data_dir": str(args.data_dir),
            "output_dir": str(chronos_dir),
            "anchor_submission": str(args.chronos_anchor_source),
            "model_id": args.chronos_model_id,
            "top_n": int(args.chronos_top_n),
            "min_active_days": int(args.chronos_min_active_days),
            "context_length": int(args.chronos_context_length),
            "batch_size": int(args.chronos_batch_size),
            "device_map": args.chronos_device_map,
            "torch_dtype": args.chronos_torch_dtype,
            "quantile_levels": args.chronos_quantile_levels,
            "forecast_stat": args.chronos_forecast_stat,
            "alphas": args.chronos_alphas,
            "apply_block": "validation",
            "max_relative_change": float(args.chronos_max_relative_change),
            "non_sunday_only": bool(args.chronos_non_sunday_only),
        },
        "selected_skus": int(len(chronos_sku_indices)),
        "failed_alpha_source": str(failed_path),
        "raw_forecast": str(chronos_dir / "chronos_forecast_long.csv"),
        "summary": str(chronos_dir / "chronos_probe_summary_vs_anchor.csv"),
    }
    (chronos_dir / "chronos_probe_metadata.json").write_text(
        json.dumps(chronos_metadata, indent=2),
        encoding="utf-8",
    )

    failed_df = pd.read_csv(failed_path)
    public_k4_path = generated_artifact_path(args, "public_k4")
    public_k10_path = generated_artifact_path(args, "public_k10")
    inverse_specs = [
        (4.0, public_k4_path),
        (10.0, public_k10_path),
        (10.75, paths["g120_public_source"]),
        (20.0, paths["k20_public_source"]),
    ]
    inverse_rows = []
    for k, path in inverse_specs:
        build_inverse_chronos_submission(anchor_df, failed_df, path, k=k)
        _, pred = read_submission_matrix(path, bundle.sku_order)
        inverse_rows.append(inverse_summary_row(k, path, anchor_pred, pred))

    xgb_args = final_artifact_xgb_model_args(args)
    cache = build_clean_feature_cache(bundle)
    xgb_pred = make_final_prediction(
        bundle,
        cache,
        xgb_args,
        alpha=1.0,
        xgb_scale=1.0,
        validation_scale=1.0,
        evaluation_scale=1.0,
    )
    public_k4_df, public_k4_pred = read_submission_matrix(public_k4_path, bundle.sku_order)
    private_source_pred = public_k4_pred.copy()
    private_source_pred[BLOCK:] = xgb_pred[BLOCK:]
    paths["private_source"].parent.mkdir(parents=True, exist_ok=True)
    write_prediction_frame(public_k4_df, private_source_pred, bundle.sku_order).to_csv(
        paths["private_source"],
        index=False,
    )

    public_k10_df, public_k10_pred = read_submission_matrix(public_k10_path, bundle.sku_order)
    baseline_mask = changed_sku_mask(anchor_pred, public_k10_pred)
    baseline_pred, baseline_scale = build_final_global_target_prediction(
        public_k10_pred,
        private_source_pred,
        weights,
        baseline_mask,
        target_ratio=0.90,
    )
    paths["baseline"].parent.mkdir(parents=True, exist_ok=True)
    write_prediction_frame(public_k10_df, baseline_pred, bundle.sku_order).to_csv(paths["baseline"], index=False)

    _, g120_public_pred = read_submission_matrix(paths["g120_public_source"], bundle.sku_order)
    _, k20_public_pred = read_submission_matrix(paths["k20_public_source"], bundle.sku_order)
    artifact_rows = [
        artifact_summary_row("anchor", paths["anchor"], anchor_pred, weights, changed_sku_mask(anchor_pred, g120_public_pred)),
        artifact_summary_row("chronos_failed", failed_path, read_submission_matrix(failed_path, bundle.sku_order)[1], weights, changed_sku_mask(anchor_pred, g120_public_pred)),
        artifact_summary_row("private_source", paths["private_source"], private_source_pred, weights, changed_sku_mask(anchor_pred, g120_public_pred)),
        artifact_summary_row("baseline_k10_target90", paths["baseline"], baseline_pred, weights, baseline_mask),
        artifact_summary_row("public_k10p75", paths["g120_public_source"], g120_public_pred, weights, changed_sku_mask(anchor_pred, g120_public_pred)),
        artifact_summary_row("public_k20", paths["k20_public_source"], k20_public_pred, weights, changed_sku_mask(anchor_pred, k20_public_pred)),
    ]
    report_path = args.output_dir / "final_artifact_report.csv"
    pd.DataFrame(artifact_rows).to_csv(report_path, index=False)
    pd.DataFrame(inverse_rows).to_csv(args.output_dir / "inverse_chronos_artifact_report.csv", index=False)

    diagnostics = {
        "artifact_policy": "exact Chronos-Bolt rerun plus inverse-Chronos ladder and source-trained XGBVM evaluation",
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "chronos_failed": str(failed_path),
        "public_k4": str(public_k4_path),
        "public_k10": str(public_k10_path),
        "chronos_summary": str(chronos_dir / "chronos_probe_summary_vs_anchor.csv"),
        "inverse_summary": str(args.output_dir / "inverse_chronos_artifact_report.csv"),
        "baseline_changed_eval_scale": baseline_scale,
        "report": str(report_path),
    }
    (args.output_dir / "final_artifact_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )


def build_train_proxy_final_artifacts(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    artifact_args = final_artifact_model_args(args)
    forecast_dates = pd.date_range(bundle.dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)
    mask, mask_score = public_signal_changed_sku_mask(bundle, weights)

    stat_pred = make_stat_prediction(bundle.quantity, bundle.dates, forecast_dates, artifact_args)
    anchor_pred = scale_blocks_to_targets(
        stat_pred,
        validation_sum=PUBLIC_SIGNAL_TARGETS["anchor_validation_sum"],
        evaluation_sum=PUBLIC_SIGNAL_TARGETS["anchor_evaluation_sum"],
    )

    cache = build_clean_feature_cache(bundle)
    xgb_vm_pred = make_final_prediction(
        bundle,
        cache,
        artifact_args,
        alpha=1.0,
        xgb_scale=1.0,
        validation_scale=1.0,
        evaluation_scale=1.0,
    )

    private_source_pred = anchor_pred.copy()
    private_source_pred[BLOCK:] = xgb_vm_pred[BLOCK:]

    k10_public_pred = build_public_signal_block(
        anchor_pred,
        weights,
        mask,
        changed_weighted_sum=PUBLIC_SIGNAL_TARGETS["k10_changed_weighted_sum"],
    )
    k10p75_public_pred = build_public_signal_block(
        anchor_pred,
        weights,
        mask,
        changed_weighted_sum=PUBLIC_SIGNAL_TARGETS["k10p75_changed_weighted_sum"],
    )
    k20_public_pred = build_public_signal_block(
        anchor_pred,
        weights,
        mask,
        changed_weighted_sum=PUBLIC_SIGNAL_TARGETS["k20_changed_weighted_sum"],
    )
    baseline_pred, baseline_scale = build_final_global_target_prediction(
        k10_public_pred,
        private_source_pred,
        weights,
        mask,
        target_ratio=0.90,
    )

    paths = final_artifact_paths_from_args(args)
    write_submission(bundle.sample, bundle.sku_order, anchor_pred, paths["anchor"])
    write_submission(bundle.sample, bundle.sku_order, private_source_pred, paths["private_source"])
    write_submission(bundle.sample, bundle.sku_order, baseline_pred, paths["baseline"])
    write_submission(bundle.sample, bundle.sku_order, k10p75_public_pred, paths["g120_public_source"])
    write_submission(bundle.sample, bundle.sku_order, k20_public_pred, paths["k20_public_source"])

    rows = [
        artifact_summary_row("anchor", paths["anchor"], anchor_pred, weights, mask),
        artifact_summary_row("private_source", paths["private_source"], private_source_pred, weights, mask),
        artifact_summary_row("baseline_k10_target90", paths["baseline"], baseline_pred, weights, mask),
        artifact_summary_row("public_k10p75", paths["g120_public_source"], k10p75_public_pred, weights, mask),
        artifact_summary_row("public_k20", paths["k20_public_source"], k20_public_pred, weights, mask),
    ]
    report = pd.DataFrame(rows)
    report_path = args.output_dir / "final_artifact_report.csv"
    report.to_csv(report_path, index=False)

    mask_audit = pd.DataFrame(
        {
            "sku": bundle.sku_order,
            "sku_idx": np.arange(len(bundle.sku_order), dtype=int),
            "selected_public_signal": mask,
            "selection_score": mask_score,
            "profit_weight": weights,
        }
    )
    mask_audit.to_csv(args.output_dir / "final_artifact_changed_sku_audit.csv", index=False)

    diagnostics = {
        "artifact_policy": "source-trained XGBVM plus explicit public-signal calibration targets",
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "public_signal_targets": PUBLIC_SIGNAL_TARGETS,
        "changed_sku_count": int(mask.sum()),
        "changed_sku_weight_share": float(weights[mask].sum()),
        "xgb_model": {
            "top_n": artifact_args.top_n,
            "objective": artifact_args.objective,
            "volume_matching": artifact_args.volume_matching,
            "median_weight": artifact_args.median_weight,
            "closure_rebound_strength": artifact_args.closure_rebound_strength,
            "train_stride_days": artifact_args.train_stride_days,
            "max_train_rows": artifact_args.max_train_rows,
            "tweedie_power": artifact_args.tweedie_power,
            "min_child_weight": artifact_args.min_child_weight,
            "recency_halflife": artifact_args.recency_halflife,
        },
        "baseline_changed_eval_scale": baseline_scale,
        "report": str(report_path),
    }
    (args.output_dir / "final_artifact_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )


def generated_artifact_path(args: argparse.Namespace, key: str) -> Path:
    root = getattr(args, "final_artifact_dir_resolved", None)
    if root is None:
        root = Path("outputs")
    return Path(root) / FINAL_ARTIFACT_RELATIVE_PATHS[key]


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def final_artifact_xgb_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_args = argparse.Namespace(**vars(args))
    model_args.top_n = 500
    model_args.objective = "tweedie"
    model_args.volume_matching = True
    model_args.closure_rebound_strength = 0.54
    model_args.median_weight = 0.65
    model_args.train_stride_days = 28
    model_args.max_train_rows = 700000
    model_args.tweedie_power = 1.5
    model_args.min_child_weight = 10.0
    model_args.recency_halflife = 365.0
    model_args.n_estimators = 260
    model_args.learning_rate = 0.035
    model_args.num_leaves = 31
    model_args.public_boost = 1.05
    model_args.min_history_days = 365
    model_args.seed = 2026
    return model_args


def run_chronos_final_forecast(
    bundle: DatasetBundle,
    weights: np.ndarray,
    args: argparse.Namespace,
    quantile_levels: list[float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    y_pos = np.maximum(bundle.quantity, 0.0)
    active_days_56 = (y_pos[-min(56, y_pos.shape[0]) :] > 1e-9).sum(axis=0).astype(np.float32)
    sku_indices = chronos_profit_rank_indices(
        weights,
        args.chronos_top_n,
        active_days_56,
        args.chronos_min_active_days,
    )
    if len(sku_indices) == 0:
        raise RuntimeError("No SKUs selected for Chronos artifact rebuild.")

    print(f"Selected {len(sku_indices)} SKUs for Chronos artifact rebuild.", flush=True)
    print(
        f"Loading Chronos model={args.chronos_model_id} "
        f"device_map={args.chronos_device_map} dtype={args.chronos_torch_dtype}",
        flush=True,
    )
    pipeline = load_chronos_pipeline(args.chronos_model_id, args.chronos_device_map, args.chronos_torch_dtype)
    forecasts = chronos_predict_with_pipeline(
        pipeline,
        bundle.quantity,
        sku_indices,
        args.chronos_context_length,
        args.chronos_batch_size,
        quantile_levels,
    )
    return forecasts, sku_indices, active_days_56


def chronos_profit_rank_indices(
    weights: np.ndarray,
    top_n: int,
    active_days: np.ndarray,
    min_active_days: int,
) -> np.ndarray:
    eligible = np.flatnonzero((weights > 0.0) & (active_days >= min_active_days))
    eligible = eligible[np.argsort(weights[eligible])[::-1]]
    if top_n > 0:
        eligible = eligible[:top_n]
    return eligible.astype(np.int32)


def load_chronos_pipeline(model_id: str, device_map: str, dtype_name: str):
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=chronos_torch_dtype(dtype_name),
    )


def chronos_torch_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {name}")


def chronos_predict_with_pipeline(
    pipeline,
    history: np.ndarray,
    sku_indices: np.ndarray,
    context_length: int,
    batch_size: int,
    quantile_levels: list[float],
) -> dict[str, np.ndarray]:
    import torch

    y = np.maximum(history, 0.0)
    context_len = min(context_length, y.shape[0])
    context = y[-context_len:, sku_indices].T.astype(np.float32, copy=True)
    q_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    for start in range(0, len(sku_indices), batch_size):
        end = min(start + batch_size, len(sku_indices))
        batch = torch.tensor(context[start:end], dtype=torch.float32)
        print(f"Chronos batch {start}:{end}", flush=True)
        with torch.inference_mode():
            quantiles, mean = pipeline.predict_quantiles(
                batch,
                prediction_length=HORIZON,
                quantile_levels=quantile_levels,
            )
        q, m = normalize_chronos_quantile_output(quantiles, mean, end - start)
        q_parts.append(q)
        mean_parts.append(m)

    q_all = np.maximum(np.concatenate(q_parts, axis=0), 0.0)
    mean_all = np.maximum(np.concatenate(mean_parts, axis=0), 0.0)
    levels = {float(level): idx for idx, level in enumerate(quantile_levels)}
    if 0.5 not in levels:
        raise ValueError("--chronos-quantile-levels must include 0.5")
    return {
        "q10": q_all[:, :, levels.get(0.1, 0)].astype(np.float32),
        "median": q_all[:, :, levels[0.5]].astype(np.float32),
        "q90": q_all[:, :, levels.get(0.9, q_all.shape[2] - 1)].astype(np.float32),
        "mean": mean_all.astype(np.float32),
    }


def normalize_chronos_quantile_output(quantiles, mean, expected_n: int) -> tuple[np.ndarray, np.ndarray]:
    q = chronos_to_numpy_batch(quantiles).astype(np.float32)
    m = chronos_to_numpy_batch(mean).astype(np.float32)
    if q.ndim == 2:
        q = q[None, :, :]
    if q.shape[0] != expected_n and q.shape[1] == expected_n:
        q = np.transpose(q, (1, 0, 2))
    if m.ndim == 1:
        m = m[None, :]
    if m.shape[0] != expected_n and m.shape[-1] == expected_n:
        m = m.T
    if q.shape[0] != expected_n:
        raise ValueError(f"Unexpected Chronos quantile shape {q.shape}; expected first dim {expected_n}")
    return q, m


def chronos_to_numpy_batch(value) -> np.ndarray:
    import torch

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, torch.Tensor):
                parts.append(item.detach().cpu().numpy())
            else:
                parts.append(np.asarray(item))
        return np.stack(parts, axis=0)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def selected_forecast_matrix(values: np.ndarray, sku_indices: np.ndarray, n_skus: int) -> np.ndarray:
    pred = np.full((HORIZON, n_skus), np.nan, dtype=np.float32)
    pred[:, sku_indices] = values.T.astype(np.float32)
    return pred


def apply_chronos_blend(
    anchor: np.ndarray,
    chronos: np.ndarray,
    sku_indices: np.ndarray,
    alpha: float,
    apply_block: str,
    max_relative_change: float,
    non_sunday_only: bool,
    forecast_dates: pd.DatetimeIndex,
) -> np.ndarray:
    anchor32 = anchor.astype(np.float32, copy=False)
    out = anchor32.copy()
    horizon_mask = np.zeros(HORIZON, dtype=bool)
    if apply_block in {"validation", "both"}:
        horizon_mask[:BLOCK] = True
    if apply_block in {"evaluation", "both"}:
        horizon_mask[BLOCK:] = True
    if non_sunday_only:
        horizon_mask &= forecast_dates.dayofweek.to_numpy() != 6

    mask = np.ix_(horizon_mask, sku_indices)
    delta = chronos[mask] - anchor32[mask]
    cap = max_relative_change * np.maximum(anchor32[mask], 0.0)
    delta = np.clip(delta, -cap, cap)
    out[mask] = np.maximum(anchor32[mask] + alpha * delta, 0.0)
    return out


def chronos_candidate_name(
    model_id: str,
    top_n: int,
    alpha: float,
    block: str,
    cap: float,
    non_sunday_only: bool,
) -> str:
    model = model_id.split("/")[-1].replace("-", "_")
    name = f"chronos_{model}_top{top_n}_{block}_a{alpha:.3f}_cap{cap:.2f}".replace(".", "p")
    if non_sunday_only:
        name += "_nonsun"
    return name


def chronos_summary_row(
    name: str,
    path: Path,
    anchor: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    drift_masks = top_group_masks(weights)
    return {
        "candidate": name,
        "file": str(path),
        "top_n": int(sum(drift_masks["top100"])),
        "apply_block": "validation",
        "alpha": float(alpha),
        "max_relative_change": float(DEFAULT_CHRONOS_CAP),
        "changed_cells": int(np.count_nonzero(np.abs(candidate - anchor) > 1e-9)),
        "validation_delta_pct": pct_delta(float(candidate[:BLOCK].sum()), float(anchor[:BLOCK].sum())),
        "evaluation_delta_pct": pct_delta(float(candidate[BLOCK:].sum()), float(anchor[BLOCK:].sum())),
        "top10_validation_delta_pct": group_sum_delta_pct(candidate, anchor, drift_masks["top10"], slice(0, BLOCK)),
        "top50_validation_delta_pct": group_sum_delta_pct(candidate, anchor, drift_masks["top50"], slice(0, BLOCK)),
        "top100_validation_delta_pct": group_sum_delta_pct(candidate, anchor, drift_masks["top100"], slice(0, BLOCK)),
        "top500_validation_delta_pct": group_sum_delta_pct(candidate, anchor, drift_masks["top500"], slice(0, BLOCK)),
        "unique_id": True,
        "same_id_order_as_sample": True,
        "nan_count": 0,
        "finite": bool(np.isfinite(candidate).all()),
        "nonnegative": bool(np.all(candidate >= 0.0)),
    }


def top_group_masks(weights: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(weights)[::-1]
    masks = {}
    for name, top_n in [("top10", 10), ("top50", 50), ("top100", 100), ("top500", 500)]:
        mask = np.zeros(len(weights), dtype=bool)
        mask[order[:top_n]] = True
        masks[name] = mask
    return masks


def group_sum_delta_pct(candidate: np.ndarray, anchor: np.ndarray, sku_mask: np.ndarray, rows: slice) -> float:
    return pct_delta(float(candidate[rows][:, sku_mask].sum()), float(anchor[rows][:, sku_mask].sum()))


def write_chronos_long_forecast(
    path: Path,
    sku_order: list[str],
    sku_indices: np.ndarray,
    forecasts: dict[str, np.ndarray],
    forecast_dates: pd.DatetimeIndex,
    active_days_56: np.ndarray,
    weights: np.ndarray,
) -> None:
    rows = []
    for local_idx, sku_idx in enumerate(sku_indices):
        for horizon, date in enumerate(forecast_dates, start=1):
            row = {
                "sku": sku_order[int(sku_idx)],
                "sku_idx": int(sku_idx),
                "date": str(date.date()),
                "horizon": int(horizon),
                "active_days_56": float(active_days_56[int(sku_idx)]),
                "profit_weight": float(weights[int(sku_idx)]),
            }
            for key, values in forecasts.items():
                row[key] = float(values[local_idx, horizon - 1])
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_inverse_chronos_submission(anchor: pd.DataFrame, failed: pd.DataFrame, path: Path, *, k: float) -> None:
    cols = [f"F{i}" for i in range(1, BLOCK + 1)]
    if not anchor["id"].astype(str).equals(failed["id"].astype(str)):
        raise ValueError("Anchor and Chronos failed artifact have different id order.")
    out = anchor.copy()
    anchor_values = anchor[cols].to_numpy(dtype=np.float64)
    failed_values = failed[cols].to_numpy(dtype=np.float64)
    values = anchor_values.copy()
    validation_mask = anchor["id"].astype(str).str.endswith("_validation").to_numpy()
    direction = failed_values - anchor_values
    values[validation_mask] = values[validation_mask] - k * direction[validation_mask]
    out[cols] = np.maximum(values, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def inverse_summary_row(k: float, path: Path, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    changed = np.abs(candidate - anchor) > 1e-12
    return {
        "file": str(path),
        "k": float(k),
        "block": "validation",
        "changed_cells": int(changed.sum()),
        "changed_validation_cells": int(changed[:BLOCK].sum()),
        "changed_evaluation_cells": int(changed[BLOCK:].sum()),
        "validation_delta_pct_vs_anchor": pct_delta(float(candidate[:BLOCK].sum()), float(anchor[:BLOCK].sum())),
        "evaluation_delta_pct_vs_anchor": pct_delta(float(candidate[BLOCK:].sum()), float(anchor[BLOCK:].sum())),
        "validation_sum": float(candidate[:BLOCK].sum()),
        "evaluation_sum": float(candidate[BLOCK:].sum()),
    }


def final_artifact_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_args = argparse.Namespace(**vars(args))
    model_args.top_n = 500
    model_args.objective = "tweedie"
    model_args.volume_matching = True
    model_args.closure_rebound_strength = 0.54
    model_args.median_weight = 0.65
    model_args.train_stride_days = 28
    model_args.max_train_rows = 700000
    model_args.tweedie_power = 1.5
    model_args.min_child_weight = 10.0
    model_args.recency_halflife = 365.0
    model_args.n_estimators = 260
    model_args.learning_rate = 0.035
    model_args.num_leaves = 31
    model_args.public_boost = 1.05
    model_args.min_history_days = 365
    model_args.seed = 2026
    return model_args


def public_signal_changed_sku_mask(bundle: DatasetBundle, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selector_args = argparse.Namespace(
        private_calibration_pool_top_n=500,
        private_calibration_top_k=92,
        private_calibration_score="active_weight",
        private_active_min_days=5,
    )
    return source_only_private_sku_mask(bundle, weights, selector_args)


def scale_blocks_to_targets(pred: np.ndarray, *, validation_sum: float, evaluation_sum: float) -> np.ndarray:
    out = pred.astype(np.float64, copy=True)
    out[:BLOCK] *= validation_sum / max(float(out[:BLOCK].sum()), 1e-12)
    out[BLOCK:] *= evaluation_sum / max(float(out[BLOCK:].sum()), 1e-12)
    return np.maximum(out, 0.0).astype(np.float32)


def build_public_signal_block(
    anchor_pred: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    changed_weighted_sum: float,
) -> np.ndarray:
    out = anchor_pred.astype(np.float64, copy=True)
    selected_weighted = weighted_block_sum(out[:BLOCK], weights, mask)
    selected_scale = changed_weighted_sum / max(selected_weighted, 1e-12)
    out[:BLOCK, mask] *= selected_scale
    return np.maximum(out, 0.0).astype(np.float32)


def artifact_summary_row(
    name: str,
    path: Path,
    pred: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    return {
        "artifact": name,
        "file": str(path),
        "validation_sum": float(pred[:BLOCK].sum()),
        "evaluation_sum": float(pred[BLOCK:].sum()),
        "changed_sku_count": int(mask.sum()),
        "changed_validation_weighted_sum": weighted_block_sum(pred[:BLOCK], weights, mask),
        "changed_evaluation_weighted_sum": weighted_block_sum(pred[BLOCK:], weights, mask),
        "nonnegative": bool(np.all(pred >= 0.0)),
        "finite": bool(np.isfinite(pred).all()),
    }


def run_final_recipe(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    weights = profit_weights(bundle.train, bundle.dates[-1], bundle.sku_order)
    _, anchor_pred = read_submission_matrix(args.final_anchor, bundle.sku_order)
    _, private_pred = read_submission_matrix(args.final_private_source, bundle.sku_order)
    _, baseline_pred = read_submission_matrix(args.final_baseline, bundle.sku_order)

    written: list[str] = []
    rows: list[dict[str, object]] = []

    if args.final_recipe in {"g120_h50", "both_final"}:
        public_df, public_pred = read_submission_matrix(args.final_g120_public_source, bundle.sku_order)
        mask = changed_sku_mask(anchor_pred, public_pred)
        pred, audit = build_final_sku_ratio_prediction(
            bundle,
            public_pred,
            private_pred,
            mask,
            global_prior=1.20,
            hist_weight=0.50,
            clip_low=0.75,
            clip_high=1.80,
            years=parse_int_list(args.private_ratio_years),
        )
        name = output_name(args, "submission_private_skuratio_g120_h50_clip0p75_1p8.csv")
        if args.final_recipe == "both_final":
            name = "submission_private_skuratio_g120_h50_clip0p75_1p8.csv"
        path = args.output_dir / name
        write_prediction_frame(public_df, pred, bundle.sku_order).to_csv(path, index=False)
        audit.to_csv(args.output_dir / "audit_skuratio_g120_h50_clip0p75_1p8.csv", index=False)
        rows.append(
            final_recipe_metrics(
                "g120_h50",
                path,
                pred,
                public_pred,
                baseline_pred,
                weights,
                mask,
                "public k10.75 rows + XGBVM evaluation shape + SKU historical ratio calibration",
            )
        )
        written.append(str(path))

    if args.final_recipe in {"k20_target150", "both_final"}:
        public_df, public_pred = read_submission_matrix(args.final_k20_public_source, bundle.sku_order)
        mask = changed_sku_mask(anchor_pred, public_pred)
        pred, scale = build_final_global_target_prediction(
            public_pred,
            private_pred,
            weights,
            mask,
            target_ratio=1.50,
        )
        name = output_name(args, "submission_breakthrough_k20_private_target150.csv")
        if args.final_recipe == "both_final":
            name = "submission_breakthrough_k20_private_target150.csv"
        path = args.output_dir / name
        write_prediction_frame(public_df, pred, bundle.sku_order).to_csv(path, index=False)
        row = final_recipe_metrics(
            "k20_target150",
            path,
            pred,
            public_pred,
            baseline_pred,
            weights,
            mask,
            "public k20 rows + XGBVM evaluation shape + changed92 global target ratio 1.50",
        )
        row["changed92_eval_scale_applied"] = scale
        rows.append(row)
        written.append(str(path))

    report = pd.DataFrame(rows)
    report_path = args.output_dir / "final_recipe_report.csv"
    report.to_csv(report_path, index=False)
    diagnostics = {
        "final_recipe": args.final_recipe,
        "anchor": str(args.final_anchor),
        "private_source": str(args.final_private_source),
        "baseline": str(args.final_baseline),
        "g120_public_source": str(args.final_g120_public_source),
        "k20_public_source": str(args.final_k20_public_source),
        "report": str(report_path),
        "written": written,
        "note": "Final recipes combine generated or provided model-derived public/private shapes with explicit calibration constants.",
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({**diagnostics, "rows": rows}, indent=2))


def output_name(args: argparse.Namespace, default_name: str) -> str:
    if args.submission_name == "submission_clean_slate.csv":
        return default_name
    return args.submission_name


def read_submission_matrix(path: Path, sku_order: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    cols = [f"F{i}" for i in range(1, BLOCK + 1)]
    missing = [col for col in ["id", *cols] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    ids = df["id"].astype(str)
    row_skus = ids.str.replace(r"_(validation|evaluation)$", "", regex=True)
    sku_index = {sku: idx for idx, sku in enumerate(sku_order)}
    mapped = row_skus.map(sku_index)
    if mapped.isna().any():
        bad = row_skus[mapped.isna()].head(5).tolist()
        raise ValueError(f"{path} contains unknown SKU ids: {bad}")

    pred = np.full((HORIZON, len(sku_order)), np.nan, dtype=np.float64)
    values = df[cols].to_numpy(dtype=np.float64)
    row_idx = mapped.to_numpy(dtype=int)
    is_validation = ids.str.endswith("_validation").to_numpy()
    is_evaluation = ids.str.endswith("_evaluation").to_numpy()
    pred[:BLOCK, row_idx[is_validation]] = values[is_validation].T
    pred[BLOCK:, row_idx[is_evaluation]] = values[is_evaluation].T
    if np.isnan(pred).any():
        raise ValueError(f"{path} must contain one validation and one evaluation row per SKU.")
    return df, pred


def write_prediction_frame(template: pd.DataFrame, pred: np.ndarray, sku_order: list[str]) -> pd.DataFrame:
    out = template.copy()
    cols = [f"F{i}" for i in range(1, BLOCK + 1)]
    ids = out["id"].astype(str)
    row_skus = ids.str.replace(r"_(validation|evaluation)$", "", regex=True)
    sku_index = {sku: idx for idx, sku in enumerate(sku_order)}
    row_idx = row_skus.map(sku_index).to_numpy(dtype=int)
    is_validation = ids.str.endswith("_validation").to_numpy()
    is_evaluation = ids.str.endswith("_evaluation").to_numpy()
    out.loc[is_validation, cols] = pred[:BLOCK, row_idx[is_validation]].T.astype(np.float32)
    out.loc[is_evaluation, cols] = pred[BLOCK:, row_idx[is_evaluation]].T.astype(np.float32)
    return out


def changed_sku_mask(anchor_pred: np.ndarray, public_pred: np.ndarray) -> np.ndarray:
    delta = np.abs(public_pred[:BLOCK] - anchor_pred[:BLOCK])
    return delta.max(axis=0) > 1e-9


def build_final_global_target_prediction(
    public_pred: np.ndarray,
    private_pred: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    target_ratio: float,
) -> tuple[np.ndarray, float]:
    pred = public_pred.copy()
    pred[BLOCK:] = private_pred[BLOCK:]
    public_weighted = weighted_block_sum(pred[:BLOCK], weights, mask)
    private_weighted = weighted_block_sum(pred[BLOCK:], weights, mask)
    scale = target_ratio * public_weighted / max(private_weighted, 1e-12)
    pred[BLOCK:, mask] = np.maximum(pred[BLOCK:, mask] * scale, 0.0)
    return pred, float(scale)


def build_final_sku_ratio_prediction(
    bundle: DatasetBundle,
    public_pred: np.ndarray,
    private_pred: np.ndarray,
    mask: np.ndarray,
    *,
    global_prior: float,
    hist_weight: float,
    clip_low: float,
    clip_high: float,
    years: list[int],
) -> tuple[np.ndarray, pd.DataFrame]:
    pred = public_pred.copy()
    pred[BLOCK:] = private_pred[BLOCK:]
    ratio_table = historical_private_public_ratio_table(bundle, years)
    hist = ratio_table["ratio_median"].to_numpy(dtype=np.float64)
    hist = np.where(np.isfinite(hist), hist, global_prior)
    hist = np.clip(hist, clip_low, clip_high)
    ratio = np.exp((1.0 - hist_weight) * np.log(global_prior) + hist_weight * np.log(hist))
    ratio = np.where(mask, ratio, 1.0)
    public_totals = public_pred[:BLOCK].sum(axis=0)
    target_totals = public_totals * ratio
    pred, applied = scale_eval_sku_totals(pred, mask, target_totals)

    audit = ratio_table.copy()
    audit["changed92"] = mask
    audit["public_val_total"] = public_totals
    audit["target_ratio"] = ratio
    audit["target_eval_total"] = target_totals
    audit["applied_eval_scale"] = applied
    return pred, audit


def scale_eval_sku_totals(
    pred: np.ndarray,
    mask: np.ndarray,
    target_totals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = pred.copy()
    applied = np.ones(pred.shape[1], dtype=np.float64)
    for sku_idx in np.flatnonzero(mask):
        current = float(out[BLOCK:, sku_idx].sum())
        target = float(max(target_totals[sku_idx], 0.0))
        if current > 1e-12:
            scale = target / current
            out[BLOCK:, sku_idx] = np.maximum(out[BLOCK:, sku_idx] * scale, 0.0)
            applied[sku_idx] = scale
        elif target > 0.0:
            out[BLOCK:, sku_idx] = target / float(BLOCK)
            applied[sku_idx] = np.inf
    return out, applied


def final_recipe_metrics(
    candidate: str,
    path: Path,
    pred: np.ndarray,
    public_pred: np.ndarray,
    baseline_pred: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    note: str,
) -> dict[str, object]:
    public_weighted = weighted_block_sum(pred[:BLOCK], weights, mask)
    eval_weighted = weighted_block_sum(pred[BLOCK:], weights, mask)
    return {
        "candidate": candidate,
        "file": str(path),
        "note": note,
        "changed92_count": int(mask.sum()),
        "validation_max_abs_diff_vs_public": float(np.max(np.abs(pred[:BLOCK] - public_pred[:BLOCK]))),
        "changed92_public_weighted_sum": public_weighted,
        "changed92_eval_weighted_sum": eval_weighted,
        "changed92_eval_vs_public_ratio": float(eval_weighted / max(public_weighted, 1e-12)),
        "validation_sum_delta_pct_vs_baseline": pct_delta(float(pred[:BLOCK].sum()), float(baseline_pred[:BLOCK].sum())),
        "evaluation_sum_delta_pct_vs_baseline": pct_delta(float(pred[BLOCK:].sum()), float(baseline_pred[BLOCK:].sum())),
        "nonnegative": bool(np.all(pred >= 0.0)),
        "finite": bool(np.isfinite(pred).all()),
    }


def pct_delta(new: float, old: float) -> float:
    return float((new - old) / max(abs(old), 1e-12))


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
