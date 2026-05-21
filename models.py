from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from .features import FEATURE_HORIZON, feature_columns
except ImportError:
    from features import FEATURE_HORIZON, feature_columns


@dataclass(frozen=True)
class XGBFit:
    model: Any
    feature_cols: list[str]
    objective: str


def starter_dow_blend(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    *,
    median56_weight: float = 0.70,
) -> np.ndarray:
    y_pos = np.maximum(history, 0.0)
    median_56 = _positive_median(y_pos, 56)
    mean_21 = _positive_mean(y_pos, 21)
    base = median56_weight * median_56 + (1.0 - median56_weight) * mean_21
    dow_factors = _global_dow_factors(y_pos, history_dates)
    pred = np.zeros((len(forecast_dates), history.shape[1]), dtype=np.float32)
    for row, date in enumerate(forecast_dates):
        pred[row] = base * dow_factors[int(date.dayofweek)]
    pred[np.abs(pred) < 1e-9] = 0.0
    return np.maximum(pred, 0.0).astype(np.float32)


def open_day_dow_blend(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    *,
    median56_weight: float = 0.70,
) -> np.ndarray:
    y_pos = np.maximum(history, 0.0)
    open_day = y_pos.sum(axis=1, dtype=np.float64) > 1e-9
    if int(open_day.sum()) < 56:
        return starter_dow_blend(history, history_dates, forecast_dates, median56_weight=median56_weight)
    open_history = y_pos[open_day]
    open_dates = history_dates[open_day]
    median_56 = _positive_median(open_history, 56)
    mean_21 = _positive_mean(open_history, 21)
    base = median56_weight * median_56 + (1.0 - median56_weight) * mean_21
    dow_factors = _global_dow_factors(open_history, open_dates)
    pred = np.zeros((len(forecast_dates), history.shape[1]), dtype=np.float32)
    for row, date in enumerate(forecast_dates):
        pred[row] = base * dow_factors[int(date.dayofweek)]
    pred[np.abs(pred) < 1e-9] = 0.0
    return np.maximum(pred, 0.0).astype(np.float32)


def closure_rebound_blend(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    *,
    median56_weight: float = 0.70,
    rebound_strength: float = 0.60,
    apply_first_block_only: bool = True,
) -> np.ndarray:
    standard = starter_dow_blend(history, history_dates, forecast_dates, median56_weight=median56_weight)
    if not _has_recent_non_sunday_closure(history, history_dates):
        return standard

    # A late non-Sunday closure depresses the rolling base. Blend toward an
    # open-business-day backbone only for the first block, while Sundays stay low.
    open_day = open_day_dow_blend(history, history_dates, forecast_dates, median56_weight=median56_weight)
    adjusted = standard.astype(np.float32, copy=True)
    horizon_mask = np.ones(len(forecast_dates), dtype=bool)
    if apply_first_block_only:
        horizon_mask[28:] = False
    non_sunday = np.asarray(forecast_dates.dayofweek < 6)
    mask = horizon_mask & non_sunday
    strength = np.float32(np.clip(rebound_strength, 0.0, 1.0))
    adjusted[mask] = standard[mask] + strength * (open_day[mask] - standard[mask])
    return np.maximum(adjusted, 0.0).astype(np.float32)


def fit_direct_xgb(
    train_frame: pd.DataFrame,
    *,
    objective: str,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    seed: int,
    max_train_rows: int | None = None,
    public_boost: float = 1.05,
    tweedie_power: float = 1.2,
    min_child_weight_val: float = 20.0,
    recency_halflife: float = 0.0,
) -> XGBFit:
    import xgboost as xgb

    if objective not in {"tweedie", "poisson"}:
        raise ValueError("objective must be 'tweedie' or 'poisson'.")
    if "target" not in train_frame.columns:
        raise ValueError("train_frame must include a target column.")

    # Keep positives and a reproducible zero sample; full sparse zeros made XGBoost
    # overfit level shifts without improving the top-weight SKU shape.
    train_frame = _sample_training_rows(train_frame, max_train_rows=max_train_rows, seed=seed)
    cols = feature_columns(train_frame)
    X = train_frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_frame["target"].to_numpy(dtype=np.float32)
    sample_weight = _row_weights(train_frame, public_boost=public_boost, recency_halflife=recency_halflife)

    params: dict[str, Any] = {
        "objective": "reg:tweedie" if objective == "tweedie" else "count:poisson",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_leaves": num_leaves,
        "max_depth": 0,
        "tree_method": "hist",
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": min_child_weight_val,
        "reg_alpha": 0.05,
        "reg_lambda": 0.25,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = tweedie_power

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=sample_weight)
    return XGBFit(model=model, feature_cols=cols, objective=objective)


def predict_direct_xgb(fitted: XGBFit, frame: pd.DataFrame) -> np.ndarray:
    X = frame[fitted.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred = fitted.model.predict(X)
    return np.maximum(np.asarray(pred, dtype=np.float32), 0.0)


def blend_top_sku_predictions(
    stat_pred: np.ndarray,
    xgb_top_pred: np.ndarray,
    sku_indices: np.ndarray,
    alpha: float,
    xgb_scale: float = 1.0,
) -> np.ndarray:
    blended = stat_pred.astype(np.float32, copy=True)
    top_stat = stat_pred[:, sku_indices]
    top_xgb = np.float32(xgb_scale) * xgb_top_pred
    blended[:, sku_indices] = (1.0 - alpha) * top_stat + alpha * top_xgb
    return np.maximum(blended, 0.0).astype(np.float32)


def apply_volume_matching(
    stat_pred: np.ndarray,
    ml_top_pred: np.ndarray,
    sku_indices: np.ndarray,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Calibrate ml_top_pred (shape: horizon, len(sku_indices)) per SKU so that their total sum 
    over the horizon matches the total sum of the robust statistical model (stat_pred).
    We perform this calibration block-wise (separately for Days 1-28 and Days 29-56)
    to prevent volume leakage across blocks due to massive holiday spikes.
    """
    BLOCK = 28
    horizon = ml_top_pred.shape[0]
    
    calibrated = np.zeros_like(ml_top_pred)
    top_stat = stat_pred[:, sku_indices]
    
    # Block 1: Days 1 to BLOCK (Validation)
    ml_totals_1 = ml_top_pred[:BLOCK].sum(axis=0)
    stat_totals_1 = top_stat[:BLOCK].sum(axis=0)
    scale_factors_1 = stat_totals_1 / (ml_totals_1 + epsilon)
    calibrated[:BLOCK] = ml_top_pred[:BLOCK] * scale_factors_1[np.newaxis, :]
    
    # Block 2: Days BLOCK+1 to end (Evaluation)
    if horizon > BLOCK:
        ml_totals_2 = ml_top_pred[BLOCK:].sum(axis=0)
        stat_totals_2 = top_stat[BLOCK:].sum(axis=0)
        scale_factors_2 = stat_totals_2 / (ml_totals_2 + epsilon)
        calibrated[BLOCK:] = ml_top_pred[BLOCK:] * scale_factors_2[np.newaxis, :]
        
    return np.maximum(calibrated, 0.0).astype(np.float32)



def _positive_mean(y_pos: np.ndarray, window: int) -> np.ndarray:
    win = y_pos[-min(window, y_pos.shape[0]) :]
    return win.mean(axis=0, dtype=np.float64).astype(np.float32)


def _positive_median(y_pos: np.ndarray, window: int) -> np.ndarray:
    win = y_pos[-min(window, y_pos.shape[0]) :]
    return np.median(win, axis=0).astype(np.float32)


def _global_dow_factors(y_pos: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    totals = y_pos.sum(axis=1, dtype=np.float64)
    dows = np.asarray(dates.dayofweek)
    avg = np.array(
        [totals[dows == day].mean() if np.any(dows == day) else 0.0 for day in range(7)],
        dtype=np.float64,
    )
    mean = avg.mean()
    if mean <= 0.0:
        return np.ones(7, dtype=np.float32)
    factors = avg / mean
    factors = factors / factors.mean()
    return factors.astype(np.float32)


def _has_recent_non_sunday_closure(
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    lookback_days: int = 7,
    max_days_since: int = 5,
) -> bool:
    if history.shape[0] == 0:
        return False
    y_pos = np.maximum(history, 0.0)
    totals = y_pos.sum(axis=1, dtype=np.float64)
    dows = np.asarray(history_dates.dayofweek)
    non_sunday_zero = (totals <= 1e-9) & (dows < 6)
    recent = non_sunday_zero[-min(lookback_days, len(non_sunday_zero)) :]
    if not bool(recent.any()):
        return False
    hits = np.flatnonzero(non_sunday_zero)
    return bool(len(hits) and (len(non_sunday_zero) - 1 - hits[-1]) <= max_days_since)


def _row_weights(
    frame: pd.DataFrame,
    public_boost: float = 1.05,
    recency_halflife: float = 0.0,
) -> np.ndarray:
    weights = frame["profit_weight"].to_numpy(dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / max(weights.mean(), 1e-12)
    horizon = frame["horizon"].to_numpy(dtype=np.float64)
    h_boost = np.where(horizon <= FEATURE_HORIZON / 2, public_boost, 1.0)
    # Recency decay: rows with recent forecast dates get higher weight
    if recency_halflife > 0.0 and "date" in frame.columns:
        dates = pd.to_datetime(frame["date"])
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.to_numpy(dtype=np.float64)
        recency = np.exp(-0.693 * days_ago / recency_halflife)
        recency = recency / max(recency.mean(), 1e-12)
    else:
        recency = np.ones(len(frame), dtype=np.float64)
    return (weights * h_boost * recency).astype(np.float32)


def _sample_training_rows(
    frame: pd.DataFrame,
    *,
    max_train_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_train_rows is None or max_train_rows <= 0 or len(frame) <= max_train_rows:
        return frame
    rng = np.random.default_rng(seed)
    positive_idx = frame.index[frame["target"].to_numpy(dtype=np.float32) > 0.0].to_numpy()
    zero_idx = frame.index[frame["target"].to_numpy(dtype=np.float32) <= 0.0].to_numpy()
    keep_positive = positive_idx
    if len(keep_positive) > max_train_rows:
        keep = rng.choice(keep_positive, size=max_train_rows, replace=False)
    else:
        remaining = max_train_rows - len(keep_positive)
        keep_zero = rng.choice(zero_idx, size=min(remaining, len(zero_idx)), replace=False)
        keep = np.concatenate([keep_positive, keep_zero])
    return frame.loc[np.sort(keep)].reset_index(drop=True)
