from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hbac_forecast import DatasetBundle, parse_number


# ═══════════════════════════════════════════
# Vietnamese Holiday Feature Constants & Helpers
# ═══════════════════════════════════════════
HOLIDAY_NAME_CODES = {
    "none": 0,
    "new_year": 1,
    "tet": 2,
    "hung_kings": 3,
    "reunification": 4,
    "labor_day": 5,
    "national_day_adjacent": 6,
    "national_day": 7,
}
TET_DATES = pd.to_datetime([
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22",
    "2024-02-10", "2025-01-29", "2026-02-17",
])
HUNG_KINGS_DATES = pd.to_datetime([
    "2020-04-02", "2021-04-21", "2022-04-10", "2023-04-29",
    "2024-04-18", "2025-04-07", "2026-04-26",
])
MID_AUTUMN_DATES = pd.to_datetime([
    "2020-10-01", "2021-09-21", "2022-09-10", "2023-09-29",
    "2024-09-17", "2025-10-06", "2026-09-25",
])
SCHOOL_START_DATES = pd.to_datetime([f"{year}-09-05" for year in range(2020, 2027)])

PUBLIC_HOLIDAY_EVENTS: list[tuple[pd.Timestamp, int]] = []
for _year in range(2020, 2027):
    PUBLIC_HOLIDAY_EVENTS.extend([
        (pd.Timestamp(_year, 1, 1), HOLIDAY_NAME_CODES["new_year"]),
        (pd.Timestamp(_year, 4, 30), HOLIDAY_NAME_CODES["reunification"]),
        (pd.Timestamp(_year, 5, 1), HOLIDAY_NAME_CODES["labor_day"]),
        (pd.Timestamp(_year, 9, 1), HOLIDAY_NAME_CODES["national_day_adjacent"]),
        (pd.Timestamp(_year, 9, 2), HOLIDAY_NAME_CODES["national_day"]),
    ])
PUBLIC_HOLIDAY_EVENTS.extend((date, HOLIDAY_NAME_CODES["tet"]) for date in TET_DATES)
PUBLIC_HOLIDAY_EVENTS.extend((date, HOLIDAY_NAME_CODES["hung_kings"]) for date in HUNG_KINGS_DATES)
PUBLIC_HOLIDAY_EVENTS = sorted(PUBLIC_HOLIDAY_EVENTS, key=lambda x: x[0])
PUBLIC_HOLIDAY_DATES = pd.DatetimeIndex([date for date, _ in PUBLIC_HOLIDAY_EVENTS])
PUBLIC_HOLIDAY_CODE_BY_DATE = {date.normalize(): code for date, code in PUBLIC_HOLIDAY_EVENTS}


def _event_distance(dt: pd.Timestamp, dates: pd.DatetimeIndex) -> tuple[int, int]:
    deltas = (dates - dt).days
    future = deltas[deltas >= 0]
    past = -deltas[deltas <= 0]
    days_to = int(future.min()) if len(future) else 9999
    days_since = int(past.min()) if len(past) else 9999
    return days_to, days_since


def holiday_event_features(dt: pd.Timestamp) -> dict:
    dt = pd.Timestamp(dt).normalize()
    days_to_holiday, days_since_holiday = _event_distance(dt, PUBLIC_HOLIDAY_DATES)
    days_to_tet, days_since_tet = _event_distance(dt, pd.DatetimeIndex(TET_DATES))
    days_to_school, days_since_school = _event_distance(dt, pd.DatetimeIndex(SCHOOL_START_DATES))
    days_to_mid_autumn, days_since_mid_autumn = _event_distance(dt, pd.DatetimeIndex(MID_AUTUMN_DATES))
    national_day = pd.Timestamp(dt.year, 9, 2)
    days_since_national_day = (dt - national_day).days if dt >= national_day else 9999

    holiday_code = PUBLIC_HOLIDAY_CODE_BY_DATE.get(dt, HOLIDAY_NAME_CODES["none"])
    return {
        "is_vn_holiday": int(holiday_code != HOLIDAY_NAME_CODES["none"]),
        "holiday_name_code": holiday_code,
        "days_to_nearest_holiday": days_to_holiday,
        "days_since_nearest_holiday": days_since_holiday,
        "days_since_national_day": days_since_national_day,
        "is_post_holiday_1_3": int(1 <= days_since_holiday <= 3),
        "is_post_holiday_4_7": int(4 <= days_since_holiday <= 7),
        "is_post_holiday_8_14": int(8 <= days_since_holiday <= 14),
        "is_pre_holiday_1_3": int(1 <= days_to_holiday <= 3),
        "is_pre_holiday_4_7": int(4 <= days_to_holiday <= 7),
        "is_tet_window": int(days_to_tet <= 14 or days_since_tet <= 14),
        "is_pre_tet_1_7": int(1 <= days_to_tet <= 7),
        "is_pre_tet_8_14": int(8 <= days_to_tet <= 14),
        "is_post_tet_1_7": int(1 <= days_since_tet <= 7),
        "is_post_tet_8_14": int(8 <= days_since_tet <= 14),
        "days_to_tet": days_to_tet,
        "days_since_tet": days_since_tet,
        "is_school_start_window": int(days_to_school <= 7 or days_since_school <= 21),
        "days_since_school_start": days_since_school,
        "days_to_school_start": days_to_school,
        "is_back_to_school_1_7": int(1 <= days_since_school <= 7),
        "is_back_to_school_8_21": int(8 <= days_since_school <= 21),
        "days_to_mid_autumn": days_to_mid_autumn,
        "days_since_mid_autumn": days_since_mid_autumn,
        "is_mid_autumn_window": int(days_to_mid_autumn <= 7 or days_since_mid_autumn <= 7),
    }


FEATURE_HORIZON = 56


@dataclass(frozen=True)
class CleanFeatureCache:
    price: np.ndarray
    cost: np.ndarray


def build_clean_feature_cache(bundle: DatasetBundle) -> CleanFeatureCache:
    train = bundle.train.copy()
    if "UnitPrice" in train.columns:
        train["UnitPrice_num"] = parse_number(train["UnitPrice"]).fillna(0.0)
    else:
        train["UnitPrice_num"] = 0.0
    if "Unit Cost" in train.columns:
        train["UnitCost_num"] = parse_number(train["Unit Cost"]).fillna(0.0)
    else:
        train["UnitCost_num"] = 0.0

    return CleanFeatureCache(
        price=_pivot_daily_mean(bundle, train, "UnitPrice_num"),
        cost=_pivot_daily_mean(bundle, train, "UnitCost_num"),
    )


def _pivot_daily_mean(bundle: DatasetBundle, train: pd.DataFrame, column: str) -> np.ndarray:
    daily = (
        train.groupby(["Date", "ItemCode"], observed=True)[column]
        .mean()
        .reset_index()
    )
    wide = daily.pivot(index="Date", columns="ItemCode", values=column)
    wide = wide.reindex(index=bundle.dates, columns=bundle.sku_order).fillna(0.0)
    return wide.to_numpy(dtype=np.float32, copy=True)


def top_profit_indices(weights: np.ndarray, top_n: int) -> np.ndarray:
    if top_n <= 0 or top_n >= len(weights):
        return np.arange(len(weights), dtype=np.int64)
    return np.argsort(weights)[::-1][:top_n].astype(np.int64)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"sku", "sku_idx", "date", "target"}
    return [col for col in frame.columns if col not in excluded]


def make_direct_feature_frame(
    bundle: DatasetBundle,
    *,
    cutoff_idx: int,
    horizons: list[int] | range | np.ndarray,
    sku_indices: np.ndarray,
    weights: np.ndarray,
    stat_pred: np.ndarray,
    cache: CleanFeatureCache,
) -> pd.DataFrame:
    if cutoff_idx < 1:
        raise ValueError("cutoff_idx must leave at least two history rows.")

    sku_indices = np.asarray(sku_indices, dtype=np.int64)
    horizons = np.asarray(list(horizons), dtype=np.int64)
    history = bundle.quantity[: cutoff_idx + 1]
    history_dates = bundle.dates[: cutoff_idx + 1]
    y_pos = np.maximum(history, 0.0)
    n_history = history.shape[0]
    closure = _global_closure_features(y_pos, history_dates)
    closure_stat_pred = _closure_adjusted_dow_blend(
        y_pos,
        history_dates,
        pd.DatetimeIndex([bundle.dates[cutoff_idx] + pd.Timedelta(days=int(h)) for h in horizons]),
        sku_indices,
    )

    base = _history_base_features(y_pos, history, history_dates, sku_indices, weights)
    price = _price_features(cache.price[: cutoff_idx + 1], sku_indices)
    cost = _price_features(cache.cost[: cutoff_idx + 1], sku_indices, prefix="cost")

    rows: list[pd.DataFrame] = []
    sku_labels = np.asarray(bundle.sku_order, dtype=object)[sku_indices]
    for h_pos, h in enumerate(horizons):
        if h < 1:
            raise ValueError("horizons are 1-based and must be positive.")
        forecast_date = bundle.dates[cutoff_idx] + pd.Timedelta(days=int(h))
        dow = int(forecast_date.dayofweek)
        month = int(forecast_date.month)
        frame = pd.DataFrame(
            {
                "sku": sku_labels,
                "sku_idx": sku_indices,
                "date": forecast_date,
                "horizon": np.full(len(sku_indices), h, dtype=np.int16),
                "dow": np.full(len(sku_indices), dow, dtype=np.int8),
                "month": np.full(len(sku_indices), month, dtype=np.int8),
                "is_weekend": np.full(len(sku_indices), int(dow >= 5), dtype=np.int8),
                "dow_sin": np.full(len(sku_indices), np.sin(2.0 * np.pi * dow / 7.0)),
                "dow_cos": np.full(len(sku_indices), np.cos(2.0 * np.pi * dow / 7.0)),
                "month_sin": np.full(len(sku_indices), np.sin(2.0 * np.pi * month / 12.0)),
                "month_cos": np.full(len(sku_indices), np.cos(2.0 * np.pi * month / 12.0)),
                "stat_pred": stat_pred[h - 1, sku_indices],
                "stat_pred_no_closure": closure_stat_pred[h_pos, :],
                "stat_pred_closure_gap": closure_stat_pred[h_pos, :] - stat_pred[h - 1, sku_indices],
                "same_dow_mean_8": _same_dow_stat(y_pos, history_dates, dow, sku_indices, 8, "mean"),
                "same_dow_active_rate_26": _same_dow_active_rate(y_pos, history_dates, dow, sku_indices, 26),
                "global_zero_days_7": np.full(len(sku_indices), closure["global_zero_days_7"], dtype=np.float32),
                "global_zero_days_14": np.full(len(sku_indices), closure["global_zero_days_14"], dtype=np.float32),
                "non_sunday_zero_days_7": np.full(len(sku_indices), closure["non_sunday_zero_days_7"], dtype=np.float32),
                "days_since_non_sunday_closure": np.full(
                    len(sku_indices),
                    closure["days_since_non_sunday_closure"],
                    dtype=np.float32,
                ),
                "days_after_non_sunday_closure": np.full(
                    len(sku_indices),
                    closure["days_since_non_sunday_closure"] + int(h),
                    dtype=np.float32,
                ),
                "is_post_recent_closure_1_7": np.full(
                    len(sku_indices),
                    int(1 <= closure["days_since_non_sunday_closure"] + int(h) <= 7 and dow < 6),
                    dtype=np.int8,
                ),
                "is_post_recent_closure_8_14": np.full(
                    len(sku_indices),
                    int(8 <= closure["days_since_non_sunday_closure"] + int(h) <= 14 and dow < 6),
                    dtype=np.int8,
                ),
            }
        )
        for key, value in base.items():
            frame[key] = value
        for key, value in price.items():
            frame[key] = value
        for key, value in cost.items():
            frame[key] = value
        h_dict = holiday_event_features(forecast_date)
        for key, value in h_dict.items():
            dtype = np.int8 if ("is_" in key or "code" in key) else np.float32
            frame[key] = np.full(len(sku_indices), value, dtype=dtype)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def attach_targets(
    frame: pd.DataFrame,
    bundle: DatasetBundle,
    *,
    target_start_idx: int,
) -> pd.DataFrame:
    out = frame.copy()
    row_idx = target_start_idx + out["horizon"].to_numpy(dtype=np.int64) - 1
    sku_idx = out["sku_idx"].to_numpy(dtype=np.int64)
    y = bundle.quantity[row_idx, sku_idx]
    out["target"] = np.maximum(y, 0.0).astype(np.float32)
    return out


def make_training_starts(
    *,
    min_start_idx: int,
    max_start_idx: int,
    stride_days: int,
) -> list[int]:
    if max_start_idx < min_start_idx:
        return []
    starts = list(range(min_start_idx, max_start_idx + 1, stride_days))
    if starts and starts[-1] != max_start_idx:
        starts.append(max_start_idx)
    if not starts:
        starts = [max_start_idx]
    return starts


def _history_base_features(
    y_pos: np.ndarray,
    raw: np.ndarray,
    dates: pd.DatetimeIndex,
    sku_indices: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(weights)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    out: dict[str, np.ndarray] = {
        "profit_weight": weights[sku_indices].astype(np.float32),
        "profit_rank": ranks[sku_indices].astype(np.float32),
    }
    for lag in (1, 7, 28):
        out[f"lag_{lag}"] = _lag(y_pos, lag, sku_indices)
    for window in (7, 21, 56, 112):
        win = y_pos[-min(window, y_pos.shape[0]) :, sku_indices]
        out[f"rmean_{window}"] = win.mean(axis=0, dtype=np.float64).astype(np.float32)
        clean_win = _last_non_closure_rows(y_pos, dates, window)[:, sku_indices]
        out[f"rmean_{window}_no_closure"] = clean_win.mean(axis=0, dtype=np.float64).astype(np.float32)
    win56 = y_pos[-min(56, y_pos.shape[0]) :, sku_indices]
    out["rmed_56"] = np.median(win56, axis=0).astype(np.float32)
    clean_win56 = _last_non_closure_rows(y_pos, dates, 56)[:, sku_indices]
    out["rmed_56_no_closure"] = np.median(clean_win56, axis=0).astype(np.float32)
    for window in (28, 56, 112):
        win = y_pos[-min(window, y_pos.shape[0]) :, sku_indices]
        out[f"active_rate_{window}"] = ((win > 1e-9).mean(axis=0)).astype(np.float32)
    win365 = y_pos[-min(365, y_pos.shape[0]) :, sku_indices]
    out["active_days_365"] = (win365 > 1e-9).sum(axis=0).astype(np.float32)
    out["days_since_last_sale"] = _days_since_last(y_pos[:, sku_indices] > 1e-9)
    out["days_since_last_return"] = _days_since_last(raw[:, sku_indices] < -1e-9)
    out["inactivity_status"] = np.select(
        [
            out["days_since_last_sale"] > 180,
            out["days_since_last_sale"] > 90,
            out["days_since_last_sale"] > 56,
        ],
        [3, 2, 1],
        default=0,
    ).astype(np.float32)
    out["trend_21_56"] = ((out["rmean_21"] + 0.05) / (out["rmean_56"] + 0.05)).astype(np.float32)
    out["trend_7_56"] = ((out["rmean_7"] + 0.05) / (out["rmean_56"] + 0.05)).astype(np.float32)
    out["closure_mean_ratio_21"] = ((out["rmean_21_no_closure"] + 0.05) / (out["rmean_21"] + 0.05)).astype(np.float32)
    out["closure_med_ratio_56"] = ((out["rmed_56_no_closure"] + 0.05) / (out["rmed_56"] + 0.05)).astype(np.float32)
    return out


def _global_closure_features(y_pos: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, float]:
    totals = y_pos.sum(axis=1, dtype=np.float64)
    global_zero = totals <= 1e-9
    non_sunday_closure = _non_sunday_global_closure_mask(y_pos, dates)
    last7 = slice(max(0, len(dates) - 7), len(dates))
    last14 = slice(max(0, len(dates) - 14), len(dates))
    closure_idx = np.flatnonzero(non_sunday_closure)
    days_since = float(len(dates)) if len(closure_idx) == 0 else float(len(dates) - 1 - closure_idx[-1])
    return {
        "global_zero_days_7": float(global_zero[last7].sum()),
        "global_zero_days_14": float(global_zero[last14].sum()),
        "non_sunday_zero_days_7": float(non_sunday_closure[last7].sum()),
        "days_since_non_sunday_closure": days_since,
    }


def _non_sunday_global_closure_mask(y_pos: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    totals = y_pos.sum(axis=1, dtype=np.float64)
    dows = np.asarray(dates.dayofweek)
    return (totals <= 1e-9) & (dows < 6)


def _global_open_day_mask(y_pos: np.ndarray) -> np.ndarray:
    return y_pos.sum(axis=1, dtype=np.float64) > 1e-9


def _last_non_closure_rows(y_pos: np.ndarray, dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    keep = _global_open_day_mask(y_pos)
    selected = np.flatnonzero(keep)
    if len(selected) == 0:
        return y_pos[-min(window, y_pos.shape[0]) :]
    selected = selected[-min(window, len(selected)) :]
    return y_pos[selected]


def _closure_adjusted_dow_blend(
    y_pos: np.ndarray,
    dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    sku_indices: np.ndarray,
    median56_weight: float = 0.70,
) -> np.ndarray:
    clean = _last_non_closure_rows(y_pos, dates, max(112, 56))
    clean_dates = dates[np.flatnonzero(_global_open_day_mask(y_pos))[-clean.shape[0] :]]
    median_56 = np.median(clean[-min(56, clean.shape[0]) :, sku_indices], axis=0)
    mean_21 = clean[-min(21, clean.shape[0]) :, sku_indices].mean(axis=0, dtype=np.float64)
    base = median56_weight * median_56 + (1.0 - median56_weight) * mean_21
    dow_factors = _global_dow_factors(clean, clean_dates)
    pred = np.zeros((len(forecast_dates), len(sku_indices)), dtype=np.float32)
    for row, date in enumerate(forecast_dates):
        pred[row] = base * dow_factors[int(date.dayofweek)]
    return np.maximum(pred, 0.0).astype(np.float32)


def _lag(y_pos: np.ndarray, lag: int, sku_indices: np.ndarray) -> np.ndarray:
    idx = y_pos.shape[0] - lag
    if idx < 0:
        return np.zeros(len(sku_indices), dtype=np.float32)
    return y_pos[idx, sku_indices].astype(np.float32)


def _days_since_last(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[1], mask.shape[0], dtype=np.float32)
    for col in range(mask.shape[1]):
        hits = np.flatnonzero(mask[:, col])
        if len(hits):
            out[col] = mask.shape[0] - 1 - hits[-1]
    return out


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


def _same_dow_stat(
    y_pos: np.ndarray,
    dates: pd.DatetimeIndex,
    dow: int,
    sku_indices: np.ndarray,
    matches: int,
    stat: str,
) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(dates.dayofweek == dow))
    if len(idx) == 0:
        return np.zeros(len(sku_indices), dtype=np.float32)
    selected = idx[-matches:]
    values = y_pos[selected][:, sku_indices]
    if stat == "median":
        return np.median(values, axis=0).astype(np.float32)
    return values.mean(axis=0, dtype=np.float64).astype(np.float32)


def _same_dow_active_rate(
    y_pos: np.ndarray,
    dates: pd.DatetimeIndex,
    dow: int,
    sku_indices: np.ndarray,
    matches: int,
) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(dates.dayofweek == dow))
    if len(idx) == 0:
        return np.zeros(len(sku_indices), dtype=np.float32)
    selected = idx[-matches:]
    return (y_pos[selected][:, sku_indices] > 1e-9).mean(axis=0).astype(np.float32)


def _price_features(
    matrix: np.ndarray,
    sku_indices: np.ndarray,
    prefix: str = "price",
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    hist = matrix[:, sku_indices]
    out[f"last_{prefix}"] = _last_positive_by_col(hist)
    for window in (56, 365):
        win = hist[-min(window, hist.shape[0]) :]
        out[f"{prefix}_median_{window}"] = _positive_median_by_col(win)
    out[f"{prefix}_change_56_365"] = (
        (out[f"{prefix}_median_56"] + 1e-3) / (out[f"{prefix}_median_365"] + 1e-3)
    ).astype(np.float32)
    return out


def _last_positive_by_col(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape[1], dtype=np.float32)
    for col in range(values.shape[1]):
        positive = values[:, col][values[:, col] > 1e-6]
        if len(positive):
            out[col] = positive[-1]
    return out


def _positive_median_by_col(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape[1], dtype=np.float32)
    for col in range(values.shape[1]):
        positive = values[:, col][values[:, col] > 1e-6]
        if len(positive):
            out[col] = float(np.median(positive))
    return out
