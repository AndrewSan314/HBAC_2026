from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON = 56
BLOCK = 28
FOLD_STARTS = [
    "2025-07-12",
    "2025-05-17",
    "2025-03-22",
    "2025-01-25",
    "2024-11-30",
    "2024-10-05",
]


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    sample: pd.DataFrame
    sku_order: list[str]
    dates: pd.DatetimeIndex
    quantity: np.ndarray


def parse_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_bundle(data_dir: Path) -> DatasetBundle:
    train = pd.read_csv(
        data_dir / "train.csv",
        dtype={"Stt": "string", "ItemCode": "string"},
        parse_dates=["Date"],
        low_memory=False,
    )
    train["Quantity"] = pd.to_numeric(train["Quantity"], errors="coerce").fillna(0.0)
    train["SalesAmount_num"] = parse_number(train["SalesAmount"])
    train["CostAmount_num"] = parse_number(train["Cost Amount"])
    train["Profit"] = train["SalesAmount_num"] - train["CostAmount_num"]

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    sku_order = (
        sample["id"]
        .str.replace(r"_(validation|evaluation)$", "", regex=True)
        .drop_duplicates()
        .tolist()
    )

    dates = pd.date_range(train["Date"].min(), train["Date"].max(), freq="D")
    daily = (
        train.groupby(["Date", "ItemCode"], observed=True)["Quantity"]
        .sum()
        .reset_index()
    )
    wide = daily.pivot(index="Date", columns="ItemCode", values="Quantity")
    wide = wide.reindex(index=dates, columns=sku_order).fillna(0.0)

    return DatasetBundle(
        train=train,
        sample=sample,
        sku_order=sku_order,
        dates=dates,
        quantity=wide.to_numpy(dtype=np.float32, copy=True),
    )


def profit_weights(train: pd.DataFrame, cutoff: pd.Timestamp, sku_order: list[str]) -> np.ndarray:
    data = train.loc[pd.to_datetime(train["Date"]) <= pd.Timestamp(cutoff)]
    sales = parse_number(data["SalesAmount"]).fillna(0.0)
    cost = parse_number(data["Cost Amount"]).fillna(0.0)
    profit_by_sku = (sales - cost).groupby(data["ItemCode"].astype(str), observed=True).sum()
    weights = (
        profit_by_sku.reindex(pd.Index(sku_order).astype(str))
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("Profit weights are undefined because no SKU has positive profit.")
    return weights / total


def rmsse_scale(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    if history.shape[0] < 2:
        raise ValueError("RMSSE scale requires at least two historical days.")
    diffs = np.diff(history, axis=0)
    return np.mean(diffs * diffs, axis=0)


def wrmsse(
    actual: np.ndarray,
    pred: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if actual.shape != pred.shape:
        raise ValueError("actual and pred must have the same shape.")

    mse = np.mean((actual - pred) ** 2, axis=0)
    ok = scale > 1e-12
    rmsse = np.zeros_like(scale, dtype=np.float64)
    rmsse[ok] = np.sqrt(mse[ok] / scale[ok])
    rmsse[(~ok) & (mse > 1e-12)] = np.inf
    positive = weights > 0.0
    score = float("inf") if np.any(positive & np.isinf(rmsse)) else float(np.sum(weights[positive] * rmsse[positive]))
    return score, rmsse


def apply_block_scales(
    pred: np.ndarray,
    validation_scale: float = 1.0,
    evaluation_scale: float = 1.0,
) -> np.ndarray:
    scaled = pred.astype(np.float32, copy=True)
    scaled[:BLOCK] *= np.float32(validation_scale)
    scaled[BLOCK:] *= np.float32(evaluation_scale)
    return np.maximum(scaled, 0.0).astype(np.float32)


def write_submission(
    sample: pd.DataFrame,
    sku_order: list[str],
    pred: np.ndarray,
    output_path: Path,
) -> None:
    cols = [f"F{i}" for i in range(1, BLOCK + 1)]
    sku_index = {sku: i for i, sku in enumerate(sku_order)}
    row_skus = sample["id"].str.replace(r"_(validation|evaluation)$", "", regex=True)
    is_validation = sample["id"].str.endswith("_validation")

    values = np.zeros((len(sample), BLOCK), dtype=np.float32)
    val_rows = np.where(is_validation.to_numpy())[0]
    eval_rows = np.where(~is_validation.to_numpy())[0]
    val_indices = [sku_index[sku] for sku in row_skus.iloc[val_rows]]
    eval_indices = [sku_index[sku] for sku in row_skus.iloc[eval_rows]]
    values[val_rows] = pred[:BLOCK, val_indices].T
    values[eval_rows] = pred[BLOCK:, eval_indices].T

    output = sample.drop(columns=cols)
    output = pd.concat([output, pd.DataFrame(values, columns=cols, index=sample.index)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.6f")
