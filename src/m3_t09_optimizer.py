"""
src/m3_t09_optimizer.py - Track 3C What-If Scoring & Discrete Discount Optimizer.

Implements M3-T09 Design Decisions D1-D4:
  - D1: R0 batches get no markdown recommendation (status='no_markdown_historical_practice',
        reason='no_discount_region'). R1 batches are scored over {0.10, 0.15, ..., 0.75}
        restricted to depths marked supported by M3-T05 discount_support().
  - D2: Maximize batch expected profit; exact ties go to the shallower depth.
  - D3: margin_per_unit = best - second-best profit_per_unit; flat_curve = margin_per_unit <
        frozen regressor CV profit_unit_mae (from track_3b_regressor.json).
  - D4: Single, one-time markdown decision at inspection time.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from src.config import TRACK_3B_MODEL_PATH
from src.m3_t06_regressor import R1_DEPTH_GRID, predict_profit_curve


@lru_cache(maxsize=1)
def get_flat_curve_threshold(model_path: Optional[Path] = None) -> float:
    """Reads frozen Track 3B CV profit-per-unit MAE from JSON metadata (never from test)."""
    target = Path(model_path) if model_path is not None else TRACK_3B_MODEL_PATH
    meta_path = target.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    return float(meta["cv_scores"]["profit_unit_mae"])


def score_whatif(
    batch_df: pd.DataFrame,
    grid: Sequence[float] = R1_DEPTH_GRID,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Scores each batch at every candidate depth in `grid` using predict_profit_curve().
    Returns long DataFrame with columns:
      [record_id, depth, supported, reason, s_hat, expected_units_sold, profit_per_unit, batch_profit]
    """
    curve = predict_profit_curve(batch_df, grid=grid, model_path=model_path)
    out = curve.rename(columns={"discount_pct": "depth", "profit": "batch_profit"})[
        [
            "record_id",
            "depth",
            "supported",
            "reason",
            "s_hat",
            "expected_units_sold",
            "profit_per_unit",
            "batch_profit",
        ]
    ]
    return out


def optimize_batches(
    batch_df: pd.DataFrame,
    grid: Sequence[float] = R1_DEPTH_GRID,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Selects the profit-maximizing supported discount depth per batch (ties -> shallower depth).
    Supports both multi-row batches and single-row DataFrames (dashboard Simulator).
    """
    if batch_df.empty:
        raise ValueError("Input batch_df is empty.")

    flat_thr = get_flat_curve_threshold(model_path=model_path)
    long_df = score_whatif(batch_df, grid=grid, model_path=model_path)

    rec_ids = batch_df["record_id"].tolist() if "record_id" in batch_df.columns else list(range(len(batch_df)))
    n_grid = len(grid)
    assert len(long_df) == len(rec_ids) * n_grid

    depths = long_df["depth"].to_numpy(dtype=float).reshape(len(rec_ids), n_grid)
    supp = long_df["supported"].to_numpy(dtype=bool).reshape(len(rec_ids), n_grid)
    reasons = long_df["reason"].to_numpy(dtype=object).reshape(len(rec_ids), n_grid)
    prof_u = long_df["profit_per_unit"].to_numpy(dtype=float).reshape(len(rec_ids), n_grid)
    prof_b = long_df["batch_profit"].to_numpy(dtype=float).reshape(len(rec_ids), n_grid)

    rows_out: List[Dict[str, Any]] = []
    for i, rid in enumerate(rec_ids):
        if reasons[i, 0] == "region_R0_no_discount_evidence":
            rows_out.append({
                "record_id": rid,
                "region": "R0",
                "status": "no_markdown_historical_practice",
                "recommended_depth": np.nan,
                "predicted_profit_at_rec": np.nan,
                "predicted_profit_per_unit_at_rec": np.nan,
                "second_best_depth": np.nan,
                "margin_per_unit": np.nan,
                "flat_curve": False,
                "n_supported_depths": 0,
                "reason": "no_discount_region",
            })
            continue

        supp_idx = np.where(supp[i])[0]
        n_supp = int(len(supp_idx))
        if n_supp == 0:
            rows_out.append({
                "record_id": rid,
                "region": "R1",
                "status": "no_supported_depths",
                "recommended_depth": np.nan,
                "predicted_profit_at_rec": np.nan,
                "predicted_profit_per_unit_at_rec": np.nan,
                "second_best_depth": np.nan,
                "margin_per_unit": np.nan,
                "flat_curve": False,
                "n_supported_depths": 0,
                "reason": "thin_depth_bin",
            })
            continue

        # Sort supported candidates by (-batch_profit, depth) so exact ties go to shallower depth
        order = sorted(supp_idx, key=lambda j: (-prof_b[i, j], depths[i, j]))
        j_best = order[0]
        rec_d = float(depths[i, j_best])
        rec_pb = float(prof_b[i, j_best])
        rec_pu = float(prof_u[i, j_best])

        if n_supp >= 2:
            j_sec = order[1]
            sec_d = float(depths[i, j_sec])
            margin_u = float(rec_pu - prof_u[i, j_sec])
            is_flat = bool(margin_u < flat_thr)
        else:
            sec_d = np.nan
            margin_u = np.nan
            is_flat = False

        rows_out.append({
            "record_id": rid,
            "region": "R1",
            "status": "recommended",
            "recommended_depth": rec_d,
            "predicted_profit_at_rec": rec_pb,
            "predicted_profit_per_unit_at_rec": rec_pu,
            "second_best_depth": sec_d,
            "margin_per_unit": margin_u,
            "flat_curve": is_flat,
            "n_supported_depths": n_supp,
            "reason": "ok",
        })

    return pd.DataFrame(rows_out)
