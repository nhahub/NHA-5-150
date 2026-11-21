"""
src/m3_t10_tiers.py - Track 3C Tier Mapping & Dashboard Wording (Pure, No Model Calls).

Implements M3-T10 Pre-Registered Rule T-edges:
  Let F(d) be the training cumulative share of R1 recommendations with recommended_depth <= d.
  Edge1 = smallest d with F >= 0.50; Edge2 = smallest d with F >= 0.80; Edge3 = smallest d with F >= 0.95.
  Tiers: Low (d <= Edge1), Medium (Edge1 < d <= Edge2), High (Edge2 < d <= Edge3), Critical (d > Edge3).
  R0 batches receive no tier (status: "No markdown (historical practice)").
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import OPTIMIZER_RECS_TRAIN_PATH, TIER_EDGES_PATH

R0_STATUS_LABEL = "No markdown (historical practice)"

TIER_WORDING: Dict[str, Dict[str, str]] = {
    "Low": {
        "label": "Low",
        "meaning": "Low: a shallow markdown of about 10-15% is recommended.",
    },
    "Medium": {
        "label": "Medium",
        "meaning": "Medium: a moderate markdown of about 20-25% is recommended.",
    },
    "High": {
        "label": "High",
        "meaning": "High: a deeper markdown of about 30-35% is recommended.",
    },
    "Critical": {
        "label": "Critical",
        "meaning": "Critical: a deep markdown above 35% is recommended.",
    },
    "R0": {
        "label": R0_STATUS_LABEL,
        "meaning": "No markdown recommended: this batch is outside the range where the data supports a discount decision.",
    },
}


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_tier_edges(
    recs_train: pd.DataFrame,
    save: bool = True,
    output_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Derives discrete tier edges from TRAINING R1 recommendations only using Rule T-edges.
    Saves metadata JSON to models/track_3c_tier_edges.json.
    """
    r1 = recs_train
    if "region" in r1.columns:
        r1 = r1[r1["region"] == "R1"]
    depths = r1["recommended_depth"].dropna().to_numpy(dtype=float)
    n_train = int(len(depths))
    assert n_train > 0, "No valid recommended_depth rows in recs_train."

    uniq_d = np.sort(np.unique(depths))
    cdf = {float(d): float((depths <= d + 1e-9).mean()) for d in uniq_d}

    edge1 = float(next(d for d in uniq_d if cdf[float(d)] >= 0.50))
    edge2 = float(next(d for d in uniq_d if cdf[float(d)] >= 0.80))
    edge3 = float(next(d for d in uniq_d if cdf[float(d)] >= 0.95))

    tier_counts = {
        "Low": int((depths <= edge1 + 1e-9).sum()),
        "Medium": int(((depths > edge1 + 1e-9) & (depths <= edge2 + 1e-9)).sum()),
        "High": int(((depths > edge2 + 1e-9) & (depths <= edge3 + 1e-9)).sum()),
        "Critical": int((depths > edge3 + 1e-9).sum()),
    }
    achieved_shares = {k: float(v / n_train) for k, v in tier_counts.items()}

    src_file = Path(source_path) if source_path is not None else OPTIMIZER_RECS_TRAIN_PATH
    payload: Dict[str, Any] = {
        "rule": "T-edges: Edge1 = min d with F(d)>=0.50; Edge2 = min d with F(d)>=0.80; Edge3 = min d with F(d)>=0.95 on training R1",
        "training_row_count": n_train,
        "edges": {"edge1": edge1, "edge2": edge2, "edge3": edge3},
        "cdf": {f"{k:.2f}": v for k, v in cdf.items()},
        "achieved_counts": tier_counts,
        "achieved_shares": achieved_shares,
        "source_file": str(src_file.name),
        "source_sha256": _sha256_file(src_file),
    }

    if save:
        out_p = Path(output_path) if output_path is not None else TIER_EDGES_PATH
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2))

    return payload


def load_tier_edges(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads frozen tier edges JSON from models/track_3c_tier_edges.json."""
    target = Path(path) if path is not None else TIER_EDGES_PATH
    return json.loads(target.read_text())


def assign_tier(
    df: pd.DataFrame,
    edges: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Pure tier assignment function (no model calls; works on single-row or batch DataFrames).
    Returns a DataFrame indexed like `df` with columns: [tier, tier_label, tier_meaning].
    R0 batches receive tier=None and tier_label="No markdown (historical practice)".
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    ed = edges["edges"] if (edges is not None and "edges" in edges) else (edges if edges is not None else load_tier_edges()["edges"])
    e1, e2, e3 = float(ed["edge1"]), float(ed["edge2"]), float(ed["edge3"])

    has_region = "region" in df.columns
    tiers: List[Optional[str]] = []
    labels: List[str] = []
    meanings: List[str] = []

    for i in range(len(df)):
        row_region = str(df["region"].iloc[i]) if has_region else None
        d_val = df["recommended_depth"].iloc[i] if "recommended_depth" in df.columns else np.nan
        is_r0 = (row_region == "R0") or pd.isna(d_val)

        if is_r0:
            tiers.append(None)
            labels.append(TIER_WORDING["R0"]["label"])
            meanings.append(TIER_WORDING["R0"]["meaning"])
        else:
            d = float(d_val)
            if d <= e1 + 1e-9:
                t = "Low"
            elif d <= e2 + 1e-9:
                t = "Medium"
            elif d <= e3 + 1e-9:
                t = "High"
            else:
                t = "Critical"
            tiers.append(t)
            labels.append(TIER_WORDING[t]["label"])
            meanings.append(TIER_WORDING[t]["meaning"])

    return pd.DataFrame(
        {"tier": tiers, "tier_label": labels, "tier_meaning": meanings},
        index=df.index,
    )
