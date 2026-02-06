#!/usr/bin/env python3
"""Compute empirical ROC from H0/H1 Parquet files using norm-based detector stats."""

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

# ====== Configure here ======
H0_PATH = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/free/sample_28_occ0_sig-noise_pwr-86.00_rxg28.00.parquet"
H1_PATH = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/dtv/sample_59_occ1_sig-dtv_pwr-82.00_rxg28.00.parquet"

I_COL = None  # e.g., "I"
Q_COL = None  # e.g., "Q"
COMPLEX_COL = None  # e.g., "iq"

N_DEFAULT = 2000
P_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
N_POINTS = 10000

OUT_DIR = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/roc"
ROC_CSV_NAME = "roc_empirical.csv"
ROC_LOG_PNG = "roc_empirical_log.png"
ROC_LINEAR_PNG = "roc_empirical_linear.png"


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        raise RuntimeError(
            "pandas is required to read Parquet files. "
            "Install with: pip install pandas pyarrow"
        )


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except Exception:
        raise RuntimeError(
            "matplotlib is required for plots. Install with: pip install matplotlib"
        )


def _load_parquet(path: Path):
    _require_pandas()
    import pandas as pd

    try:
        return pd.read_parquet(path, engine="pyarrow", use_legacy_dataset=True)
    except TypeError:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            pass
    except Exception:
        pass

    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        raise RuntimeError(
            "Failed to read Parquet with pyarrow and fastparquet. "
            "Try: pip install --user fastparquet"
        )


def _infer_iq_columns(columns: Sequence[str]):
    cols = [c.lower() for c in columns]
    if "i" in cols and "q" in cols:
        return columns[cols.index("i")], columns[cols.index("q")]
    if "real" in cols and "imag" in cols:
        return columns[cols.index("real")], columns[cols.index("imag")]
    if "re" in cols and "im" in cols:
        return columns[cols.index("re")], columns[cols.index("im")]
    if "iq_i" in cols and "iq_q" in cols:
        return columns[cols.index("iq_i")], columns[cols.index("iq_q")]
    if "i_data" in cols and "q_data" in cols:
        return columns[cols.index("i_data")], columns[cols.index("q_data")]
    return None, None


def load_iq_from_parquet(path, *, i_col=None, q_col=None, complex_col=None):
    df = _load_parquet(Path(path))

    if complex_col is not None:
        if complex_col not in df.columns:
            raise ValueError("complex_col not found.")
        data = df[complex_col].to_numpy()
        return np.asarray(data, dtype=np.complex128)

    if i_col is None or q_col is None:
        i_col, q_col = _infer_iq_columns(df.columns)
        if i_col is None or q_col is None:
            raise ValueError(
                "Could not infer I/Q columns. Provide I_COL and Q_COL."
            )

    if i_col not in df.columns or q_col not in df.columns:
        raise ValueError("I/Q columns not found.")

    i_vals = np.asarray(df[i_col].to_numpy(), dtype=np.float64)
    q_vals = np.asarray(df[q_col].to_numpy(), dtype=np.float64)
    return i_vals + 1j * q_vals


def segment_samples(x: np.ndarray, n: int):
    if n <= 0:
        raise ValueError("N must be > 0")
    total = x.size
    k = total // n
    if k == 0:
        raise ValueError("Not enough samples to form a single segment.")
    usable = k * n
    x = x[:usable]
    return x.reshape(k, n)


def compute_test_statistics(x: np.ndarray, n: int, p: float):
    if p <= 0:
        raise ValueError("p must be > 0")
    segments = segment_samples(x, n)
    mags = np.abs(segments) ** p
    return np.mean(mags, axis=1)


def _empirical_roc_for_column(h0: np.ndarray, h1: np.ndarray, n_points: int):
    min_thr = float(np.mean(h0) - 3 * np.std(h0))
    max_thr = float(np.mean(h1) + 3 * np.std(h1))
    thresholds = np.linspace(min_thr, max_thr, num=n_points)

    cdf_h0 = (h0[None, :] < thresholds[:, None]).mean(axis=1)
    cdf_h1 = (h1[None, :] < thresholds[:, None]).mean(axis=1)

    pfa = 1.0 - cdf_h0
    pd_vals = 1.0 - cdf_h1
    return thresholds, pfa, pd_vals


def compute_empirical_rocs(
    stats_h0: dict,
    stats_h1: dict,
    p_values: Iterable[float],
    n_points: int,
):
    records = []
    for p in p_values:
        h0_vals = stats_h0[p]
        h1_vals = stats_h1[p]
        thresholds, pfa, pd_vals = _empirical_roc_for_column(h0_vals, h1_vals, n_points)
        p_col = f"p_{p}"
        for thr, pfa_val, pd_val in zip(thresholds, pfa, pd_vals):
            records.append(
                {"p_column": p_col, "threshold": thr, "pfa": pfa_val, "pd": pd_val}
            )
    return records


def plot_empirical_rocs(roc_df, log_path: Path, linear_path: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    eps = 1e-6
    grouped = roc_df.groupby("p_column")

    plt.figure(figsize=(7, 5))
    for col, df_col in grouped:
        pfa = np.clip(df_col["pfa"].to_numpy(copy=False), eps, 1.0)
        pd_vals = np.clip(df_col["pd"].to_numpy(copy=False), eps, 1.0)
        plt.plot(pfa, pd_vals, label=col)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-4, 1e0)
    plt.ylim(1e-4, 1e0)
    plt.xlabel("Pfa")
    plt.ylabel("Pd")
    plt.title("Empirical ROC (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path, dpi=200)

    plt.figure(figsize=(7, 5))
    for col, df_col in grouped:
        plt.plot(df_col["pfa"].to_numpy(copy=False), df_col["pd"].to_numpy(copy=False), label=col)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.01, 0.1))
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.xlabel("Pfa")
    plt.ylabel("Pd")
    plt.title("Empirical ROC (linear)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(linear_path, dpi=200)


def main():
    h0_samples = load_iq_from_parquet(H0_PATH, i_col=I_COL, q_col=Q_COL, complex_col=COMPLEX_COL)
    h1_samples = load_iq_from_parquet(H1_PATH, i_col=I_COL, q_col=Q_COL, complex_col=COMPLEX_COL)

    stats_h0 = {}
    stats_h1 = {}
    for p in P_VALUES:
        stats_h0[p] = compute_test_statistics(h0_samples, N_DEFAULT, p)
        stats_h1[p] = compute_test_statistics(h1_samples, N_DEFAULT, p)

    _require_pandas()
    import pandas as pd

    records = compute_empirical_rocs(stats_h0, stats_h1, P_VALUES, N_POINTS)
    roc_df = pd.DataFrame.from_records(records, columns=["p_column", "threshold", "pfa", "pd"])

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_csv = out_dir / ROC_CSV_NAME
    roc_df.to_csv(roc_csv, index=False)

    plot_empirical_rocs(roc_df, out_dir / ROC_LOG_PNG, out_dir / ROC_LINEAR_PNG)


if __name__ == "__main__":
    main()
