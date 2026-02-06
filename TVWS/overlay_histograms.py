#!/usr/bin/env python3
"""Overlay multiple histograms with same scale and left-tail zoom."""

import math
from pathlib import Path

import numpy as np

# ====== Configure here ======
FILE_A = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/usrp_only/sample_28_occ0_sig-noise_pwr-90.00_rxg28.00.parquet"
FILE_B = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/dtv/sample_59_occ1_sig-dtv_pwr-82.00_rxg28.00.parquet"
FILE_C = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/free/sample_28_occ0_sig-noise_pwr-86.00_rxg28.00.parquet"
LABEL_A = "USRP-only (free, -90dBm): 28 dB"
LABEL_B = "DTV Channel (occupied, -82dBm): 28 dB"
LABEL_C = "Free air (free, -86dBm): 28 dB"

# Use one of: "I", "Q", "MAG", "REAL", "IMAG"
COMPONENT = "I"

I_COL = None  # e.g., "I"
Q_COL = None  # e.g., "Q"
COMPLEX_COL = None  # e.g., "iq"

OUT_DIR = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/overlay_histograms"
OUT_NAME = "overlay_hist.png"
OUT_IQ_A_NAME = "iq_timeseries_a.png"
OUT_IQ_B_NAME = "iq_timeseries_b.png"
OUT_IQ_C_NAME = "iq_timeseries_c.png"

NUM_BINS = 45
MAX_SAMPLES = 2000  # number of samples for index plots and histograms
LEFT_TAIL_PERCENTILE = 5.0  # zoom left tail up to this percentile
X_LIM = 0.007  # set to None to auto-scale from data
GENERATE_TIME_SERIES = True


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


def _infer_iq_columns(columns):
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


def select_component(x, component):
    component = component.upper()
    if component in ("I", "REAL"):
        return np.real(x)
    if component in ("Q", "IMAG"):
        return np.imag(x)
    if component == "MAG":
        return np.abs(x)
    raise ValueError("COMPONENT must be one of I, Q, MAG, REAL, IMAG.")


def main():
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_a = load_iq_from_parquet(FILE_A, i_col=I_COL, q_col=Q_COL, complex_col=COMPLEX_COL)
    x_b = load_iq_from_parquet(FILE_B, i_col=I_COL, q_col=Q_COL, complex_col=COMPLEX_COL)
    x_c = load_iq_from_parquet(FILE_C, i_col=I_COL, q_col=Q_COL, complex_col=COMPLEX_COL)

    a = select_component(x_a, COMPONENT)
    b = select_component(x_b, COMPONENT)
    c = select_component(x_c, COMPONENT)

    if MAX_SAMPLES is not None:
        a = a[:MAX_SAMPLES]
        b = b[:MAX_SAMPLES]
        c = c[:MAX_SAMPLES]

    if X_LIM is not None:
        data_min = -float(X_LIM)
        data_max = float(X_LIM)
    else:
        data_min = float(np.min([a.min(), b.min(), c.min()]))
        data_max = float(np.max([a.max(), b.max(), c.max()]))
    if not math.isfinite(data_min) or not math.isfinite(data_max):
        raise ValueError("Invalid data range for histogram.")

    bins = np.linspace(data_min, data_max, NUM_BINS + 1)
    left_tail_max = np.percentile(np.concatenate([a, b, c]), LEFT_TAIL_PERCENTILE)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(a, bins=bins, alpha=0.6, label=LABEL_A)
    axes[0].hist(b, bins=bins, alpha=0.6, label=LABEL_B)
    axes[0].hist(c, bins=bins, alpha=0.6, label=LABEL_C)
    axes[0].set_title(f"Overlay ({COMPONENT}) - full range")
    axes[0].set_xlabel("Amplitude")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right")

    axes[1].hist(a, bins=bins, alpha=0.6, label=LABEL_A)
    axes[1].hist(b, bins=bins, alpha=0.6, label=LABEL_B)
    axes[1].hist(c, bins=bins, alpha=0.6, label=LABEL_C)
    axes[1].set_xlim(data_min, left_tail_max)
    axes[1].set_title(f"Left tail (<= {LEFT_TAIL_PERCENTILE}th pct)")
    axes[1].set_xlabel("Amplitude")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / OUT_NAME, dpi=150)
    plt.close(fig)

    if GENERATE_TIME_SERIES:
        # Time-series plots for I and Q per input (same scale)
        i_a = np.real(x_a)
        q_a = np.imag(x_a)
        i_b = np.real(x_b)
        q_b = np.imag(x_b)
        i_c = np.real(x_c)
        q_c = np.imag(x_c)
        if MAX_SAMPLES is not None:
            i_a = i_a[:MAX_SAMPLES]
            q_a = q_a[:MAX_SAMPLES]
            i_b = i_b[:MAX_SAMPLES]
            q_b = q_b[:MAX_SAMPLES]
            i_c = i_c[:MAX_SAMPLES]
            q_c = q_c[:MAX_SAMPLES]

        idx_a = np.arange(len(i_a))
        fig_a, ax_a = plt.subplots(figsize=(10, 4))
        ax_a.plot(idx_a, i_a, linewidth=0.8, label=f"I - {LABEL_A}")
        ax_a.plot(idx_a, q_a, linewidth=0.8, label=f"Q - {LABEL_A}")
        ax_a.autoscale(enable=True, axis="y", tight=True)
        ax_a.set_title(f"I/Q vs index - {LABEL_A}")
        ax_a.set_xlabel("Index")
        ax_a.set_ylabel("Amplitude")
        ax_a.legend(loc="upper right")
        fig_a.tight_layout()
        fig_a.savefig(out_dir / OUT_IQ_A_NAME, dpi=150)
        plt.close(fig_a)

        idx_b = np.arange(len(i_b))
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        ax_b.plot(idx_b, i_b, linewidth=0.8, label=f"I - {LABEL_B}")
        ax_b.plot(idx_b, q_b, linewidth=0.8, label=f"Q - {LABEL_B}")
        ax_b.autoscale(enable=True, axis="y", tight=True)
        ax_b.set_title(f"I/Q vs index - {LABEL_B}")
        ax_b.set_xlabel("Index")
        ax_b.set_ylabel("Amplitude")
        ax_b.legend(loc="upper right")
        fig_b.tight_layout()
        fig_b.savefig(out_dir / OUT_IQ_B_NAME, dpi=150)
        plt.close(fig_b)

        idx_c = np.arange(len(i_c))
        fig_c, ax_c = plt.subplots(figsize=(10, 4))
        ax_c.plot(idx_c, i_c, linewidth=0.8, label=f"I - {LABEL_C}")
        ax_c.plot(idx_c, q_c, linewidth=0.8, label=f"Q - {LABEL_C}")
        ax_c.autoscale(enable=True, axis="y", tight=True)
        ax_c.set_title(f"I/Q vs index - {LABEL_C}")
        ax_c.set_xlabel("Index")
        ax_c.set_ylabel("Amplitude")
        ax_c.legend(loc="upper right")
        fig_c.tight_layout()
        fig_c.savefig(out_dir / OUT_IQ_C_NAME, dpi=150)
        plt.close(fig_c)


if __name__ == "__main__":
    main()
