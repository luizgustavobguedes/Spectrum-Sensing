#!/usr/bin/env python3
"""Norm-based detector for I/Q samples stored in Parquet files.

Implements T(p) = (1/N) * sum_{n=1..N} |x(n)|^p for windowed sensing.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


P_VALUES_DEFAULT = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# ====== Configure here (paths, columns, N, Pfa) ======
H0_PATHS_DEFAULT = [
    # "/abs/path/to/h0_file_1.parquet",
    "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/free/sample_3_occ0_sig-noise_pwr-86.00_rxg3.00.parquet",
]
H1_PATHS_DEFAULT = [
    # "/abs/path/to/h1_file_1.parquet",
    "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/dtv/sample_56_occ1_sig-dtv_pwr-82.00_rxg25.00.parquet",
]
H0_DIR_DEFAULT = "/media/inatel-crr/FCB0ACA7B0AC69BA/luiz/tvws/sink_dataset/p-norm-4-feb-2026/free"  # e.g., "/abs/path/to/h0_folder"
H1_DIR_DEFAULT = None  # e.g., "/abs/path/to/h1_folder"
I_COL_DEFAULT = None  # e.g., "I"
Q_COL_DEFAULT = None  # e.g., "Q"
COMPLEX_COL_DEFAULT = None  # e.g., "iq"
N_DEFAULT = 2000
PFA_DEFAULT = 0.1
OUT_DIR_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results"
TOTAL_SAMPLES_EXPECTED = 6_000_000
MAX_H0_STATS_DEFAULT = 80_000
THRESHOLDS_CSV_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/thresholds.csv"
TEST_STATS_CSV_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/test_statistics.csv"
PD_FROM_THRESHOLDS_CSV_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/pd_from_thresholds.csv"
ONLY_RECOMPUTE_PD_DEFAULT = True
WRITE_H1_STATS_FROM_RECOMPUTE_DEFAULT = True
H1_STATS_FROM_RECOMPUTE_CSV_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/test_statistics_h1_only.csv"
GENERATE_PLOTS_DEFAULT = True
PLOT_SAMPLES_DEFAULT = 50_000
IQ_PLOTS_DIR_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/iq_plots"
HIST_PLOTS_DIR_DEFAULT = "/home/inatel-crr/Documents/Atividades/2026/TVWS/results/iq_histograms"
HIST_BINS_DEFAULT = 200
PLOT_H0_DEFAULT = False
PLOT_H1_DEFAULT = True


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
    except Exception:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "pandas is required to read Parquet files. "
            "Install with: pip install pandas pyarrow"
        )


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except Exception:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "matplotlib is required for plots. Install with: pip install matplotlib"
        )


def _load_parquet(path: Path):
    _require_pandas()
    import pandas as pd

    # Try pyarrow first with legacy dataset (more compatible with older files)
    try:
        return pd.read_parquet(path, engine="pyarrow", use_legacy_dataset=True)
    except TypeError:
        # pandas version does not support use_legacy_dataset
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            pass
    except Exception:
        pass

    # Fallback to fastparquet if available
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception as exc:
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


def load_iq_from_parquet(
    path: Union[str, Path],
    *,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
) -> np.ndarray:
    df = _load_parquet(Path(path))

    if complex_col is not None:
        if complex_col not in df.columns:
            raise ValueError(f"complex_col '{complex_col}' not found in {path}")
        data = df[complex_col].to_numpy()
        return np.asarray(data, dtype=np.complex128)

    if i_col is None or q_col is None:
        i_col, q_col = _infer_iq_columns(df.columns)
        if i_col is None or q_col is None:
            raise ValueError(
                "Could not infer I/Q columns. Provide --i-col and --q-col, "
                "or --complex-col."
            )

    if i_col not in df.columns or q_col not in df.columns:
        raise ValueError(
            f"I/Q columns not found in {path}. "
            f"Got i_col={i_col}, q_col={q_col}."
        )

    i_vals = np.asarray(df[i_col].to_numpy(), dtype=np.float64)
    q_vals = np.asarray(df[q_col].to_numpy(), dtype=np.float64)
    return i_vals + 1j * q_vals


def load_iq_many(
    paths: Iterable[Union[str, Path]],
    *,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
) -> np.ndarray:
    arrays = [
        load_iq_from_parquet(p, i_col=i_col, q_col=q_col, complex_col=complex_col)
        for p in paths
    ]
    if not arrays:
        raise ValueError("No input paths provided.")
    return np.concatenate(arrays, axis=0)


def list_parquet_files(folder: Union[str, Path]):
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError("H0 folder does not exist or is not a directory.")
    return sorted(str(p) for p in folder.glob("*.parquet"))


def _safe_stem(path: Union[str, Path]):
    return Path(path).stem.replace(" ", "_")


def plot_iq_time_series(
    path: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    max_samples: int,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
):
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = load_iq_from_parquet(path, i_col=i_col, q_col=q_col, complex_col=complex_col)
    if max_samples is not None:
        x = x[: max_samples]
    idx = np.arange(x.size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, np.real(x), linewidth=0.8, label="I")
    ax.plot(idx, np.imag(x), linewidth=0.8, label="Q")
    ax.set_title(f"I/Q vs index - {_safe_stem(path)}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe_stem(path)}_iq_timeseries.png", dpi=150)
    plt.close(fig)


def plot_iq_histograms(
    path: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    max_samples: int,
    bins: int,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
):
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = load_iq_from_parquet(path, i_col=i_col, q_col=q_col, complex_col=complex_col)
    if max_samples is not None:
        x = x[: max_samples]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.real(x), bins=bins, alpha=0.6, label="I")
    ax.hist(np.imag(x), bins=bins, alpha=0.6, label="Q")
    ax.set_title(f"I/Q histogram - {_safe_stem(path)}")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe_stem(path)}_iq_hist.png", dpi=150)
    plt.close(fig)


def segment_samples(x: np.ndarray, n: int, *, drop_remainder: bool = True):
    if n <= 0:
        raise ValueError("N must be > 0")
    total = x.size
    k = total // n
    if k == 0:
        raise ValueError("Not enough samples to form a single segment.")
    usable = k * n
    if usable < total and not drop_remainder:
        raise ValueError("Remainder samples exist; set drop_remainder=True.")
    x = x[:usable]
    return x.reshape(k, n)


def compute_test_statistics(x: np.ndarray, n: int, p: float):
    if p <= 0:
        raise ValueError("p must be > 0")
    segments = segment_samples(x, n)
    mags = np.abs(segments) ** p
    return np.mean(mags, axis=1)


def compute_threshold(sorted_stats: np.ndarray, pfa: float):
    if not (0.0 < pfa < 1.0):
        raise ValueError("Pfa must be in (0, 1)")
    k = sorted_stats.size
    if k == 0:
        raise ValueError("No statistics provided.")
    idx = int(math.floor((1.0 - pfa) * k))
    idx = max(0, min(idx, k - 1))
    return float(sorted_stats[idx])


def compute_pd(stats: np.ndarray, threshold: float):
    if stats.size == 0:
        raise ValueError("No statistics provided for Pd.")
    return float(np.mean(stats > threshold))


def _accumulate_h0_stats(
    *,
    h0_paths: Sequence[Union[str, Path]],
    n: int,
    p: float,
    max_stats: int,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
):
    collected = []
    remaining = max_stats
    for path in h0_paths:
        if remaining <= 0:
            break
        samples = load_iq_from_parquet(
            path, i_col=i_col, q_col=q_col, complex_col=complex_col
        )
        stats = compute_test_statistics(samples, n, p)
        if stats.size > remaining:
            stats = stats[:remaining]
        collected.append(stats)
        remaining -= stats.size
    if not collected:
        raise ValueError("No H0 statistics collected.")
    return np.concatenate(collected, axis=0)


def process_detector(
    *,
    h0_paths: Sequence[Union[str, Path]],
    h1_samples: np.ndarray,
    n: int,
    p_values: Sequence[float],
    pfa: float,
    max_h0_stats: int,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
):
    thresholds = {}
    pd_values = {}
    stats_h0 = {}
    stats_h1 = {}

    for p in p_values:
        h0_stats = _accumulate_h0_stats(
            h0_paths=h0_paths,
            n=n,
            p=p,
            max_stats=max_h0_stats,
            i_col=i_col,
            q_col=q_col,
            complex_col=complex_col,
        )
        h1_stats = compute_test_statistics(h1_samples, n, p)
        thresholds[p] = compute_threshold(np.sort(h0_stats), pfa)
        pd_values[p] = compute_pd(h1_stats, thresholds[p])
        stats_h0[p] = h0_stats
        stats_h1[p] = h1_stats

    return thresholds, pd_values, stats_h0, stats_h1


def write_thresholds_csv(path: Path, thresholds: Dict[float, float]):
    lines = ["p,threshold"]
    for p in thresholds:
        lines.append(f"{p},{thresholds[p]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pd_csv(path: Path, pd_values: Dict[float, float], *, pfa: float):
    lines = ["p,pd,pfa"]
    for p in pd_values:
        lines.append(f"{p},{pd_values[p]},{pfa}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_statistics_csv(
    path: Path,
    stats_h0: Dict[float, np.ndarray],
    stats_h1: Dict[float, np.ndarray],
):
    lines = ["p,hypothesis,segment_index,value"]
    for p, stats in stats_h0.items():
        for idx, value in enumerate(stats):
            lines.append(f"{p},H0,{idx},{value}")
    for p, stats in stats_h1.items():
        for idx, value in enumerate(stats):
            lines.append(f"{p},H1,{idx},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_thresholds_csv(path: Union[str, Path]):
    thresholds = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise ValueError("thresholds.csv is empty.")
        for line in f:
            line = line.strip()
            if not line:
                continue
            p_str, thr_str = line.split(",", 1)
            thresholds[float(p_str)] = float(thr_str)
    return thresholds


def _load_h1_stats_from_csv(path: Union[str, Path]):
    stats = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise ValueError("test_statistics.csv is empty.")
        for line in f:
            line = line.strip()
            if not line:
                continue
            p_str, hyp, _idx, value_str = line.split(",", 3)
            if hyp != "H1":
                continue
            p_val = float(p_str)
            stats.setdefault(p_val, []).append(float(value_str))
    return {p: np.asarray(vals, dtype=np.float64) for p, vals in stats.items()}


def write_pd_from_thresholds_csv(
    thresholds_csv: Union[str, Path],
    out_csv: Union[str, Path],
    *,
    h1_samples: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    p_values: Optional[Sequence[float]] = None,
    i_col: Optional[str] = None,
    q_col: Optional[str] = None,
    complex_col: Optional[str] = None,
    h1_paths: Optional[Sequence[Union[str, Path]]] = None,
    write_h1_stats_csv: Optional[Union[str, Path]] = None,
):
    thresholds = _load_thresholds_csv(thresholds_csv)
    if h1_samples is None:
        if h1_paths is None:
            raise ValueError("Provide h1_samples or h1_paths.")
        if n is None:
            raise ValueError("Provide n to compute H1 statistics.")
        h1_samples = load_iq_many(
            h1_paths, i_col=i_col, q_col=q_col, complex_col=complex_col
        )
    if n is None:
        raise ValueError("Provide n to compute H1 statistics.")
    if p_values is None:
        p_values = sorted(thresholds.keys())

    h1_stats = {}
    for p in p_values:
        h1_stats[p] = compute_test_statistics(h1_samples, n, p)

    if write_h1_stats_csv is not None:
        lines = ["p,hypothesis,segment_index,value"]
        for p, stats in h1_stats.items():
            for idx, value in enumerate(stats):
                lines.append(f"{p},H1,{idx},{value}")
        Path(write_h1_stats_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["p,pd,threshold_source,test_stats_source"]
    for p, thr in thresholds.items():
        if p not in h1_stats:
            continue
        pd_val = compute_pd(h1_stats[p], thr)
        source = "computed_from_h1_parquet"
        lines.append(f"{p},{pd_val},{thresholds_csv},{source}")
    Path(out_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Norm-based detector for Parquet I/Q.")
    parser.add_argument(
        "--h0-paths",
        nargs="+",
        default=H0_PATHS_DEFAULT,
        help="Parquet files for H0 (idle).",
    )
    parser.add_argument(
        "--h0-dir",
        default=H0_DIR_DEFAULT,
        help="Folder with H0 parquet files (no subfolders).",
    )
    parser.add_argument(
        "--h1-paths",
        nargs="+",
        default=H1_PATHS_DEFAULT,
        help="Parquet files for H1 (occupied).",
    )
    parser.add_argument(
        "--h1-dir",
        default=H1_DIR_DEFAULT,
        help="Folder with H1 parquet files (no subfolders).",
    )
    parser.add_argument("--i-col", default=I_COL_DEFAULT, help="Column name for I samples.")
    parser.add_argument("--q-col", default=Q_COL_DEFAULT, help="Column name for Q samples.")
    parser.add_argument("--complex-col", default=COMPLEX_COL_DEFAULT, help="Column name for complex samples.")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Segment length N.")
    parser.add_argument(
        "--pfa",
        type=float,
        default=PFA_DEFAULT,
        help="False alarm probability (default 0.1).",
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=P_VALUES_DEFAULT,
        help="List of p values.",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR_DEFAULT,
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--max-h0-stats",
        type=int,
        default=MAX_H0_STATS_DEFAULT,
        help="Max H0 test statistics to accumulate for thresholds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.h0_dir:
        args.h0_paths = list_parquet_files(args.h0_dir)
    if args.h1_dir:
        args.h1_paths = list_parquet_files(args.h1_dir)

    if not args.h0_paths or not args.h1_paths:
        raise ValueError(
            "Defina H0_PATHS_DEFAULT/H1_PATHS_DEFAULT no topo do script, "
            "ou defina H0_DIR_DEFAULT/H1_DIR_DEFAULT, "
            "ou passe --h0-paths/--h1-paths via CLI."
        )

    if GENERATE_PLOTS_DEFAULT:
        h0_plot_dir = Path(IQ_PLOTS_DIR_DEFAULT) / "H0"
        h1_plot_dir = Path(IQ_PLOTS_DIR_DEFAULT) / "H1"
        h0_hist_dir = Path(HIST_PLOTS_DIR_DEFAULT) / "H0"
        h1_hist_dir = Path(HIST_PLOTS_DIR_DEFAULT) / "H1"

        if PLOT_H0_DEFAULT:
            for path in args.h0_paths:
                plot_iq_time_series(
                    path,
                    out_dir=h0_plot_dir,
                    max_samples=PLOT_SAMPLES_DEFAULT,
                    i_col=args.i_col,
                    q_col=args.q_col,
                    complex_col=args.complex_col,
                )
                plot_iq_histograms(
                    path,
                    out_dir=h0_hist_dir,
                    max_samples=PLOT_SAMPLES_DEFAULT,
                    bins=HIST_BINS_DEFAULT,
                    i_col=args.i_col,
                    q_col=args.q_col,
                    complex_col=args.complex_col,
                )

        if PLOT_H1_DEFAULT:
            for path in args.h1_paths:
                plot_iq_time_series(
                    path,
                    out_dir=h1_plot_dir,
                    max_samples=PLOT_SAMPLES_DEFAULT,
                    i_col=args.i_col,
                    q_col=args.q_col,
                    complex_col=args.complex_col,
                )
                plot_iq_histograms(
                    path,
                    out_dir=h1_hist_dir,
                    max_samples=PLOT_SAMPLES_DEFAULT,
                    bins=HIST_BINS_DEFAULT,
                    i_col=args.i_col,
                    q_col=args.q_col,
                    complex_col=args.complex_col,
                )

    if ONLY_RECOMPUTE_PD_DEFAULT:
        write_pd_from_thresholds_csv(
            THRESHOLDS_CSV_DEFAULT,
            PD_FROM_THRESHOLDS_CSV_DEFAULT,
            h1_paths=args.h1_paths,
            n=args.n,
            p_values=args.p_values,
            i_col=args.i_col,
            q_col=args.q_col,
            complex_col=args.complex_col,
            write_h1_stats_csv=H1_STATS_FROM_RECOMPUTE_CSV_DEFAULT
            if WRITE_H1_STATS_FROM_RECOMPUTE_DEFAULT
            else None,
        )
        return

    h1_samples = load_iq_many(
        args.h1_paths, i_col=args.i_col, q_col=args.q_col, complex_col=args.complex_col
    )

    if h1_samples.size != TOTAL_SAMPLES_EXPECTED:
        print(
            f"Aviso: H1 tem {h1_samples.size} amostras "
            f"(esperado {TOTAL_SAMPLES_EXPECTED})."
        )

    thresholds, pd_values, stats_h0, stats_h1 = process_detector(
        h0_paths=args.h0_paths,
        h1_samples=h1_samples,
        n=args.n,
        p_values=args.p_values,
        pfa=args.pfa,
        max_h0_stats=args.max_h0_stats,
        i_col=args.i_col,
        q_col=args.q_col,
        complex_col=args.complex_col,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_thresholds_csv(out_dir / "thresholds.csv", thresholds)
    write_statistics_csv(out_dir / "test_statistics.csv", stats_h0, stats_h1)
    write_pd_csv(out_dir / "pd.csv", pd_values, pfa=args.pfa)
    write_pd_from_thresholds_csv(
        THRESHOLDS_CSV_DEFAULT,
        PD_FROM_THRESHOLDS_CSV_DEFAULT,
        h1_samples=h1_samples,
        n=args.n,
        p_values=args.p_values,
        write_h1_stats_csv=H1_STATS_FROM_RECOMPUTE_CSV_DEFAULT
        if WRITE_H1_STATS_FROM_RECOMPUTE_DEFAULT
        else None,
    )


if __name__ == "__main__":
    main()
