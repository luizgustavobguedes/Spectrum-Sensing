#!/usr/bin/env python3
"""Generate I/Q noise, plot samples/histograms, and report noise power."""

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "matplotlib is required for plotting. Install it (e.g., pip install matplotlib)."
    ) from exc

import noise_generators as ng


def _noise_power(i_noise: np.ndarray, q_noise: np.ndarray) -> float:
    return float(np.mean(i_noise * i_noise + q_noise * q_noise))


def _plot_samples(idx, i_noise, q_noise, outdir: Path, tag: str) -> None:
    plt.figure(figsize=(9, 4.2))
    plt.plot(idx, i_noise, label="I", linewidth=1.0)
    plt.plot(idx, q_noise, label="Q", linewidth=1.0)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(f"Noise samples vs index ({tag})")
    plt.ylim(-12, 12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"samples_{tag}.png", dpi=140)
    plt.close()


def _plot_histograms(i_noise, q_noise, outdir: Path, tag: str) -> None:
    plt.figure(figsize=(9, 4.2))
    plt.hist(i_noise, bins=120, density=True, alpha=0.6, label="I")
    plt.hist(q_noise, bins=120, density=True, alpha=0.6, label="Q")
    plt.xlabel("Amplitude")
    plt.ylabel("PDF (normalized)")
    plt.title(f"Histogram of I/Q ({tag})")
    plt.xlim(-3, 3)
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"hist_{tag}.png", dpi=140)
    plt.close()


def _summarize(i_noise, q_noise, tag: str) -> None:
    var_i = float(np.var(i_noise))
    var_q = float(np.var(q_noise))
    pwr = _noise_power(i_noise, q_noise)
    print(f"[{tag}] Var(I)={var_i:.6g}, Var(Q)={var_q:.6g}, E[|w|^2]={pwr:.6g}")


def run_one(noise_type: str, num_samples: int, sigma2: float, outdir: Path) -> None:
    tag = noise_type
    kwargs = {}
    if noise_type == "gaussian_impulsive_paper":
        kwargs = {"epsilon": 0.05, "sigma1_sq": 1.0, "sigma2_sq": 100.0}
    elif noise_type == "gaussian_impulsive":
        kwargs = {"impulse_prob": 0.05, "impulse_scale": 10.0}

    i_noise, q_noise = ng.generate_iq_noise(
        noise_type,
        num_samples,
        sigma2=sigma2,
        **kwargs,
    )
    idx = np.arange(num_samples)
    _plot_samples(idx, i_noise, q_noise, outdir, tag)
    _plot_histograms(i_noise, q_noise, outdir, tag)
    _summarize(i_noise, q_noise, tag)
    return i_noise, q_noise


def _plot_i_hist_overlay(
    data_map: dict,
    outdir: Path,
    tag: str,
    xlim,
    ylim,
) -> None:
    plt.figure(figsize=(9, 4.2))
    for name, i_noise in data_map.items():
        plt.hist(
            i_noise,
            bins=140,
            density=True,
            alpha=0.35,
            histtype="stepfilled",
            linewidth=1.0,
            label=name,
        )
    plt.xlabel("Amplitude (I)")
    plt.ylabel("PDF (normalized)")
    plt.title(f"I-part histogram overlay ({tag})")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"hist_i_overlay_{tag}.png", dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test noise generation.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Number of samples per noise type.",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=1.0,
        help="Complex noise power sigma2 = Var(I) + Var(Q).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("noise_plots"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.sigma2) or args.sigma2 < 0:
        raise SystemExit("sigma2 must be non-negative and finite.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    noise_types = [
        "gaussian",
        "laplacian",
        "gaussian_impulsive",
        "gaussian_impulsive_paper",
    ]

    results = {}
    for noise_type in noise_types:
        i_noise, _q_noise = run_one(noise_type, args.num_samples, args.sigma2, outdir)
        results[noise_type] = i_noise

    overlay_types = ["gaussian", "laplacian", "gaussian_impulsive_paper"]
    overlay_data = {k: results[k] for k in overlay_types}
    _plot_i_hist_overlay(
        overlay_data,
        outdir,
        tag="full",
        xlim=(-3, 3),
        ylim=(0, 1.2),
    )
    _plot_i_hist_overlay(
        overlay_data,
        outdir,
        tag="left_tail",
        xlim=(-3, -0.5),
        ylim=(0, 1.2),
    )

    print(f"Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
