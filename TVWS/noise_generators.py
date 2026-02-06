"""Generate I/Q noise samples for baseband signal synthesis and SDR transmission.

This module provides a focused, reusable API for creating various noise models
including Gaussian (AWGN), Laplacian (AWLN), and impulsive distributions. The
implementations mirror the noise models from ``awln.py`` and
``non_gaussian_noise.py`` but are optimized for hardware use (USRP B210) with
independent I/Q channels, float32 defaults, and flexible variance control per
channel.

Key features:
  - Variance-matched mode: automatically scales to reference I/Q statistics
  - SNR-controlled mode: fair noise power across models for a target SNR
  - Fixed variance mode: allows independent scaling control
  - Complex and real-valued outputs: flexible representation for different use cases
  - Reproducible generation: seed and RNG control for testing
  - Hardware-friendly defaults: float32 output for SDR systems

Example: stream AWGN to USRP B210

>>> import uhd
>>> from sink.generators.noise_generators import generate_complex_noise
>>> usrp = uhd.usrp.MultiUSRP()  # configure gain, rate, freq as needed
>>> z = generate_complex_noise("awgn", num_samples=10_000, i_variance=0.01)
>>> usrp.get_tx_streamer(uhd.usrp.StreamArgs("fc32")).send(z, timeout=0.1)

Example: match noise to a captured signal's variance

>>> i_ref, q_ref = captured_i, captured_q
>>> noise_i, noise_q = generate_iq_noise(
...     "laplacian", len(i_ref), i_reference=i_ref, q_reference=q_ref, match_variance=True
... )

Example: fixed Gaussian-impulsive noise with specific variance

>>> noise_i, noise_q = generate_iq_noise(
...     "gaussian_impulsive", len(i_ref),
...     match_variance=False, i_variance=0.05, q_variance=0.05, seed=1
... )
"""

import math
from typing import Callable, Mapping, Optional

import numpy as np

NoisePair = tuple  # Represents (i_noise, q_noise) array pair


def _resolve_rng(rng, seed):
    """Return the active random number generator for reproducible sampling.

    Args:
        rng: Existing np.random.Generator. If provided, returned as-is
            (takes precedence over seed).
        seed: Seed value. If rng is None, creates a new Generator seeded with
            this value.

    Returns:
        The np.random.Generator to use for all noise generation.
    """
    return rng if rng is not None else np.random.default_rng(seed)


def _validate_variance(value, name):
    """Validate a variance value (non-negative and finite)."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_sigma2(sigma2):
    """Validate complex noise power sigma2 (non-negative and finite)."""
    if not math.isfinite(sigma2):
        raise ValueError("sigma2 must be finite")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative")
    return sigma2


def _pair_variances(i_variance, q_variance):
    """Validate and pair I/Q variance values.

    Ensures both variances are non-negative and defaults Q variance to I if
    not explicitly provided.

    Args:
        i_variance: Variance for the I (in-phase) channel.
        q_variance: Variance for the Q (quadrature) channel; defaults to
            i_variance if None.

    Returns:
        Tuple of validated (i_variance, q_variance) pair.

    Raises:
        ValueError: If either variance is negative or non-finite.
    """
    _validate_variance(i_variance, "i_variance")
    if q_variance is None:
        q_variance = i_variance
    _validate_variance(q_variance, "q_variance")
    return i_variance, q_variance


def _resolve_variances(
    num_samples,
    i_variance,
    q_variance,
    *,
    sigma2: Optional[float] = None,
    i_reference=None,
    q_reference=None,
    match_variance=False,
    target_snr_db: Optional[float] = None,
):
    """Resolve variances for fixed, matched, or SNR-controlled modes."""
    if match_variance:
        if i_reference is None or q_reference is None:
            raise ValueError(
                "match_variance=True requires both i_reference and q_reference"
            )
        i_ref = np.asarray(i_reference, dtype=np.float64)
        q_ref = np.asarray(q_reference, dtype=np.float64)
        if i_ref.shape != q_ref.shape:
            raise ValueError("i_reference and q_reference must have the same shape")
        num_samples = i_ref.size
        if target_snr_db is not None:
            if not math.isfinite(target_snr_db):
                raise ValueError("target_snr_db must be finite")
            # sigma2 = Ps * 10^(-SNR/10), Var(I)=Var(Q)=sigma2/2
            ps = float(np.mean(i_ref * i_ref + q_ref * q_ref))
            sigma2 = ps * (10.0 ** (-target_snr_db / 10.0))
            i_variance = sigma2 / 2.0
            q_variance = sigma2 / 2.0
        else:
            i_variance = float(np.var(i_ref))
            q_variance = float(np.var(q_ref))
    else:
        # Fixed variance mode: sigma2 overrides i/q variances
        if sigma2 is not None:
            sigma2 = _validate_sigma2(sigma2)
            i_variance = sigma2 / 2.0
            q_variance = sigma2 / 2.0
        elif q_variance is None:
            q_variance = i_variance
    i_variance, q_variance = _pair_variances(i_variance, q_variance)
    return num_samples, i_variance, q_variance


def generate_iq_gaussian(
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    sigma2: Optional[float] = None,
    rng=None,
    seed=None,
    dtype=np.float32,
):
    """Generate additive white Gaussian noise (AWGN) for I and Q channels.

    Produces zero-mean, independent Gaussian samples for in-phase (I) and
    quadrature (Q) channels with per-channel variance control.

    Args:
        num_samples: Number of samples to generate for each channel.
        i_variance: Target variance of the I channel (default: 1.0).
        q_variance: Target variance of the Q channel; defaults to i_variance
            if not specified.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided,
            overrides i_variance/q_variance with circular noise:
            Var(I)=Var(Q)=sigma2/2.
        rng: Reusable np.random.Generator for reproducibility. Takes precedence
            over seed parameter.
        seed: Seed for reproducibility when rng is None.
        dtype: Output array data type (default: float32 for SDR hardware).

    Returns:
        Tuple of (i_noise, q_noise) as float arrays of shape (num_samples,).
    """
    if sigma2 is not None:
        sigma2 = _validate_sigma2(sigma2)
        i_var = sigma2 / 2.0
        q_var = sigma2 / 2.0
    else:
        i_var, q_var = _pair_variances(i_variance, q_variance)
    rng = _resolve_rng(rng, seed)
    noise_i = rng.normal(loc=0.0, scale=math.sqrt(i_var), size=num_samples)
    noise_q = rng.normal(loc=0.0, scale=math.sqrt(q_var), size=num_samples)
    return noise_i.astype(dtype, copy=False), noise_q.astype(dtype, copy=False)


def generate_iq_laplacian(
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    sigma2: Optional[float] = None,
    rng=None,
    seed=None,
    dtype=np.float32,
):
    """Generate additive white Laplacian noise (AWLN) for I and Q channels.

    Produces zero-mean, independent Laplacian (double-exponential) samples
    with per-channel variance. This distribution has sharper peaks and heavier
    tails than Gaussian noise.

    Args:
        num_samples: Number of samples to generate for each channel.
        i_variance: Target variance of the I channel (default: 1.0).
        q_variance: Target variance of the Q channel; defaults to i_variance
            if not specified.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided,
            overrides i_variance/q_variance with circular noise:
            Var(I)=Var(Q)=sigma2/2.
        rng: Reusable np.random.Generator for reproducibility.
        seed: Seed for reproducibility when rng is None.
        dtype: Output array data type (default: float32).

    Returns:
        Tuple of (i_noise, q_noise) with Laplacian distribution.
    """
    if sigma2 is not None:
        sigma2 = _validate_sigma2(sigma2)
        i_var = sigma2 / 2.0
        q_var = sigma2 / 2.0
    else:
        i_var, q_var = _pair_variances(i_variance, q_variance)
    rng = _resolve_rng(rng, seed)
    scale_i = math.sqrt(i_var / 2.0) if i_var > 0 else 0.0
    scale_q = math.sqrt(q_var / 2.0) if q_var > 0 else 0.0
    noise_i = rng.laplace(loc=0.0, scale=scale_i, size=num_samples)
    noise_q = rng.laplace(loc=0.0, scale=scale_q, size=num_samples)
    return noise_i.astype(dtype, copy=False), noise_q.astype(dtype, copy=False)


def _gaussian_mixture_channel(num_samples, sigma1_sq, sigma2_sq, epsilon, rng, mask=None):
    """Generate a two-term Gaussian mixture for a single channel."""
    if sigma1_sq < 0 or sigma2_sq < 0:
        raise ValueError("sigma1_sq and sigma2_sq must be non-negative")
    if mask is None:
        mask = rng.random(size=num_samples) < epsilon
    base = rng.normal(scale=math.sqrt(sigma1_sq), size=num_samples)
    if epsilon > 0:
        base[mask] = rng.normal(scale=math.sqrt(sigma2_sq), size=mask.sum())
    return base, mask


def _scale_to_variance(noise, target_var, var_mix, dtype):
    """Scale noise to the target variance using the theoretical mixture variance."""
    if target_var == 0:
        return np.zeros_like(noise, dtype=dtype)
    if var_mix == 0:
        raise ValueError("Cannot scale to non-zero variance when var_mix is 0")
    scale = math.sqrt(target_var / var_mix)
    return (noise * scale).astype(dtype, copy=False)


def generate_iq_gaussian_impulsive_paper(
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    sigma2: Optional[float] = None,
    epsilon=0.05, # original = 0.05
    sigma1_sq=1.0,
    sigma2_sq=100.0,
    joint_iq_impulses=False,
    rng=None,
    seed=None,
    dtype=np.float32,
):
    """Generate impulsive noise using the paper (ε, σ1^2, σ2^2) parameterization.

    The mixture is exactly:
        n ~ (1-ε) N(0, σ1^2) + ε N(0, σ2^2)
    with optional scaling to match target per-channel variances using:
        var_mix = (1-ε)*σ1^2 + ε*σ2^2
    so that Var(I)=i_variance and Var(Q)=q_variance.

    Args:
        num_samples: Number of samples to generate for each channel.
        i_variance: Target variance of the I channel (default: 1.0).
        q_variance: Target variance of the Q channel; defaults to i_variance
            if not specified.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided,
            overrides i_variance/q_variance with circular noise:
            Var(I)=Var(Q)=sigma2/2.
        epsilon: Impulse probability (paper ε), in [0, 1].
        sigma1_sq: Nominal Gaussian variance (paper σ1^2), non-negative.
        sigma2_sq: Impulsive Gaussian variance (paper σ2^2), non-negative.
        joint_iq_impulses: If True, share one impulse mask across I and Q.
        rng: Reusable np.random.Generator for reproducibility.
        seed: Seed for reproducibility when rng is None.
        dtype: Output array data type (default: float32).

    Returns:
        Tuple of (i_noise, q_noise) with paper-parameterized mixture.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    _validate_variance(sigma1_sq, "sigma1_sq")
    _validate_variance(sigma2_sq, "sigma2_sq")
    if sigma2 is not None:
        sigma2 = _validate_sigma2(sigma2)
        i_var = sigma2 / 2.0
        q_var = sigma2 / 2.0
    else:
        i_var, q_var = _pair_variances(i_variance, q_variance)
    rng = _resolve_rng(rng, seed)
    var_mix = (1.0 - epsilon) * sigma1_sq + epsilon * sigma2_sq

    if joint_iq_impulses:
        mask = rng.random(size=num_samples) < epsilon
        noise_i, _ = _gaussian_mixture_channel(
            num_samples, sigma1_sq, sigma2_sq, epsilon, rng, mask=mask
        )
        noise_q, _ = _gaussian_mixture_channel(
            num_samples, sigma1_sq, sigma2_sq, epsilon, rng, mask=mask
        )
    else:
        noise_i, _ = _gaussian_mixture_channel(
            num_samples, sigma1_sq, sigma2_sq, epsilon, rng
        )
        noise_q, _ = _gaussian_mixture_channel(
            num_samples, sigma1_sq, sigma2_sq, epsilon, rng
        )

    noise_i = _scale_to_variance(noise_i, i_var, var_mix, dtype)
    noise_q = _scale_to_variance(noise_q, q_var, var_mix, dtype)
    return noise_i, noise_q


def generate_iq_gaussian_impulsive(
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    sigma2: Optional[float] = None,
    impulse_prob=0.05,
    impulse_scale=10.0,
    rng=None,
    seed=None,
    dtype=np.float32,
):
    """Generate Gaussian mixture noise with rare high-amplitude impulses.

    This is the ratio/scale parameterization (impulse_prob, impulse_scale),
    not the paper's (ε, σ1^2, σ2^2) parameterization. For the paper model,
    use generate_iq_gaussian_impulsive_paper.

    Produces a mixture of two Gaussians: most samples from a low-variance
    base Gaussian, with rare samples (controlled by impulse_prob) drawn from
    a high-variance Gaussian. Models non-Gaussian impulsive noise common in
    real radio environments.

    Args:
        num_samples: Number of samples to generate for each channel.
        i_variance: Target variance of the I channel (default: 1.0).
        q_variance: Target variance of the Q channel; defaults to i_variance
            if not specified.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided,
            overrides i_variance/q_variance with circular noise:
            Var(I)=Var(Q)=sigma2/2.
        impulse_prob: Probability of drawing from the high-variance impulse
            distribution (default: 0.05, i.e., 5% of samples). Must be in [0, 1].
        impulse_scale: Scale factor for impulse standard deviation relative to
            base (default: 10.0). Higher values create more extreme impulses.
        rng: Reusable np.random.Generator for reproducibility.
        seed: Seed for reproducibility when rng is None.
        dtype: Output array data type (default: float32).

    Returns:
        Tuple of (i_noise, q_noise) with Gaussian-impulsive distribution.

    Raises:
        ValueError: If impulse_prob is not in [0, 1].
    """
    if not 0 <= impulse_prob <= 1:
        raise ValueError("impulse_prob must be in [0, 1]")
    if sigma2 is not None:
        sigma2 = _validate_sigma2(sigma2)
        i_var = sigma2 / 2.0
        q_var = sigma2 / 2.0
    else:
        i_var, q_var = _pair_variances(i_variance, q_variance)
    rng = _resolve_rng(rng, seed)

    def _channel(var):
        """Generate Gaussian-impulsive samples for a single channel."""
        if var == 0:
            return np.zeros(num_samples, dtype=dtype)
        # Compute base and impulse scales to achieve target variance
        base = impulse_prob * impulse_scale * impulse_scale + (1.0 - impulse_prob)
        sigma_base = math.sqrt(var / base)
        sigma_imp = impulse_scale * sigma_base
        # Draw base Gaussian for all samples
        noise = rng.normal(scale=sigma_base, size=num_samples)
        # Replace impulse_prob fraction with high-variance samples
        mask = rng.random(size=num_samples) < impulse_prob
        if mask.any():
            noise[mask] = rng.normal(scale=sigma_imp, size=mask.sum())
        return noise

    noise_i = _channel(i_var)
    noise_q = _channel(q_var)
    return noise_i.astype(dtype, copy=False), noise_q.astype(dtype, copy=False)


SUPPORTED_NOISE_TYPES: Mapping[str, Callable[..., NoisePair]] = {
    "gaussian": generate_iq_gaussian,
    "awgn": generate_iq_gaussian,
    "laplacian": generate_iq_laplacian,
    "gaussian_impulsive": generate_iq_gaussian_impulsive,
    "gaussian_impulsive_paper": generate_iq_gaussian_impulsive_paper,
}


def generate_iq_noise(
    noise_type,
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    i_reference=None,
    q_reference=None,
    match_variance=False,
    target_snr_db: Optional[float] = None,
    sigma2: Optional[float] = None,
    rng=None,
    seed=None,
    dtype=np.float32,
    **noise_kwargs,
):
    """Generate I/Q noise with fixed, variance-matched, or SNR-controlled scaling.

    Provides a unified interface for generating various noise distributions
    with either fixed per-channel variance, automatic matching to reference
    I/Q statistics, or SNR-controlled scaling.
    
    Supported noise types and their specific kwargs:
        - "gaussian" or "awgn": No additional kwargs.
        - "laplacian": No additional kwargs.
        - "gaussian_impulsive": impulse_prob (float, default=0.05, range [0,1])
          and impulse_scale (float, default=10.0).
        - "gaussian_impulsive_paper": epsilon (float, default=0.05),
          sigma1_sq (float, default=1.0), sigma2_sq (float, default=100.0),
          joint_iq_impulses (bool, default=False).

    Args:
        noise_type: Name of the noise model to generate. Must be a key in
            SUPPORTED_NOISE_TYPES. Special aliases: "awgn" -> "gaussian".
        num_samples: Number of samples to generate for each channel. Ignored
            if match_variance=True and i_reference is provided.
        i_variance: Target variance of the I channel (default: 1.0). Ignored
            if match_variance=True and target_snr_db is not None.
        q_variance: Target variance of the Q channel; defaults to i_variance
            if not specified. Ignored if match_variance=True and target_snr_db
            is not None.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided and
            match_variance=False, overrides i_variance/q_variance with circular
            noise: Var(I)=Var(Q)=sigma2/2.
        i_reference: Reference I-channel samples for variance matching. If
            provided with match_variance=True, num_samples and i_variance are
            ignored. Must have same shape as q_reference.
        q_reference: Reference Q-channel samples for variance matching. If
            provided with match_variance=True, q_variance is ignored. Must
            have same shape as i_reference.
        match_variance: If True, compute variances from reference arrays
            (requires both i_reference and q_reference). If False, use fixed
            i_variance and q_variance (default: False).
        target_snr_db: If set with match_variance=True, compute noise variance
            from the reference signal power:
            sigma2 = Ps * 10^(-SNR/10), Var(I)=Var(Q)=sigma2/2.
        rng: Reusable np.random.Generator for reproducibility. Takes precedence
            over seed.
        seed: Seed value for reproducibility when rng is None.
        dtype: Output array data type (default: float32).
        **noise_kwargs: Additional parameters passed to the specific generator
            function. See Notes for supported options per noise type.

    Returns:
        Tuple of (i_noise, q_noise) arrays matching the requested distribution.

    Raises:
        ValueError: If noise_type is not in SUPPORTED_NOISE_TYPES, or if
            match_variance=True but reference arrays are missing or have
            mismatched shapes.
    """
    num_samples, i_variance, q_variance = _resolve_variances(
        num_samples,
        i_variance,
        q_variance,
        sigma2=sigma2,
        i_reference=i_reference,
        q_reference=q_reference,
        match_variance=match_variance,
        target_snr_db=target_snr_db,
    )

    if noise_type not in SUPPORTED_NOISE_TYPES:
        raise ValueError(
            f"Unsupported noise type '{noise_type}'. Supported: {list(SUPPORTED_NOISE_TYPES)}"
        )

    gen = SUPPORTED_NOISE_TYPES[noise_type]
    return gen(
        num_samples,
        i_variance,
        q_variance,
        rng=rng,
        seed=seed,
        dtype=dtype,
        **noise_kwargs,
    )


def generate_complex_noise(
    noise_type,
    num_samples,
    i_variance=1.0,
    q_variance=None,
    *,
    i_reference=None,
    q_reference=None,
    match_variance=False,
    target_snr_db: Optional[float] = None,
    sigma2: Optional[float] = None,
    rng=None,
    seed=None,
    dtype=np.complex64,
    **noise_kwargs,
):
    """Generate complex baseband noise with fixed or variance-matched scaling.

    Generates noise using generate_iq_noise and combines it into a complex
    array where the real part is I (in-phase) and the imaginary part is Q
    (quadrature). Useful for direct transmission over SDR systems like the
    USRP B210.

    Supported noise types and their specific kwargs:
        - "gaussian" or "awgn": No additional kwargs.
        - "laplacian": No additional kwargs.
        - "gaussian_impulsive": impulse_prob (float, default=0.05, range [0,1])
            and impulse_scale (float, default=10.0).
        - "gaussian_impulsive_paper": epsilon (float, default=0.05),
            sigma1_sq (float, default=1.0), sigma2_sq (float, default=100.0),
            joint_iq_impulses (bool, default=False).

    Args:
        noise_type: Name of the noise model (see generate_iq_noise for
            supported types).
        num_samples: Number of complex samples to generate. Ignored if
            match_variance=True and i_reference is provided.
        i_variance: Target variance of the real (I) component (default: 1.0).
            Ignored if match_variance=True and target_snr_db is not None.
        q_variance: Target variance of the imaginary (Q) component; defaults to
            i_variance if not specified. Ignored if match_variance=True and
            target_snr_db is not None.
        sigma2: Complex noise power sigma2 = Var(I) + Var(Q). If provided and
            match_variance=False, overrides i_variance/q_variance with circular
            noise: Var(I)=Var(Q)=sigma2/2.
        i_reference: Reference I-channel samples for variance matching. If
            provided with match_variance=True, num_samples and i_variance are
            ignored. Must have same shape as q_reference.
        q_reference: Reference Q-channel samples for variance matching. If
            provided with match_variance=True, q_variance is ignored. Must
            have same shape as i_reference.
        match_variance: If True, compute variances from reference arrays
            (requires both i_reference and q_reference). If False, use fixed
            variances (default: False).
        target_snr_db: If set with match_variance=True, compute noise variance
            from the reference signal power:
            sigma2 = Ps * 10^(-SNR/10), Var(I)=Var(Q)=sigma2/2.
        rng: Reusable np.random.Generator for reproducibility.
        seed: Seed for reproducibility when rng is None.
        dtype: Complex output data type (default: complex64 for USRP B210
            fc32 format). Use complex128 for complex64 internal arithmetic to
            avoid precision loss.
        **noise_kwargs: Additional noise-specific parameters (e.g., df,
            impulse_prob).

    Returns:
        Complex noise array of shape (num_samples,) where z = I + 1j*Q.

    Raises:
        ValueError: If match_variance=True but reference arrays are missing
            or have mismatched shapes, or if noise_type is not supported.
    """
    i_noise, q_noise = generate_iq_noise(
        noise_type,
        num_samples,
        i_variance=i_variance,
        q_variance=q_variance,
        i_reference=i_reference,
        q_reference=q_reference,
        match_variance=match_variance,
        target_snr_db=target_snr_db,
        sigma2=sigma2,
        rng=rng,
        seed=seed,
        dtype=np.float32 if dtype == np.complex64 else np.float64,
        **noise_kwargs,
    )
    return (i_noise + 1j * q_noise).astype(dtype, copy=False)


# def verify_noise_power(
#     noise_type,
#     num_samples=200_000,
#     *,
#     i_reference=None,
#     q_reference=None,
#     match_variance=False,
#     target_snr_db: Optional[float] = None,
#     sigma2: Optional[float] = None,
#     rng=None,
#     seed=None,
#     dtype=np.float32,
#     **noise_kwargs,
# ):
#     """Empirically verify noise power for the selected model."""
#     num_samples, i_var, q_var = _resolve_variances(
#         num_samples,
#         1.0,
#         None,
#         sigma2=sigma2,
#         i_reference=i_reference,
#         q_reference=q_reference,
#         match_variance=match_variance,
#         target_snr_db=target_snr_db,
#     )
#     noise_i, noise_q = generate_iq_noise(
#         noise_type,
#         num_samples,
#         i_variance=i_var,
#         q_variance=q_var,
#         rng=rng,
#         seed=seed,
#         dtype=dtype,
#         **noise_kwargs,
#     )

#     emp_i = float(np.var(noise_i))
#     emp_q = float(np.var(noise_q))
#     emp_w2 = float(np.mean(noise_i * noise_i + noise_q * noise_q))

#     print(f"empirical Var(I): {emp_i:.6g}")
#     print(f"empirical Var(Q): {emp_q:.6g}")
#     print(f"empirical E[|w|^2]: {emp_w2:.6g}")

#     if match_variance and target_snr_db is not None and i_reference is not None:
#         i_ref = np.asarray(i_reference, dtype=np.float64)
#         q_ref = np.asarray(q_reference, dtype=np.float64)
#         ps = float(np.mean(i_ref * i_ref + q_ref * q_ref))
#         sigma2 = ps * (10.0 ** (-target_snr_db / 10.0))
#         print(f"target Var(I)=Var(Q): {sigma2 / 2.0:.6g}")
#         print(f"target E[|w|^2]: {sigma2:.6g}")

#     if noise_type == "gaussian_impulsive_paper":
#         epsilon = float(noise_kwargs.get("epsilon", 0.05))
#         sigma1_sq = float(noise_kwargs.get("sigma1_sq", 1.0))
#         sigma2_sq = float(noise_kwargs.get("sigma2_sq", 100.0))
#         rng = _resolve_rng(rng, seed)
#         mask = rng.random(size=num_samples) < epsilon
#         emp_eps = float(mask.mean())
#         var_mix = (1.0 - epsilon) * sigma1_sq + epsilon * sigma2_sq
#         print(f"empirical impulse fraction: {emp_eps:.6g}")
#         print(f"theoretical var_mix: {var_mix:.6g}")


__all__ = [
    "SUPPORTED_NOISE_TYPES",
    "generate_iq_noise",
    "generate_complex_noise",
    "generate_iq_gaussian",
    "generate_iq_laplacian",
    "generate_iq_gaussian_impulsive",
    "generate_iq_gaussian_impulsive_paper",
    # "verify_noise_power",
]
