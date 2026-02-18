from __future__ import annotations

import os
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc_context
from lmfit.models import VoigtModel, LinearModel
from lmfit import Model


# Setup logging
def setup_logging(log_file: Optional[str] = None, verbose: bool = True):
    """Configure logging for the application in an idempotent way."""
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # If there are no handlers, add our defaults
    if not root_logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_h = logging.StreamHandler()
        stream_h.setLevel(level)
        stream_h.setFormatter(fmt)
        root_logger.addHandler(stream_h)

        if log_file:
            try:
                file_h = logging.FileHandler(log_file)
                file_h.setLevel(level)
                file_h.setFormatter(fmt)
                root_logger.addHandler(file_h)
            except Exception:
                # Fallback silently if file handler fails
                pass
    else:
        # Update level on existing handlers
        for h in root_logger.handlers:
            h.setLevel(level)

logger = logging.getLogger(__name__)


@dataclass
class PeakSpec:
    label: str
    center_range: Tuple[float, float]
    # User input: FWHM range (cm^-1) for the Voigt peak. If None: auto/robust guess.
    fwhm_range: Optional[Tuple[float, float]] = None
    # Optional user input: amplitude (area) range for the Voigt peak
    amplitude_range: Optional[Tuple[float, float]] = None

    def mid(self) -> float:
        return 0.5 * (float(self.center_range[0]) + float(self.center_range[1]))

    def mid_fwhm(self) -> Optional[float]:
        if self.fwhm_range is None:
            return None
        return 0.5 * (float(self.fwhm_range[0]) + float(self.fwhm_range[1]))


@dataclass
class FitConfig:
    # Instrument / alignment
    instrument_source: str = "Si"         # "Si" or "Manual"
    si_range: Tuple[float, float] = (510.0, 540.0)
    fit_range: Tuple[float, float] = (300.0, 480.0)
    manual_instr_sigma: Optional[float] = None
    align_si: bool = True
    si_target_cm1: float = 520.0

    # Peaks (dynamic)
    peaks: List[PeakSpec] = field(default_factory=list)

    # Modeling options
    enforce_pure_instrument_gauss: bool = True   # If False: allow sample broadening term
    baseline_model: str = "shirley"              # "linear" or "shirley"
    normalize: Optional[str] = "si"              # None, 'si', or 'range'
    normalize_range: Optional[Tuple[float, float]] = None  # Used if normalize=='range'
    verbose: bool = True

    # Output behavior
    save_plots: bool = False
    show_plots: bool = False
    summary_file: str = "fit_summary.csv"
    save_fit_reports: bool = True  # Save detailed fit report per file

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.
        """
        if self.fit_range[0] >= self.fit_range[1]:
            return False, "Fit range minimum must be less than maximum"
        
        if self.si_range[0] >= self.si_range[1]:
            return False, "Si range minimum must be less than maximum"
        
        if self.instrument_source == "Manual" and self.manual_instr_sigma is not None:
            if self.manual_instr_sigma <= 0:
                return False, "Manual instrument sigma must be positive"
        
        if not self.peaks:
            return False, "At least one peak must be defined"
        
        for peak in self.peaks:
            if peak.center_range[0] >= peak.center_range[1]:
                return False, f"Peak '{peak.label}' has invalid center range: {peak.center_range}"
            if peak.fwhm_range is not None:
                if peak.fwhm_range[0] >= peak.fwhm_range[1]:
                    return False, f"Peak '{peak.label}' has invalid FWHM range: {peak.fwhm_range}"
                if peak.fwhm_range[0] < 0:
                    return False, f"Peak '{peak.label}' FWHM minimum must be >= 0"
            if peak.amplitude_range is not None:
                amin, amax = float(peak.amplitude_range[0]), float(peak.amplitude_range[1])
                if amin < 0:
                    return False, f"Peak '{peak.label}' amplitude minimum must be >= 0"
                if amin >= amax:
                    return False, f"Peak '{peak.label}' has invalid amplitude range: {peak.amplitude_range}"
        
        if self.normalize == 'range' and self.normalize_range:
            if self.normalize_range[0] >= self.normalize_range[1]:
                return False, "Normalization range minimum must be less than maximum"
        
        return True, ""


def _log(cfg: FitConfig, *args):
    """Legacy logging function for backward compatibility."""
    if cfg.verbose:
        logger.info(" ".join(str(a) for a in args))


def cumtrapz_np(y, x):
    if len(x) < 2:
        return np.zeros_like(x)
    dx = np.diff(x)
    area_segments = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(area_segments)))


def shirley_background(x, y_meas, y0, y1, max_iter=200, tol=1e-6):
    try:
        max_iter = int(max_iter)
    except Exception:
        max_iter = 200
    try:
        tol = float(tol)
    except Exception:
        tol = 1e-6
    if not np.isfinite(max_iter) or max_iter < 1:
        max_iter = 200
    if not np.isfinite(tol) or tol <= 0:
        tol = 1e-6

    B = np.interp(x, [x[0], x[-1]], [y0, y1]).astype(float)
    if np.all(~np.isfinite(B)) or np.any(~np.isfinite(y_meas)):
        return np.full_like(x, (y0 + y1) / 2.0)

    for _ in range(max_iter):
        diff = y_meas - B
        diff_pos = np.clip(diff, 0, None)

        denom = np.trapz(diff_pos, x)
        if not np.isfinite(denom) or denom <= 1e-18:
            return np.interp(x, [x[0], x[-1]], [y0, y1])

        cum = cumtrapz_np(diff_pos, x)
        B_new = y0 + (y1 - y0) * (cum / denom)

        delta = np.max(np.abs(B_new - B))
        scale = np.max(np.abs(B)) + 1e-12
        B = B_new
        if delta < tol * scale:
            break

    return B


def fit_sigma_from_range(df, fit_range, peak_center_guess=None, label="peak", verbose=True):
    """
    Fit a peak to estimate instrumental broadening.
    
    Returns:
        Tuple of (sigma_inst, lmfit_result, report_text)
    """
    report_lines: List[str] = []
    fr0, fr1 = float(fit_range[0]), float(fit_range[1])
    d = df[(df["wn"] >= fr0) & (df["wn"] <= fr1)].copy().reset_index(drop=True)
    if len(d) < 5:
        msg = f"Warning: Not enough data points in {label} range {fit_range} to fit. Falling back to default sigma=1.0."
        if verbose:
            logger.warning(msg)
        report_lines.append(msg)
        return 1.0, None, "\n".join(report_lines)

    x = d["wn"].values
    y = d["intensity"].values
    p = VoigtModel(prefix='p_')
    b = LinearModel(prefix='b_')
    model = p + b
    params = model.make_params()
    cg = float(x[np.argmax(y)]) if peak_center_guess is None else float(peak_center_guess)
    cg = min(max(cg, fr0 + 0.3), fr1 - 0.3)
    params['p_center'].set(value=cg, min=fr0, max=fr1, vary=True)
    params['p_sigma'].set(value=1.0, min=0.01, max=10.0, vary=True)
    params['p_gamma'].set(value=1.0, min=0.0, max=20.0, vary=True)
    params['p_amplitude'].set(value=max(y) - min(y), min=0, vary=True)
    params['b_intercept'].set(value=float(np.median(y)), vary=True)
    params['b_slope'].set(value=0.0, vary=True)
    result = model.fit(y, params, x=x)
    sigma_inst = float(result.params['p_sigma'].value)

    header = f"\n=== {label} Fit (Instrument Broadening Estimation) ==="
    report_lines.append(header)
    report_lines.append(result.fit_report(min_correl=0.5))
    fwhm_g = 2.0*np.sqrt(2.0*np.log(2.0))*sigma_inst
    report_lines.append(f"Estimated instrumental Gaussian sigma from {label}: {sigma_inst:.6g}")
    report_lines.append(f"Corresponding Gaussian FWHM: {fwhm_g:.6g} cm^-1")

    if verbose:
        logger.info(header)
        logger.debug(result.fit_report(min_correl=0.5))
        logger.info(f"Estimated instrumental Gaussian sigma from {label}: {sigma_inst:.6g}")
        logger.info(f"Corresponding Gaussian FWHM: {fwhm_g:.6g} cm^-1")

    return sigma_inst, result, "\n".join(report_lines)


def set_peak_sigma(params, prefix, fixed_sigma, enforce_pure_instrument_gauss):
    if enforce_pure_instrument_gauss:
        params[f'{prefix}sigma'].set(value=fixed_sigma, vary=False)
    else:
        params.add(f'{prefix}sigma_sample', value=0.2, min=0.0, max=10.0, vary=True)
        params[f'{prefix}sigma'].set(expr=f'sqrt(sigma_instr**2 + {prefix}sigma_sample**2)')


@lru_cache(maxsize=128)
def voigt_fwhm(sigma_g: float, gamma_l: float) -> float:
    """
    Calculate Voigt profile FWHM (cached for performance).
    
    Args:
        sigma_g: Gaussian sigma parameter
        gamma_l: Lorentzian gamma parameter
        
    Returns:
        Full width at half maximum
    """
    fwhm_g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_g
    fwhm_l = 2.0 * gamma_l
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def gamma_from_fwhm(sigma_g: float, fwhm_target: float) -> float:
    """
    Invert the Voigt FWHM approximation to estimate gamma (Lorentzian HWHM)
    from target FWHM and a given Gaussian sigma.

    Uses: w ≈ 0.5346*L + sqrt(0.2166*L^2 + G^2), where L=2*gamma, G=fwhm_g.
    Selects the smaller positive root (physical solution).
    """
    if fwhm_target <= 0 or not np.isfinite(fwhm_target):
        return 0.0

    # Gaussian FWHM
    G = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_g
    # If target width is not larger than the Gaussian width, no Lorentzian is needed
    if fwhm_target <= G:
        return 0.0

    a = 0.5346
    b = 0.2166
    c1 = a * a - b  # ≈ 0.0692

    # Quadratic in L: c1*L^2 - 2*a*w*L + (w^2 - G^2) = 0
    w = float(fwhm_target)
    A = c1
    B = -2.0 * a * w
    C = w * w - G * G

    disc = B * B - 4.0 * A * C
    if disc < 0:
        disc = 0.0  # numerical safety

    sqrt_disc = np.sqrt(disc)
    # Two candidate roots
    L_plus = (-B + sqrt_disc) / (2.0 * A)
    L_minus = (-B - sqrt_disc) / (2.0 * A)

    # Choose the smaller positive root
    candidates = [val for val in (L_plus, L_minus) if np.isfinite(val) and val > 0]
    if not candidates:
        return 0.0

    L = min(candidates)
    gamma = 0.5 * L
    return float(max(0.0, gamma))


def local_amplitude_guess(x: np.ndarray,
                          y: np.ndarray,
                          rmin: float,
                          rmax: float,
                          fwhm_mid: float,
                          sigma_g_est: float) -> float:
    """
    Estimate a reasonable Voigt 'amplitude' (area) from local data.

    Strategy:
    - Use a local window [rmin, rmax] (slightly expanded if too few points).
    - Robust baseline = 20th percentile; local height = max - baseline.
    - Convert height to area using an effective FWHM estimate:
        fwhm_est = max(Gaussian_FWHM, fwhm_mid)
      where Gaussian_FWHM = 2*sqrt(2*ln2)*sigma_g_est.

    Returns:
        A positive area guess suitable for lmfit VoigtModel amplitude.
    """
    # Ensure numeric
    rmin, rmax = float(rmin), float(rmax)
    if rmin > rmax:
        rmin, rmax = rmax, rmin

    mask = (x >= rmin) & (x <= rmax)
    # If very few points, gently expand the window
    if np.count_nonzero(mask) < 5:
        width = max(0.1, rmax - rmin)
        expand = 0.25 * width + 0.25  # add at least ~0.25 cm^-1
        mask = (x >= rmin - expand) & (x <= rmax + expand)

    seg = y[mask]
    if seg.size < 3:
        # Fallback to a global conservative guess
        global_span = float(np.max(y) - np.min(y))
        return max(1.0, 0.1 * global_span)

    baseline = float(np.percentile(seg, 20.0))
    height = float(np.max(seg) - baseline)
    if not np.isfinite(height) or height <= 0:
        # fallback to variability
        height = float(np.std(seg))
        height = max(height, 0.1 * (np.max(y) - np.median(y)))

    # Effective FWHM for area conversion
    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(sigma_g_est)
    fwhm_est = max(gaussian_fwhm, float(fwhm_mid))
    # area ~ height * width (order-of-magnitude), guardrails applied
    area_guess = height * max(0.5, fwhm_est)

    # Clamp to sane limits to avoid exploding guesses
    global_span = float(np.max(y) - np.min(y))
    max_area = 50.0 * global_span * max(1.0, fwhm_est)
    area_guess = float(np.clip(area_guess, 1e-6, max_area))
    return area_guess


def robust_fwhm_guess(x: np.ndarray,
                      y: np.ndarray,
                      rmin: float,
                      rmax: float,
                      sigma_g_est: float) -> float:
    """
    Robustly estimate a peak FWHM from local data window [rmin, rmax].
    - Local baseline: 20th percentile
    - Peak height: max - baseline
    - Half-max crossing points found by linear interpolation on each side of the peak
    Fallback to a conservative value if the estimate is ill-conditioned.

    Returns: FWHM in cm^-1 (>= Gaussian FWHM).
    """
    rmin, rmax = float(rmin), float(rmax)
    if rmin > rmax:
        rmin, rmax = rmax, rmin

    mask = (x >= rmin) & (x <= rmax)
    # Expand window if too few points
    if np.count_nonzero(mask) < 7:
        width = max(0.1, rmax - rmin)
        expand = 0.5 * width + 0.25
        mask = (x >= rmin - expand) & (x <= rmax + expand)

    xin = x[mask]
    yin = y[mask]
    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(sigma_g_est)

    if xin.size < 5:
        return float(max(gaussian_fwhm, 8.0))

    baseline = float(np.percentile(yin, 20.0))
    ymax = float(np.max(yin))
    if not np.isfinite(ymax):
        return float(max(gaussian_fwhm, 8.0))

    height = ymax - baseline
    if height <= 1e-12:
        return float(max(gaussian_fwhm, 8.0))

    half = baseline + 0.5 * height
    imax = int(np.argmax(yin))

    # Left crossing
    x_left = None
    for i in range(imax, 0, -1):
        if yin[i] >= half and yin[i-1] <= half:
            # linear interp between (i-1,i)
            t = (half - yin[i-1]) / max(1e-18, (yin[i] - yin[i-1]))
            x_left = float(xin[i-1] + t * (xin[i] - xin[i-1]))
            break
    # Right crossing
    x_right = None
    for i in range(imax, xin.size - 1):
        if yin[i] >= half and yin[i+1] <= half:
            t = (half - yin[i+1]) / max(1e-18, (yin[i] - yin[i+1]))
            x_right = float(xin[i+1] + t * (xin[i] - xin[i+1]))
            break

    if x_left is None or x_right is None or not np.isfinite(x_left) or not np.isfinite(x_right):
        return float(max(gaussian_fwhm, 8.0))

    fwhm_est = float(max(gaussian_fwhm, x_right - x_left))
    return fwhm_est


def estimate_peak_center_parabolic(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Estimate peak center using 3-point parabolic interpolation around the max."""
    if len(x) < 3:
        return None
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:
        delta = 0.0
    dx = 0.5 * (x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)])
    return float(x[i] + delta * dx)


def estimate_si_center(df, si_range):
    """Estimate Si center using parabolic sub-sample interpolation within the range."""
    sr0, sr1 = float(si_range[0]), float(si_range[1])
    d = df[(df["wn"] >= sr0) & (df["wn"] <= sr1)]
    if d.empty:
        return None
    x = d["wn"].values
    y = d["intensity"].values
    c = estimate_peak_center_parabolic(x, y)
    if c is None or not np.isfinite(c):
        # Fallback to simple max
        i = int(np.argmax(y))
        return float(x[i])
    return float(c)


def sanitize_label(label: str) -> str:
    if not label:
        return "Peak"
    safe = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
    if safe == "":
        safe = "Peak"
    return safe


def load_spectrum(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate a Raman spectrum file.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {file_path}")
    
    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=["wn", "intensity"],
            comment="#",
            dtype=str,
            engine="python",
        )
    except Exception as e:
        raise ValueError(f"Failed to read spectrum file {file_path}: {str(e)}")
    
    df = df.apply(lambda s: s.str.strip())
    for col in ["wn", "intensity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["wn", "intensity"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} non-numeric row(s) in {file_path.name}")
    
    if len(df) < 5:
        raise ValueError(f"Insufficient data in {file_path}: only {len(df)} valid rows")
    
    return df.sort_values("wn").reset_index(drop=True)


def process_file(file_path: Union[str, Path], cfg: FitConfig) -> Dict[str, any]:
    """
    Process a single Raman spectrum file.
    """
    # Validate configuration
    is_valid, error_msg = cfg.validate()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {error_msg}")
    
    report_lines: List[str] = []
    report_lines.append(f"Processing file: {file_path}")
    report_lines.append("==============================")

    logger.info("=" * 60)
    logger.info(f"Processing file: {file_path}")
    logger.info("=" * 60)

    df_full = load_spectrum(file_path)

    # Ensure numeric ranges
    cfg.si_range = (float(cfg.si_range[0]), float(cfg.si_range[1]))
    cfg.fit_range = (float(cfg.fit_range[0]), float(cfg.fit_range[1]))
    if cfg.normalize == 'range' and cfg.normalize_range is not None:
        cfg.normalize_range = (float(cfg.normalize_range[0]), float(cfg.normalize_range[1]))

    # Align axis using Si peak if requested
    if cfg.align_si:
        si_center_raw = estimate_si_center(df_full, cfg.si_range)
        if si_center_raw is None:
            msg = f"Align to Si requested but no data found in range {cfg.si_range[0]:.3f}-{cfg.si_range[1]:.3f} cm^-1 — skipping alignment."
            logger.warning(msg)
            report_lines.append(msg)
        else:
            shift_applied = float(cfg.si_target_cm1 - si_center_raw)
            if np.isfinite(shift_applied) and abs(shift_applied) > 0:
                align_msg = (
                    f"Aligning Si peak (max in {cfg.si_range[0]:.3f}-{cfg.si_range[1]:.3f} cm^-1): "
                    f"raw center ~ {si_center_raw:.3f} cm^-1; applying shift {shift_applied:+.3f} cm^-1 "
                    f"-> {cfg.si_target_cm1:.1f} cm^-1"
                )
                logger.info(align_msg)
                report_lines.append(align_msg)
                df_full['wn'] = df_full['wn'] + shift_applied
            else:
                msg = "Align to Si requested, but no shift necessary."
                logger.info(msg)
                report_lines.append(msg)
    else:
        msg = "Si alignment disabled."
        logger.info(msg)
        report_lines.append(msg)

    # Instrument sigma estimation or manual
    instr_fit_result = None
    post_si_report_text = ""
    if cfg.instrument_source.lower() == "si":
        sigma_instrument, instr_fit_result, post_si_report_text = fit_sigma_from_range(
            df_full, cfg.si_range, peak_center_guess=cfg.si_target_cm1,
            label=f"Si (~{cfg.si_target_cm1:.1f} cm$^{{-1}}$) [post-shift]", verbose=cfg.verbose
        )
        if post_si_report_text:
            report_lines.append(post_si_report_text)
    elif cfg.instrument_source.lower() == "manual":
        if cfg.manual_instr_sigma is not None and np.isfinite(cfg.manual_instr_sigma) and cfg.manual_instr_sigma > 0:
            sigma_instrument = float(cfg.manual_instr_sigma)
            msg = f"Using manual instrument sigma: {sigma_instrument:.6g} cm^-1"
            logger.info(msg)
            report_lines.append(msg)
        else:
            logger.warning("Manual instrument sigma invalid; falling back to 1.0 cm^-1")
            report_lines.append("Manual instrument sigma invalid; falling back to 1.0 cm^-1")
            sigma_instrument = 1.0
    else:
        raise ValueError("instrument_source must be 'Si' or 'Manual'")

    fixed_sigma = float(sigma_instrument if np.isfinite(sigma_instrument) and sigma_instrument > 0 else 1.0)

    # Restrict to overall fit range
    fr0, fr1 = cfg.fit_range
    df = df_full[(df_full["wn"] >= fr0) & (df_full["wn"] <= fr1)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No data points within specified fit range {cfg.fit_range}")
    x = df["wn"].values
    y = df["intensity"].values
    baseline_span = max(y) - min(y)

    # Build dynamic model
    model = None
    peak_prefixes = []
    for i, spec in enumerate(cfg.peaks):
        prefix = f'pk{i}_'
        vm = VoigtModel(prefix=prefix)
        model = vm if model is None else (model + vm)
        peak_prefixes.append((prefix, spec))

    # Baseline
    if cfg.baseline_model.lower() == 'shirley':
        baseline = Model(shirley_background, independent_vars=['x', 'y_meas'], prefix='b_')
    else:
        baseline = LinearModel(prefix='b_')
    model = model + baseline

    params = model.make_params()
    params.add('sigma_instr', value=fixed_sigma, vary=False)

    # Initialize peak params
    n_peaks = max(len(cfg.peaks), 1)

    # For Fit mode (sample broadening allowed), initial total Gaussian sigma is a bit larger
    # to reflect the starting sigma_sample=0.2 used below.
    sigma_eff_init = fixed_sigma if cfg.enforce_pure_instrument_gauss else float(np.sqrt(fixed_sigma**2 + 0.2**2))

    for i, (prefix, spec) in enumerate(peak_prefixes):
        rmin, rmax = spec.center_range
        init_center = spec.mid()
        params[f'{prefix}center'].set(value=init_center, min=float(rmin), max=float(rmax), vary=True)
        set_peak_sigma(params, prefix, fixed_sigma, cfg.enforce_pure_instrument_gauss)

        # FWHM handling
        if spec.fwhm_range is not None:
            # Respect user-provided range
            fmin, fmax = float(spec.fwhm_range[0]), float(spec.fwhm_range[1])
            if fmin > fmax:
                fmin, fmax = fmax, fmin
            fwhm_mid = 0.5 * (fmin + fmax)

            # Convert to gamma bounds
            gmin = gamma_from_fwhm(sigma_eff_init, fmin)
            gmax = gamma_from_fwhm(sigma_eff_init, fmax)
            if gmin > gmax:
                gmin, gmax = gmax, gmin
            gmin = max(0.0, gmin)

            # Centered initial gamma within finite bounds
            g_init = 0.5 * (gmin + gmax)
            if not np.isfinite(g_init):
                g_init = max(0.01, gmin)
            eps = max(1e-6, 1e-3 * (gmax - gmin))
            g_init = min(max(g_init, gmin + eps), max(gmin + eps, gmax))

            params[f'{prefix}gamma'].set(value=float(g_init), min=float(gmin), max=float(gmax), vary=True)
        else:
            # Blank FWHM -> robust guess, but NO upper bound on gamma
            fwhm_guess = robust_fwhm_guess(x, y, rmin, rmax, sigma_eff_init)
            fwhm_mid = fwhm_guess
            g_guess = gamma_from_fwhm(sigma_eff_init, fwhm_guess)
            if not np.isfinite(g_guess) or g_guess < 0:
                g_guess = 0.5  # safe small start
            params[f'{prefix}gamma'].set(value=float(g_guess), min=0.0, max=None, vary=True)

        # Local, data-driven amplitude (area) guess and bounds
        amp_guess = local_amplitude_guess(x, y, rmin, rmax, fwhm_mid, sigma_eff_init)

        if spec.amplitude_range is not None:
            amin, amax = float(spec.amplitude_range[0]), float(spec.amplitude_range[1])
            if amin > amax:
                amin, amax = amax, amin
            amin = max(0.0, amin)
            if amax <= amin:
                amax = amin + max(amp_guess, 1e-3)
        else:
            # Wide, data-driven default upper bound
            global_span = float(np.max(y) - np.min(y))
            default_max = 50.0 * global_span * max(1.0, fwhm_mid)
            amin, amax = 0.0, max(default_max, 5.0 * amp_guess)

        eps_a = max(1e-9, 1e-3 * (amax - amin))
        amp_start = float(np.clip(amp_guess, amin + eps_a, amax - eps_a))
        params[f'{prefix}amplitude'].set(value=amp_start, min=amin, max=amax, vary=True)

    # Baseline params and fit
    if cfg.baseline_model.lower() == 'shirley':
        params['b_y0'].set(value=float(y[0]), min=0, vary=True)
        params['b_y1'].set(value=float(y[-1]), min=0, vary=True)
        params['b_max_iter'].set(value=200, vary=True)
        params['b_tol'].set(value=1e-8, vary=False)
        result = model.fit(y, params, x=x, y_meas=y)
    else:
        params['b_intercept'].set(value=float(np.median(y)), vary=True)
        params['b_slope'].set(value=0.0, vary=True)
        result = model.fit(y, params, x=x)

    logger.info("\n=== Dynamic Peaks Fit ===")
    logger.info(f"Instrument sigma (Gaussian) fixed value: {fixed_sigma:.6g}")
    if cfg.verbose:
        logger.debug(result.fit_report(min_correl=0.5))

    report_lines.append("\n=== Dynamic Peaks Fit ===")
    report_lines.append(f"Instrument sigma (Gaussian) fixed value: {fixed_sigma:.6g}")
    report_lines.append(result.fit_report(min_correl=0.5))

    comps = result.eval_components(x=x, y_meas=y) if cfg.baseline_model.lower() == 'shirley' else result.eval_components(x=x)

    # Determine baseline component key robustly
    baseline_key = None
    if isinstance(comps, dict):
        if 'b_' in comps:
            baseline_key = 'b_'
        else:
            for k in comps.keys():
                if k.startswith('b_'):
                    baseline_key = k
                    break

    # Normalization logic
    scale = 1.0
    norm_report_header_added = False

    def add_norm_msg(msg: str):
        nonlocal norm_report_header_added
        if not norm_report_header_added:
            report_lines.append("\n=== Normalization ===")
            norm_report_header_added = True
        report_lines.append(msg)
        logger.info(msg)

    if cfg.normalize is not None:
        if cfg.normalize.lower() == 'si':
            dsi = df_full[(df_full["wn"] >= cfg.si_range[0]) & (cfg.si_range[1] >= df_full["wn"])]
            dsi = df_full[(df_full["wn"] >= cfg.si_range[0]) & (df_full["wn"] <= cfg.si_range[1])]
            si_val = float(dsi["intensity"].max()) if not dsi.empty else None
            if (si_val is None or not np.isfinite(si_val) or si_val <= 0) and instr_fit_result is not None:
                try:
                    si_val = float(instr_fit_result.params.get('p_amplitude').value)
                except Exception:
                    si_val = None
            if si_val is not None and np.isfinite(si_val) and si_val > 0:
                scale = si_val
                add_norm_msg(f"Normalizing to Si peak: dividing intensities by {scale:.6g}")
            else:
                add_norm_msg("Requested normalization to Si but could not determine a valid Si peak height — skipping normalization.")
        elif cfg.normalize.lower() == 'range':
            if cfg.normalize_range is not None:
                nr0, nr1 = cfg.normalize_range
                if nr1 < nr0:
                    nr0, nr1 = nr1, nr0
                dnorm = df_full[(df_full["wn"] >= nr0) & (df_full["wn"] <= nr1)]
                if not dnorm.empty:
                    scale_candidate = float(dnorm["intensity"].max())
                    if np.isfinite(scale_candidate) and scale_candidate > 0:
                        scale = scale_candidate
                        add_norm_msg(f"Normalizing to max intensity in range {nr0:.3f}-{nr1:.3f} cm^-1: dividing by {scale:.6g}")
                    else:
                        add_norm_msg(f"Range {nr0:.3f}-{nr1:.3f} produced non-positive scale; skipping normalization.")
                else:
                    add_norm_msg(f"No data points found in normalization range {nr0:.3f}-{nr1:.3f}; skipping normalization.")
            else:
                add_norm_msg("Normalization 'range' selected but no range provided; skipping normalization.")
        else:
            add_norm_msg(f"Unknown normalization option '{cfg.normalize}' ignored.")
    # Apply scaling
    if scale != 1.0:
        y_scaled = y / scale
        best_fit_scaled = result.best_fit / scale
        comps_scaled = {k: v / scale for k, v in comps.items()}
        chosen_norm = cfg.normalize
    else:
        y_scaled = y
        best_fit_scaled = result.best_fit
        comps_scaled = comps
        chosen_norm = "none"

    # Calculate per-peak metrics
    row = {'file_name': os.path.basename(file_path)}
    row['normalization'] = chosen_norm

    # Handle duplicate labels by making unique output names
    used_labels = {}
    label_map = {}
    for prefix, spec in peak_prefixes:
        base_label = spec.label.strip() or "Peak"
        safe_label = sanitize_label(base_label)
        if safe_label in used_labels:
            used_labels[safe_label] += 1
            safe_label = f"{safe_label}_{used_labels[safe_label]}"
        else:
            used_labels[safe_label] = 1
        label_map[prefix] = safe_label

    areas_trapz = {}
    for prefix, spec in peak_prefixes:
        if prefix in comps_scaled:
            areas_trapz[prefix] = np.trapz(comps_scaled[prefix], x)
        else:
            areas_trapz[prefix] = np.nan

    # Add per-peak summary to report
    if peak_prefixes:
        report_lines.append("\n--- Per-peak fitted parameters (scaled if normalized) ---")

    for prefix, spec in peak_prefixes:
        label = label_map[prefix]
        center = float(result.params[f'{prefix}center'].value)
        comp_arr = comps_scaled.get(prefix, np.zeros_like(x))
        intensity = float(np.max(comp_arr))
        area = float(areas_trapz[prefix])
        sigma = float(result.params[f'{prefix}sigma'].value)
        gamma = float(result.params[f'{prefix}gamma'].value)
        fwhm = float(voigt_fwhm(sigma, gamma))

        row[f'{label}_center'] = center
        row[f'{label}_intensity'] = intensity
        row[f'{label}_area'] = area
        row[f'{label}_FWHM'] = fwhm

        report_lines.append(
            f"{label}: center={center:.6g} cm^-1, intensity={intensity:.6g}, area={area:.6g}, FWHM={fwhm:.6g} cm^-1 "
            f"(sigma={sigma:.6g}, gamma={gamma:.6g})"
        )

    # Plot
    with rc_context({'font.family': 'Arial'}):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(cfg.fit_range[0], cfg.fit_range[1])
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.plot(x, y_scaled, 'k', label='Data', lw=1.5)
        ax.plot(x, best_fit_scaled, 'r--', label='Total Fit')

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(peak_prefixes), 1)))
        for (prefix, spec), color in zip(peak_prefixes, colors):
            ax.plot(x, comps_scaled.get(prefix, np.zeros_like(x)),
                    label=spec.label, color=color, linestyle='dotted')

        # Baseline plot using detected key
        ax.plot(x, comps_scaled.get(baseline_key, np.zeros_like(x)),
                label=f'Baseline ({cfg.baseline_model})', color='gray', linestyle='dotted')

        # Shade normalization range if applicable
        if chosen_norm == 'range' and cfg.normalize_range is not None:
            r0, r1 = cfg.normalize_range
            ax.axvspan(min(r0, r1), max(r0, r1), color='yellow', alpha=0.15, label='Normalization range')

        ax.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend()
        plt.tight_layout()

    # Output dataframe
    out_dict = {
        'Wavenumber': x,
        'Data': y_scaled,
        'Fit': best_fit_scaled,
        'Baseline': comps_scaled.get(baseline_key, np.zeros_like(x)),
    }
    for prefix, spec in peak_prefixes:
        out_dict[spec.label] = comps_scaled.get(prefix, np.zeros_like(x))

    out_df = pd.DataFrame(out_dict)

    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    report_text = "\n".join(report_lines)
    return {
        "row": row,
        "fig": fig,
        "dataframe": out_df,
        "input_filename": input_filename,
        "report_text": report_text,
    }


def export_results(dataframe: pd.DataFrame, output_path: Union[str, Path], format: str = 'tsv'):
    """
    Export fit results in various formats.
    """
    output_path = Path(output_path)
    
    if format == 'tsv':
        dataframe.to_csv(output_path, sep='\t', index=False)
    elif format == 'csv':
        dataframe.to_csv(output_path, index=False)
    elif format == 'excel':
        dataframe.to_excel(output_path, index=False)
    elif format == 'json':
        dataframe.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    logger.info(f"Exported results to {output_path} (format: {format})")


def save_per_file_outputs(result: Dict, directory: str, cfg: Optional[FitConfig] = None):
    """
    Save the per-file TSV of data and (optionally) a detailed text report.
    """
    input_filename = result["input_filename"]
    out_df: pd.DataFrame = result["dataframe"]
    out_file = os.path.join(directory, f"{input_filename}_fit_plot.txt")
    out_df.to_csv(out_file, sep='\t', index=False)
    logger.info(f"Saved fit data to {out_file}")

    if getattr(cfg, "save_fit_reports", True):
        report_text = result.get("report_text", "")
        if isinstance(report_text, str) and report_text.strip():
            out_report = os.path.join(directory, f"{input_filename}_fit_report.txt")
            with open(out_report, "w", encoding="utf-8") as fh:
                fh.write(report_text)
            logger.info(f"Saved fit report to {out_report}")

    return out_file


def run_batch(directory: str, cfg: FitConfig, 
              progress_callback: Optional[Callable[[int, int, str], None]] = None,
              cancel_event: Optional[threading.Event] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Process all .txt files in a directory.
    """
    all_txt = sorted(glob(os.path.join(directory, "*.txt")))
    files = [f for f in all_txt if not f.endswith("_fit_lt.txt")]
    if not files:
        logger.warning("No input .txt files found to process.")
        return pd.DataFrame(), [], []

    logger.info(f"Processing batch of {len(files)} files in {directory}")
    
    summary_rows = []
    saved_txt, saved_plots = [], []

    for idx, f in enumerate(files):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Batch processing cancelled by user.")
            break

        if progress_callback:
            try:
                progress_callback(idx, len(files), os.path.basename(f))
            except Exception:
                # Keep batch running even if UI callback fails
                pass
        
        try:
            result = process_file(f, cfg)
            summary_rows.append(result["row"])
            out_txt = save_per_file_outputs(result, directory, cfg)
            saved_txt.append(out_txt)
            if cfg.save_plots:
                fig = result["fig"]
                out_plot = os.path.join(directory, f"{result['input_filename']}_fit.png")
                fig.savefig(out_plot, dpi=200, bbox_inches='tight')
                saved_plots.append(out_plot)
                logger.info(f"Saved plot to {out_plot}")
            plt.close(result["fig"])
        except Exception as e:
            logger.error(f"Error processing {f}: {e}", exc_info=True)

    if summary_rows:
        columns = ['file_name', 'normalization']
        label_set = []
        for r in summary_rows:
            for k in r.keys():
                if k.endswith('_center'):
                    base = k[:-7]
                    label_set.append(base)
        # preserve order
        seen = set()
        ordered_labels = []
        for lbl in label_set:
            if lbl not in seen:
                seen.add(lbl)
                ordered_labels.append(lbl)

        for lbl in ordered_labels:
            columns.extend([f'{lbl}_center', f'{lbl}_intensity', f'{lbl}_area', f'{lbl}_FWHM'])

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.reindex(columns=columns)
        summary_path = os.path.join(directory, cfg.summary_file)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        return summary_df, saved_txt, saved_plots

    return pd.DataFrame(), saved_txt, saved_plots