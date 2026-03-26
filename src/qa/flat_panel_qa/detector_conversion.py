import streamlit as st
import numpy as np
import io
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydicom
from scipy.optimize import least_squares
from src.core.io.raw_endian import frombuffer_with_endian
from src.core.io.analysis_payload import ImagePayload, file_name_and_bytes

"""
Detector conversion feature: Upload RAW/STD files, assign kerma values, extract ROI stats,
fit MPV vs Kerma curves (linear/log/poly), and analyze noise characteristics.
"""

# ==================== HELPER FUNCTIONS ====================

def _read_raw_as_square(raw_bytes, dtype, little_endian=True, auto_endian_from_dicom=True):
    """Read raw bytes and reshape into largest possible square array.

    If a DICOM header with Transfer Syntax UID is present, endian can be auto-detected.
    """
    pixels, endian_used, endian_source = frombuffer_with_endian(
        raw_bytes,
        dtype,
        default_little_endian=little_endian,
        auto_endian_from_dicom=auto_endian_from_dicom,
    )
    side = int(np.floor(np.sqrt(pixels.size)))
    if side < 100:
        raise ValueError(f"Inferred square size {side} < 100; cannot extract 100x100 ROI")
    return pixels[:side*side].reshape((side, side)), endian_used, endian_source


def _extract_dicom_header_hints(file_bytes):
    """Extract optional DICOM header hints from RAW/STD files when present.

    Returns:
        dict with keys:
            - pixel_intensity_relationship_raw: str | None
            - pixel_intensity_relationship_fit_method: 'linear' | 'log' | None
            - relative_xray_exposure: float | None
    """
    hints = {
        "pixel_intensity_relationship_raw": None,
        "pixel_intensity_relationship_fit_method": None,
        "relative_xray_exposure": None,
    }

    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True, stop_before_pixels=True)
    except Exception:
        return hints

    pir = getattr(ds, "PixelIntensityRelationship", None)
    if pir is not None:
        pir_str = str(pir).strip().upper()
        if pir_str:
            hints["pixel_intensity_relationship_raw"] = pir_str
            if "LOG" in pir_str:
                hints["pixel_intensity_relationship_fit_method"] = "log"
            elif "LIN" in pir_str:
                hints["pixel_intensity_relationship_fit_method"] = "linear"

    rel_xray_exp = getattr(ds, "RelativeXRayExposure", None)
    if rel_xray_exp is not None:
        try:
            hints["relative_xray_exposure"] = float(rel_xray_exp)
        except (TypeError, ValueError):
            pass

    return hints


def _select_default_fit_method_from_pir(per_file_fit_hints, fallback_method):
    """Choose default fit method from Pixel Intensity Relationship hints.

    Uses PIR first when present and consistent; otherwise falls back to existing logic.
    """
    valid_hints = [m for m in per_file_fit_hints if m in {"linear", "log"}]
    if not valid_hints:
        return fallback_method

    unique_hints = sorted(set(valid_hints))
    if len(unique_hints) == 1:
        return unique_hints[0]

    st.warning(
        "Conflicting Pixel Intensity Relationship values detected across files. "
        f"Using fallback default fit method: {fallback_method}."
    )
    return fallback_method


def _central_roi_stats(img_array, roi_h=100, roi_w=100):
    """Extract central ROI and compute mean pixel value (MPV) and standard deviation."""
    H, W = img_array.shape
    if roi_h > H or roi_w > W:
        raise ValueError("Image smaller than requested ROI size")
    y0, x0 = (H - roi_h) // 2, (W - roi_w) // 2
    roi = img_array[y0:y0+roi_h, x0:x0+roi_w]
    return float(np.mean(roi)), float(np.std(roi)), roi


def _detrend_roi(roi):
    """Remove planar trend from ROI using first-order surface fit.
    
    Fits a plane Z = ax + by + c to remove systematic effects (heel effect, geometric dome)
    and leaves only random noise (quantum + electronic).
    
    Args:
        roi: 2D array (e.g., 100x100 kerma values)
    
    Returns:
        Detrended 2D array with plane removed
    """
    H, W = roi.shape
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Flatten arrays for fitting
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = roi.flatten()
    
    # Remove NaN/inf values for fitting
    valid_mask = np.isfinite(z_flat)
    if valid_mask.sum() < 3:
        # Not enough valid points, return original
        return roi
    
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    
    # Design matrix: [x, y, 1] for plane Z = ax + by + c
    X = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])
    
    # Fit plane using least squares
    coeffs, _, _, _ = np.linalg.lstsq(X, z_valid, rcond=None)
    a, b, c = coeffs
    
    # Compute fitted plane for all pixels
    plane = a * x_coords + b * y_coords + c
    
    # Subtract plane to get detrended data (pure noise)
    detrended = roi - plane
    
    return detrended


def _detrend_from_area(full_img, area_fraction=0.80):
    """Fit a detrending plane to the central *area_fraction* of the image and
    subtract it from the entire image.

    Using a larger fitting region increases spatial leverage for estimating the
    heel-effect / geometric-dome gradient compared to fitting a small 100\u00d7100 ROI.

    Args:
        full_img: 2D array (e.g. full kerma-domain image).
        area_fraction: Fraction of total image *area* used for fitting (0.1\u20131.0).
            Linear dimension scale = sqrt(area_fraction).

    Returns:
        (detrended_full_img, (a, b, c), (y0, x0, cH, cW))
    """
    H, W = full_img.shape
    if area_fraction >= 1.0:
        cH, cW, y0, x0 = H, W, 0, 0
    else:
        scale = np.sqrt(area_fraction)
        cH = int(np.floor(H * scale))
        cW = int(np.floor(W * scale))
        y0 = (H - cH) // 2
        x0 = (W - cW) // 2

    # Coordinates in absolute pixel space so the plane extends correctly
    y_fit, x_fit = np.meshgrid(
        np.arange(y0, y0 + cH), np.arange(x0, x0 + cW), indexing='ij'
    )
    z_flat = full_img[y0:y0 + cH, x0:x0 + cW].flatten()
    x_flat = x_fit.flatten()
    y_flat = y_fit.flatten()

    valid = np.isfinite(z_flat)
    if valid.sum() < 3:
        return full_img, (0.0, 0.0, 0.0), (y0, x0, cH, cW)

    X_design = np.column_stack([x_flat[valid], y_flat[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, z_flat[valid], rcond=None)
    a, b, c_val = coeffs

    # Plane over the FULL image (same absolute coordinate system)
    y_full, x_full = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    plane = a * x_full + b * y_full + c_val

    detrended = full_img - plane
    return detrended, (float(a), float(b), float(c_val)), (y0, x0, cH, cW)


def _plot_detrending_3d(roi_original, roi_detrended, filename=""):
    """Visualise the fitted detrending plane.

    Left subplot : 3D surface of the fitted plane (slope is clearly visible).
    Right subplot: 2D colour-map of the same plane (colour = Z / kerma value).
    """
    H, W = roi_original.shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Recompute the fitted plane
    z_flat = roi_original.flatten()
    valid = np.isfinite(z_flat)
    X_design = np.column_stack([x_coords.flatten()[valid],
                                 y_coords.flatten()[valid],
                                 np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, z_flat[valid], rcond=None)
    plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]

    fig = plt.figure(figsize=(16, 6))

    # --- Left: 3D surface of fitted plane only ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # Subsample for 3D rendering performance on large fitting regions
    step = max(1, max(H, W) // 150)
    surf = ax1.plot_surface(x_coords[::step, ::step], y_coords[::step, ::step],
                            plane[::step, ::step],
                            cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title(f'Fitted plane (heel / dome)\n{filename}', fontsize=10)
    ax1.set_xlabel('x (px)')
    ax1.set_ylabel('y (px)')
    ax1.set_zlabel('Kerma')
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Kerma')

    # --- Right: 2D colour-map of the fitted plane ---
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(plane, cmap='viridis', origin='lower',
                    extent=[0, W, 0, H], aspect='equal')
    ax2.set_title(f'Fitted plane colour-map\n{filename}', fontsize=10)
    ax2.set_xlabel('x (px)')
    ax2.set_ylabel('y (px)')
    fig.colorbar(im, ax=ax2, label='Kerma')

    fig.tight_layout()
    st.pyplot(fig)


def _fit_mpv_vs_kerma(kerma_vals, mpv_vals, method, poly_degree=2):
    """Fit MPV vs kerma using specified method. Returns fit values, formula, R², and coefficients."""
    k, m = np.array(kerma_vals, dtype=float), np.array(mpv_vals, dtype=float)
    
    if method == 'linear':
        p = np.polyfit(k, m, 1)
        fit_vals = np.polyval(p, k)
        formula = f"m = {p[0]:.4g} * k + {p[1]:.4g}"
    elif method == 'log':
        if np.any(k <= 0):
            raise ValueError("Kerma values must be positive for logarithmic fit")
        p = np.polyfit(np.log(k), m, 1)
        fit_vals = np.polyval(p, np.log(k))
        formula = f"m = {p[0]:.4g} * ln(k) + {p[1]:.4g}"
    elif method == 'poly':
        p = np.polyfit(k, m, poly_degree)
        fit_vals = np.polyval(p, k)
        formula = "m = " + " + ".join([f"{coef:.4g}*k^{deg}" for deg, coef in enumerate(p[::-1])])
    else:
        raise ValueError("Unknown fit method")
    
    ss_res, ss_tot = np.sum((m - fit_vals)**2), np.sum((m - np.mean(m))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return fit_vals, formula, r2, p


def _latex_formula(method, coeffs, poly_deg=2):
    """Generate LaTeX formula string for display."""
    c = np.array(coeffs, dtype=float)
    if method == 'linear':
        return rf"m = {c[0]:.4g}\,k + {c[1]:.4g}"
    elif method == 'log':
        return rf"m = {c[0]:.4g}\,\ln(k) + {c[1]:.4g}"
    else:  # poly
        terms = [f"{c[i]:.4g}\,k^{{{len(c)-1-i}}}" if len(c)-1-i > 1 else 
                 (f"{c[i]:.4g}\,k" if len(c)-1-i == 1 else f"{c[i]:.4g}") 
                 for i in range(len(c))]
        return "m = " + " + ".join(terms)

def _build_prediction_fn(method, coeffs, poly_deg=2):
    """Build prediction function: kerma -> MPV."""
    c = np.array(coeffs, dtype=float)
    if method == 'linear':
        return lambda k: c[0] * np.asarray(k, dtype=float) + c[1]
    elif method == 'log':
        return lambda k: c[0] * np.log(np.asarray(k, dtype=float)) + c[1]
    else:
        return lambda k: np.polyval(c, np.asarray(k, dtype=float))

def _build_inverse_fn(conv):
    """Build inverse function: MPV -> kerma."""
    method, coeffs = conv.get("method"), np.array(conv.get("coeffs"), dtype=float)
    if method == 'linear':
        a, b = coeffs
        if a == 0:
            raise ValueError("Inverse conversion undefined for a=0 in linear fit")
        return lambda m: (np.asarray(m, dtype=float) - b) / a
    elif method == 'log':
        a, b = coeffs
        if a == 0:
            raise ValueError("Inverse conversion undefined for a=0 in log fit")
        return lambda m: np.exp((np.asarray(m, dtype=float) - b) / a)
    else:
        raise NotImplementedError("Inverse conversion for 'poly' fit not supported. Use linear or log.")

def _compute_deviations(actual, fitted):
    """Compute percentage deviations and return list with max deviation."""
    actual, fitted = np.array(actual, dtype=float), np.array(fitted, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        devs = 100.0 * (actual - fitted) / fitted
    dev_list = [None if np.isnan(x) else float(x) for x in devs]
    abs_devs = [abs(d) for d in dev_list if d is not None]
    return dev_list, (max(abs_devs) if abs_devs else None)

def _constrained_weighted_fit(k_valid, y_valid, weights, n_starts=4, loss='linear'):
    """Perform constrained weighted least squares fit with multi-start verification.
    
    Args:
        k_valid: Valid kerma values
        y_valid: Valid y values (σ²)
        weights: Weights for each point
        n_starts: Number of initial guesses (default 4)
        loss: Loss function for least_squares ('linear' or 'soft_l1')
    
    Returns:
        Tuple of (a, b, c, success_msg, warning_msg)
    """
    def residuals(params):
        a, b, c = params
        y_pred = a * k_valid**2 + b * k_valid + c
        return np.sqrt(weights) * (y_valid - y_pred)
    
    # Generate initial guesses
    initial_guesses = [np.abs(np.polyfit(k_valid, y_valid, 2))]  # From polyfit
    np.random.seed(42)
    initial_guesses.extend([np.random.uniform(0, 1, size=3) for _ in range(n_starts - 1)])
    
    # Run optimization from each starting point
    results_list = []
    for x0 in initial_guesses:
        result = least_squares(
            residuals, x0=x0, bounds=(0, np.inf),
            loss=loss, f_scale=1.0
        )
        if result.success:
            results_list.append(result)
    
    if not results_list:
        return None, None, None, None, "All optimization attempts failed."
    
    # Select best solution and verify convergence
    params_array = np.array([r.x for r in results_list])
    best_idx = np.argmin([r.cost for r in results_list])
    a, b, c = results_list[best_idx].x
    
    # Check convergence
    param_stds = np.std(params_array, axis=0)
    param_means = np.mean(params_array, axis=0)
    relative_stds = param_stds / (param_means + 1e-10)
    
    warning_msg = None
    success_msg = None
    if np.any(relative_stds > 0.0001):
        warning_msg = (f"Multi-start convergence check: Some variation detected. "
                      f"Relative std: a={relative_stds[0]:.3f}, b={relative_stds[1]:.3f}, c={relative_stds[2]:.3f}")
    else:
        success_msg = (f"Multi-start convergence verified: All {len(results_list)} runs converged to same solution. "
                      f"Relative std: a={relative_stds[0]:.4g}, b={relative_stds[1]:.4g}, c={relative_stds[2]:.4g}")
    
    return a, b, c, success_msg, warning_msg


def _free_weighted_fit(k_valid, y_valid, weights, loss='linear'):
    """Perform unconstrained weighted least-squares quadratic fit.

    Uses the same least_squares solver as the constrained path but with
    unbounded parameters, so the only difference is the absence of a>=0,
    b>=0, c>=0 bounds.

    Returns:
        Tuple of (a, b, c, info_msg)
    """
    def residuals(params):
        a, b, c = params
        y_pred = a * k_valid**2 + b * k_valid + c
        return np.sqrt(weights) * (y_valid - y_pred)

    x0 = np.polyfit(k_valid, y_valid, 2)
    result = least_squares(residuals, x0=x0, loss=loss, f_scale=1.0)
    a, b, c = result.x
    info_msg = f"Free (unconstrained, loss={loss}) fit: a={a:.4g}, b={b:.4g}, c={c:.4g}"
    return float(a), float(b), float(c), info_msg


def _multi_roi_variance(img_array, roi_h=100, roi_w=100, shift=10, detrend=False):
    """Compute averaged σ² over 9 overlapping ROIs (central + 8 neighbours).

    The central ROI is extracted first, then 8 additional ROIs are obtained by
    shifting the centre by ±shift pixels in x and/or y.  Each ROI is optionally
    detrended, and the final σ² is the mean across all 9 estimates.

    Returns:
        (mean_sd2, individual_sd2_list, central_roi_for_plotting)
    """
    H, W = img_array.shape
    cy, cx = H // 2, W // 2

    offsets = [
        ( 0,  0),                                    # centre
        (-shift,  0), ( shift,  0),                   # N, S
        ( 0, -shift), ( 0,  shift),                   # W, E
        (-shift, -shift), (-shift,  shift),            # NW, NE
        ( shift, -shift), ( shift,  shift),            # SW, SE
    ]

    sd2_list = []
    central_roi = None
    for dy, dx in offsets:
        y0 = cy + dy - roi_h // 2
        x0 = cx + dx - roi_w // 2
        # Clamp to image bounds
        y0 = max(0, min(y0, H - roi_h))
        x0 = max(0, min(x0, W - roi_w))
        roi = img_array[y0:y0 + roi_h, x0:x0 + roi_w]

        if detrend:
            roi = _detrend_roi(roi)

        if central_roi is None:
            central_roi = roi  # keep first (central) for plotting

        sd2_list.append(float(np.nanvar(roi, ddof=1)))

    mean_sd2 = float(np.mean(sd2_list))
    return mean_sd2, sd2_list, central_roi


def _moving_roi_variance(img_array, roi_h=100, roi_w=100, detrend=False):
    """Compute σ² via non-overlapping moving ROI across 80% central area.

    Extracts the central 80% of the image area (linear dimension scale =
    sqrt(0.8)), then tiles it with non-overlapping roi_h × roi_w windows.
    Each window's variance is computed (optionally after planar detrending).

    Returns:
        (mean_sd2, sd2_array, n_rois, grid_shape)
    """
    H, W = img_array.shape
    scale = np.sqrt(0.8)
    cH, cW = int(np.floor(H * scale)), int(np.floor(W * scale))
    y0, x0 = (H - cH) // 2, (W - cW) // 2
    central = img_array[y0:y0 + cH, x0:x0 + cW]

    n_rows = cH // roi_h
    n_cols = cW // roi_w
    if n_rows < 1 or n_cols < 1:
        raise ValueError(
            f"80% central area ({cH}\u00d7{cW}) too small for "
            f"{roi_h}\u00d7{roi_w} non-overlapping ROI"
        )

    sd2_list = []
    for r in range(n_rows):
        for c in range(n_cols):
            roi = central[r * roi_h:(r + 1) * roi_h, c * roi_w:(c + 1) * roi_w]
            if detrend:
                roi = _detrend_roi(roi)
            sd2_list.append(float(np.nanvar(roi, ddof=1)))

    sd2_arr = np.array(sd2_list)
    mean_sd2 = float(np.nanmean(sd2_arr))
    return mean_sd2, sd2_arr, len(sd2_list), (n_rows, n_cols)


def _compute_dominance_interval(a, b, c):
    """Compute quantum noise dominance interval from fit coefficients.
    
    Returns:
        Dict with interval info: {abc_positive, k_min, k_max, exists, degenerate}
    """
    abc_positive = (a > 0) and (b > 0) and (c > 0)
    k_min = c / b if (abc_positive and b != 0) else None
    k_max = b / a if (abc_positive and a != 0) else None
    interval_exists, interval_degenerate = None, None
    
    if abc_positive and k_min and k_max:
        delta = b**2 - a*c
        if np.isclose(delta, 0.0, rtol=1e-10, atol=1e-12):
            interval_exists, interval_degenerate = False, True
        elif delta > 0:
            interval_exists, interval_degenerate = True, False
        else:
            interval_exists, interval_degenerate = False, False
    
    return {
        "abc_positive": bool(abc_positive),
        "k_min": (float(k_min) if k_min else None),
        "k_max": (float(k_max) if k_max else None),
        "upper_limit_k": (float(k_max) if k_max else None),
        "dominance_interval_exists": (bool(interval_exists) if interval_exists is not None else None),
        "dominance_interval_degenerate": (bool(interval_degenerate) if interval_degenerate is not None else None)
    }

def _explicit_noise_decomposition(images, k0, roi_h=100, roi_w=100,
                                   detrend=True, area_fraction=0.80):
    """Explicit noise decomposition following Monnin et al. 2014.

    Pixel-by-pixel method within the central ROI:

    1. (Optional) Detrend each full frame in kerma domain.
    2. Extract central roi_h×roi_w ROI from each of N frames.
    3. For pixel (i,j), compute the mean across N ROIs:
       mean_roi(i,j) = (1/N) Σ_n roi_n(i,j)
    4. Divide each ROI by the pixel-wise mean → N division images:
       div_n(i,j) = roi_n(i,j) / mean_roi(i,j)
    5. Compute spatial variance of each division image → S²_div,n
    6. Average: S²_div = (1/N) Σ_n S²_div,n
    7. Bessel correction: S²_stoch_rel = N/(N-1) · S²_div
    8. Convert to absolute: S²_stoch = S²_stoch_rel × ⟨mean_roi⟩²
    9. Total variance: S² = mean of Var(roi_n) across n
    10. Fixed-pattern: S²_fp = S² - S²_stoch

    Args:
        images: 3D array (N, H, W) — N kerma-domain full images at the same k0.
        k0: The kerma value (µGy) at which the images were acquired.
        roi_h, roi_w: Central ROI dimensions (default 100×100).
        detrend: Whether to apply planar detrending before ROI extraction.
        area_fraction: Fraction of image area used for detrending plane fit.

    Returns:
        dict with decomposition results including relative S²_div.
    """
    N = images.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 frames for explicit noise decomposition.")

    # --- Step 0: Optional detrending of full images ---
    processed = np.empty_like(images, dtype=float)
    for i in range(N):
        if detrend:
            processed[i], _, _ = _detrend_from_area(images[i], area_fraction)
        else:
            processed[i] = images[i].astype(float)

    # --- Step 1: Extract central ROI from each frame ---
    H, W = processed.shape[1], processed.shape[2]
    y0, x0 = (H - roi_h) // 2, (W - roi_w) // 2
    rois = processed[:, y0:y0 + roi_h, x0:x0 + roi_w]  # (N, roi_h, roi_w)

    # --- Step 2: Pixel-wise mean across N ROIs ---
    mean_roi = np.nanmean(rois, axis=0)  # (roi_h, roi_w)

    # --- Step 3: Divide each ROI by the pixel-wise mean → N division images ---
    with np.errstate(divide='ignore', invalid='ignore'):
        div_images = rois / mean_roi[np.newaxis, :, :]  # (N, roi_h, roi_w)
    div_images[~np.isfinite(div_images)] = np.nan

    # --- Step 4+5: Spatial variance of each division image, then average ---
    sd2_div_per_frame = [float(np.nanvar(div_images[i], ddof=1)) for i in range(N)]
    sd2_div = float(np.mean(sd2_div_per_frame))

    # --- Step 6: Bessel correction → relative stochastic variance ---
    bessel = N / (N - 1)
    sd2_stoch_rel = bessel * sd2_div

    # --- Step 7: Convert to absolute units (kerma²) ---
    mean_signal = float(np.nanmean(mean_roi))
    sd2_stoch = sd2_stoch_rel * mean_signal ** 2

    # --- Step 8: Total variance per frame (spatial variance of each ROI) ---
    sd2_per_frame = [float(np.nanvar(rois[i], ddof=1)) for i in range(N)]
    sd2_total = float(np.mean(sd2_per_frame))

    # --- Step 9: Fixed-pattern by subtraction ---
    sd2_fp = sd2_total - sd2_stoch

    return {
        "sd2_total": sd2_total,
        "sd2_stoch": sd2_stoch,
        "sd2_stoch_rel": sd2_stoch_rel,
        "sd2_fp": sd2_fp,
        "sd2_div": sd2_div,
        "sd2_div_per_frame": sd2_div_per_frame,
        "sd2_per_frame": sd2_per_frame,
        "mean_signal": mean_signal,
        "n_frames": N,
        "k0": float(k0),
    }


def _extrapolate_electronic_noise(kerma_levels, sd2_stoch_values, max_kerma=None):
    """Estimate electronic noise by linear extrapolation of stochastic variance to k→0.

    Follows Monnin et al. 2014 (eq. 10): S²_e = lim_{Q→0} S²_st(Q).
    Fits S²_stoch = β·k + S²_e using weighted least squares (w = 1/S²_stoch).

    Args:
        kerma_levels: array of kerma values.
        sd2_stoch_values: corresponding stochastic variance values.
        max_kerma: if set, only use points with k ≤ max_kerma for the fit.

    Returns:
        (sd2_electronic, beta, r2) — intercept, slope, and R².
    """
    k = np.array(kerma_levels, dtype=float)
    y = np.array(sd2_stoch_values, dtype=float)

    mask = np.isfinite(k) & np.isfinite(y) & (k > 0)
    if max_kerma is not None:
        mask &= k <= max_kerma
    k_v, y_v = k[mask], y[mask]
    if len(k_v) < 2:
        raise ValueError("Need at least 2 stochastic-variance points for electronic noise extrapolation.")

    w = 1.0 / y_v
    # Weighted linear fit: y = beta*k + se2
    W = np.diag(w)
    A = np.column_stack([k_v, np.ones_like(k_v)])
    params = np.linalg.lstsq(W @ A, W @ y_v, rcond=None)[0]
    beta, se2 = float(params[0]), float(params[1])

    y_fit = beta * k_v + se2
    ss_res = float(np.sum(w * (y_v - y_fit)**2))
    ss_tot = float(np.sum(w * (y_v - np.average(y_v, weights=w))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return se2, beta, r2


def _get_detector_conversion_state() -> dict:
    """Get unified detector conversion state container from session.

    Structure:
      detector_conversion = {
        'fit': {...},
        'results': {...} | None,
        'ei_fit': {...},
        'sd2_fit': {...},
        'explicit_decomp': {...}
      }
    """
    state = st.session_state.get("detector_conversion")
    if not isinstance(state, dict):
        state = {}
    state.setdefault("fit", {})
    state.setdefault("results", None)
    state.setdefault("ei_fit", {})
    state.setdefault("sd2_fit", {})
    state.setdefault("explicit_decomp", {})
    st.session_state["detector_conversion"] = state
    return state


def _render_cached_fit(cached, title, x_label, y_label):
    """Render a cached fit with formula, R², deviations, and plot."""
    if not (isinstance(cached, dict) and cached.get("coeffs") is not None):
        return
    
    st.caption(f"Fitted using: {cached.get('method', '?')}")
    if cached.get("latex_formula"):
        st.latex(cached.get("latex_formula"))
    else:
        st.write(cached.get("formula", ""))
    
    if cached.get("r2") is not None:
        st.write(f"R² = {cached['r2']:.4f}")
    
    # Show max deviation if available
    max_dev = cached.get("max_deviation")
    if max_dev is not None:
        st.write(f"Maximum deviation from fit: {max_dev:.3f}%")
    
    # Plot if data available
    if "x_data" in cached and "y_data" in cached and "y_fit" in cached:
        fig, ax = plt.subplots()
        x_data = np.array(cached["x_data"])
        y_data = np.array(cached["y_data"])
        
        # Generate dense x values for smooth curve plotting
        x_min, x_max = x_data.min(), x_data.max()
        x_smooth = np.linspace(x_min, x_max, 200)
        
        # Compute fitted curve at dense points for smooth visualization
        method = cached.get("method")
        coeffs = np.array(cached.get("coeffs"))
        
        if method == 'linear':
            y_smooth = np.polyval(coeffs, x_smooth)
        elif method == 'log':
            with np.errstate(divide='ignore', invalid='ignore'):
                y_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]
        elif method == 'poly':
            y_smooth = np.polyval(coeffs, x_smooth)
        else:
            # Fallback: just plot original points
            y_smooth = np.array(cached["y_fit"])
            x_smooth = x_data
        
        ax.scatter(x_data, y_data, label='data', s=50, zorder=5)
        ax.plot(x_smooth, y_smooth, color='C1', label='fit', linewidth=2)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def display_detector_conversion_section(uploaded_files: list[ImagePayload] | None = None):
    st.subheader("Measured Kerma Values and Exposition Index (EI) Assignment")

    uploaded = uploaded_files if uploaded_files else st.file_uploader(
        "Upload RAW or STD files", type=["raw", "RAW", "std", "STD"], accept_multiple_files=True
    )
    if not uploaded:
        st.info("Upload one or more RAW/STD files to begin")
        return None

    dc_state = _get_detector_conversion_state()

    file_header_hints = []
    uploaded_exts = []
    for f in uploaded:
        fname, fbytes = file_name_and_bytes(f)
        uploaded_exts.append((fname.split('.')[-1] if '.' in fname else '').lower())
        file_header_hints.append(_extract_dicom_header_hints(fbytes))

    shared_raw_params = st.session_state.get("shared_raw_params_current_test")
    if isinstance(shared_raw_params, dict) and shared_raw_params.get("dtype") is not None:
        dtype = np.dtype(shared_raw_params["dtype"]).type
    else:
        dtype = np.uint16
        st.caption("Pixel dtype inferred from shared RAW params when available; fallback used: uint16")
    default_little_endian = bool(st.session_state.get("raw_little_endian_default", True))

    st.write("Enter kerma values for each file. EI values might be auto-populated from file metadata when available, but can be edited as needed.\n\n After entering values, use the buttons below to run fits and analyze results.")
    kerma_vals, results = [], {"files": []}
    
    for idx, f in enumerate(uploaded):
        fname, fbytes = file_name_and_bytes(f)
        metadata_hints = file_header_hints[idx] if idx < len(file_header_hints) else {}
        ei_default = float(metadata_hints.get("relative_xray_exposure") or 0.0)
        col_a, col_b = st.columns(2)
        with col_a:
            kerma_val = st.number_input(f"Kerma (μGy) — {fname}", value=0.0, format="%.4f", key=f"kerma_{fname}")
        with col_b:
            ei_val = st.number_input(
                f"Exposition Index (EI) — {fname}",
                value=ei_default,
                format="%.2f",
                key=f"ei_{fname}"
            )
        kerma_vals.append(kerma_val)
        
        try:
            arr, endian_used, endian_source = _read_raw_as_square(
                fbytes,
                dtype,
                little_endian=default_little_endian,
                auto_endian_from_dicom=True
            )
            mpv, sd, roi = _central_roi_stats(arr)
            results["files"].append({
                "filename": fname, "kerma": float(kerma_val), "ei": float(ei_val),
                "mpv": float(mpv), "sd": float(sd), "roi": roi, "array": arr
            })
            st.write(f"MPV: {mpv:.3f}, σ: {sd:.3f}")
        except Exception as e:
            st.error(f"Failed to process {fname}: {e}")
            return None
    
    st.markdown("---")
    # --- Detector Response Curve (MPV vs Kerma) ---
    st.write("### Detector Response Curve")
    
    all_std = len(uploaded_exts) > 0 and all(ext == 'std' for ext in uploaded_exts)
    fallback_fit_method = 'log' if all_std else 'linear'
    pir_fit_hints = [h.get("pixel_intensity_relationship_fit_method") for h in file_header_hints]
    default_fit_method = _select_default_fit_method_from_pir(pir_fit_hints, fallback_fit_method)

    detected_pir_values = sorted({h.get("pixel_intensity_relationship_raw") for h in file_header_hints if h.get("pixel_intensity_relationship_raw")})
    if detected_pir_values:
        st.caption(
            "Detected Pixel Intensity Relationship (0028,1040): "
            + ", ".join(detected_pir_values)
        )

    fit_method = st.selectbox(
        "Fit method", options=['linear', 'log', 'poly'], index=['linear', 'log', 'poly'].index(default_fit_method),
        help="RAW defaults to linear; STD defaults to log. EI vs Kerma is always linear.",
        key="fit_method_detector_curve"
    )
    poly_degree = st.selectbox("Polynomial degree", [2, 3], index=0, key="poly_degree_detector_curve") if fit_method == 'poly' else 2
    
    mpv_vals = [r['mpv'] for r in results["files"]]
    ei_vals = [float(r.get('ei', 0)) if r.get('ei') is not None else None for r in results["files"]]
    detector_curve_fitted = False
    
    if st.button("Run fit: Detector Response Curve", key="run_fit_detector_curve_out"):
        try:
            fit_vals, formula, r2, p = _fit_mpv_vs_kerma(kerma_vals, mpv_vals, fit_method, poly_degree)
            pred_fn = _build_prediction_fn(fit_method, p, poly_degree)
            deviations_list, max_dev = _compute_deviations(mpv_vals, fit_vals)
            
            dc_state["fit"] = {
                "method": fit_method, "coeffs": p.tolist(), 
                "poly_degree": int(poly_degree) if fit_method == 'poly' else None,
                "formula": formula, "latex_formula": _latex_formula(fit_method, p, poly_degree),
                "r2": float(r2) if not np.isnan(r2) else None, "predict_mpv": pred_fn,
                "x_data": kerma_vals, "y_data": mpv_vals, "y_fit": fit_vals.tolist(),
                "max_deviation": max_dev
            }
            st.session_state["detector_conversion"] = dc_state
            detector_curve_fitted = True
        except Exception as e:
            st.error(f"Fit failed: {e}")

    _render_cached_fit(dc_state.get("fit"), "Detector Response", r"$k$ (μGy)", "MPV")

    # --- Exposition Index (EI) vs Kerma ---
    st.write("### Exposition Index (EI) vs Kerma Fit")
    if st.button("Run fit: EI vs Kerma", key="run_fit_ei_curve_out"):
        if any(v is None for v in ei_vals):
            st.error("EI values missing for one or more files; cannot fit EI vs Kerma.")
        else:
            try:
                fit_vals_ei, formula_ei, r2_ei, p_ei = _fit_mpv_vs_kerma(kerma_vals, ei_vals, 'linear', 1)
                deviations_ei, max_dev_ei = _compute_deviations(ei_vals, fit_vals_ei)
                dc_state["ei_fit"] = {
                    "method": "linear", "coeffs": p_ei.tolist(), "formula": formula_ei,
                    "latex_formula": rf"EI = {p_ei[0]:.4g}\,k + {p_ei[1]:.4g}",
                    "r2": float(r2_ei) if not np.isnan(r2_ei) else None,
                    "x_data": kerma_vals, "y_data": ei_vals, "y_fit": fit_vals_ei.tolist(),
                    "max_deviation": max_dev_ei
                }
                st.session_state["detector_conversion"] = dc_state
            except Exception as e:
                st.error(f"EI fit failed: {e}")
    
    _render_cached_fit(dc_state.get("ei_fit"), "EI vs Kerma", r"$k$ (μGy)", "Exposition Index (EI)")

    # --- Noise: σ² vs Kerma ---
    st.write("### Noise: σ² vs Kerma")
    st.caption("Compute σ on linearized (kerma-domain) ROI, square it (σ²), then fit σ² = a·k² + b·k + c.")

    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
    with col_opt1:
        use_detrend = st.checkbox("Apply planar detrending (heel / dome removal)", value=False, key="sd2_detrend")
    with col_opt2:
        use_constrained = st.checkbox("Non-negative constrained fit (a,b,c ≥ 0)", value=True, key="sd2_constrained")
    with col_opt3:
        use_multi_roi = st.checkbox("Multi-ROI averaging (9 × 100×100)", value=False, key="sd2_multi_roi")
    with col_opt4:
        use_moving_roi = st.checkbox("Moving-ROI variance (100×100, 80% area)", value=False, key="sd2_moving_roi")
    detrend_area_pct = 80
    if use_detrend:
        detrend_area_pct = st.slider(
            "Detrending area (% of image area)",
            min_value=10, max_value=100, value=80, step=5,
            key="sd2_detrend_area",
            help="Fraction of the image area used to fit the detrending plane. "
                 "Larger areas give better spatial leverage for gradient estimation "
                 "but may include edge artefacts near 100%."
        )
    detrend_area_frac = detrend_area_pct / 100.0
    if use_multi_roi and use_moving_roi:
        st.warning("Multi-ROI and Moving-ROI cannot be used simultaneously. Using Moving-ROI.")
        use_multi_roi = False

    # Loss function depends on detrending: detrend ON → pure L2; OFF → robust soft_l1
    fit_loss = 'linear' if use_detrend else 'soft_l1'
    st.caption(f"Loss function: **{fit_loss}** ({'detrending removes systematics → L2 sufficient' if use_detrend else 'no detrending → robust loss to handle residual systematics'})")

    if st.button("Run fit: σ² vs Kerma", key="run_fit_sd2"):
        conv = dc_state.get("fit")
        if not (isinstance(conv, dict) and conv.get("coeffs")):
            st.error("Run the Detector Response Curve fit first.")
        else:
            try:
                inv_fn = _build_inverse_fn(conv)
                sd2_vals = []
                moving_roi_distributions = []

                for rec in results["files"]:
                    if rec.get("roi") is None:
                        st.error(f"Missing ROI data for {rec['filename']}")
                        return None

                    full_arr = rec.get("array")

                    if use_detrend:
                        # --- Global detrending in KERMA frame ---
                        # The plane must be fitted in kerma domain, not pixel
                        # domain.  For a log detector response the inverse is
                        # exponential, so a pixel-space plane would become a
                        # non-planar surface in kerma space.  Converting first
                        # ensures the detrending is physically correct.
                        if full_arr is None:
                            st.error(f"Full image not available for {rec['filename']}")
                            return None
                        kerma_full = np.asarray(inv_fn(full_arr), dtype=float)
                        kerma_full[~np.isfinite(kerma_full)] = np.nan
                        detrended_full, plane_abc, fit_bounds = _detrend_from_area(
                            kerma_full, detrend_area_frac
                        )
                        y0f, x0f, cHf, cWf = fit_bounds
                        fit_region = kerma_full[y0f:y0f + cHf, x0f:x0f + cWf]
                        _plot_detrending_3d(
                            fit_region, _detrend_roi(fit_region),
                            filename=f"{rec['filename']} (detrend area: {detrend_area_pct}%)"
                        )

                        if use_moving_roi:
                            mean_sd2, sd2_arr, n_rois, grid_shape = _moving_roi_variance(
                                detrended_full, detrend=False
                            )
                            sd2_vals.append(mean_sd2)
                            moving_roi_distributions.append({
                                "filename": rec["filename"],
                                "sd2_values": sd2_arr.tolist(),
                                "n_rois": n_rois,
                                "grid_shape": list(grid_shape),
                                "mean_sd2": mean_sd2,
                            })
                            st.caption(
                                f"{rec['filename']}: {n_rois} ROIs "
                                f"({grid_shape[0]}\u00d7{grid_shape[1]} grid), "
                                f"mean \u03c3\u00b2 = {mean_sd2:.4g}"
                            )
                        elif use_multi_roi:
                            mean_sd2, _, _ = _multi_roi_variance(
                                detrended_full, detrend=False
                            )
                            sd2_vals.append(mean_sd2)
                        else:
                            _, _, roi_kerma = _central_roi_stats(detrended_full)
                            sd2_vals.append(float(np.nanvar(roi_kerma, ddof=1)))

                    else:
                        # --- No detrending ---
                        if use_moving_roi:
                            if full_arr is None:
                                st.error(f"Full image not available for {rec['filename']}")
                                return None
                            kerma_full = np.asarray(inv_fn(full_arr), dtype=float)
                            kerma_full[~np.isfinite(kerma_full)] = np.nan
                            mean_sd2, sd2_arr, n_rois, grid_shape = _moving_roi_variance(
                                kerma_full, detrend=False
                            )
                            sd2_vals.append(mean_sd2)
                            moving_roi_distributions.append({
                                "filename": rec["filename"],
                                "sd2_values": sd2_arr.tolist(),
                                "n_rois": n_rois,
                                "grid_shape": list(grid_shape),
                                "mean_sd2": mean_sd2,
                            })
                            st.caption(
                                f"{rec['filename']}: {n_rois} ROIs "
                                f"({grid_shape[0]}\u00d7{grid_shape[1]} grid), "
                                f"mean \u03c3\u00b2 = {mean_sd2:.4g}"
                            )
                        else:
                            kerma_img = np.asarray(inv_fn(rec["roi"]), dtype=float)
                            kerma_img[~np.isfinite(kerma_img)] = np.nan

                            if use_multi_roi:
                                mean_sd2, _, _ = _multi_roi_variance(
                                    kerma_img, detrend=False
                                )
                                sd2_vals.append(mean_sd2)
                            else:
                                _, _, roi_kerma = _central_roi_stats(kerma_img)
                                sd2_vals.append(float(np.nanvar(roi_kerma, ddof=1)))

                k_arr = np.array(kerma_vals, dtype=float)
                y_arr = np.array(sd2_vals, dtype=float)

                mask = np.isfinite(k_arr) & np.isfinite(y_arr) & (k_arr > 0)
                if mask.sum() < 3:
                    st.error("Need at least 3 valid points to fit a quadratic.")
                else:
                    k_v, y_v = k_arr[mask], y_arr[mask]
                    w = 1.0 / y_v

                    if use_constrained:
                        st.info(f"Non-negative constrained fit (loss={fit_loss}), weights = 1/σ².")
                        a_, b_, c_, ok_msg, warn_msg = _constrained_weighted_fit(k_v, y_v, w, loss=fit_loss)
                        if a_ is None:
                            st.error(warn_msg)
                            return None
                        if warn_msg:
                            st.warning(warn_msg)
                        if ok_msg:
                            st.success(ok_msg)
                    else:
                        a_, b_, c_, info_msg = _free_weighted_fit(k_v, y_v, w, loss=fit_loss)
                        st.info(info_msg)

                    y_fit = a_ * k_arr**2 + b_ * k_arr + c_
                    ss_res = np.nansum((y_arr - y_fit)**2)
                    ss_tot = np.nansum((y_arr - np.nanmean(y_arr))**2)
                    r2_sd = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    interval_info = _compute_dominance_interval(a_, b_, c_)

                    label_detrend = f"detrended ({detrend_area_pct}%)" if use_detrend else "raw"
                    label_fit = "constrained" if use_constrained else "free"
                    label_roi = ("moving-ROI 80%" if use_moving_roi
                                 else ("9-ROI avg" if use_multi_roi else "single ROI"))

                    dc_state["sd2_fit"] = {
                        "coeffs": [float(a_), float(b_), float(c_)],
                        "formula": f"\u03c3\u00b2 = {a_:.4g}\u00b7k\u00b2 + {b_:.4g}\u00b7k + {c_:.4g}",
                        "latex_formula": rf"\sigma^{{2}} = {a_:.4g}\,k^{{2}} + {b_:.4g}\,k + {c_:.4g}",
                        "r2": float(r2_sd) if not np.isnan(r2_sd) else None,
                        "sd2": [None if not np.isfinite(v) else float(v) for v in y_arr],
                        "detrended": use_detrend,
                        "detrend_area_pct": detrend_area_pct if use_detrend else None,
                        "constrained": use_constrained,
                        "multi_roi": use_multi_roi,
                        "moving_roi": use_moving_roi,
                        "moving_roi_distributions": moving_roi_distributions if use_moving_roi else None,
                        "loss": fit_loss,
                        "label": f"{label_detrend} / {label_fit} / {label_roi} / loss={fit_loss}",
                        **interval_info,
                    }
                    st.session_state["detector_conversion"] = dc_state
            except NotImplementedError as nie:
                st.error(str(nie))
            except Exception as e:
                st.error(f"σ² fit failed: {e}")

    # Render cached SD² fit
    cached_sd = dc_state.get("sd2_fit")
    if isinstance(cached_sd, dict) and cached_sd.get("coeffs"):
        if cached_sd.get("label"):
            st.caption(f"Last fit: {cached_sd['label']}")
        st.latex(cached_sd.get("latex_formula")) if cached_sd.get("latex_formula") else st.write(cached_sd.get("formula", ""))
        if cached_sd.get("r2") is not None:
            st.write(f"R² = {cached_sd['r2']:.4f}")

        # Dominance interval feedback
        abc_positive = cached_sd.get("abc_positive")
        k_min, k_max = cached_sd.get("k_min"), cached_sd.get("k_max")
        interval_exists = cached_sd.get("dominance_interval_exists")
        interval_degenerate = cached_sd.get("dominance_interval_degenerate")

        if not abc_positive:
            st.info("Coefficients not all positive; dominance interval not computed.")
        elif k_min and k_max and k_min > 0 and k_max > 0:
            if interval_exists:
                st.write("Quantum noise dominance interval (over structural and electronic noise):")
                st.latex(rf"(k_\mathrm{{min}},\,k_\mathrm{{max}}) = \left({k_min:.4g},\,{k_max:.4g}\right)\;\mu\,Gy")
            elif interval_degenerate:
                st.latex(rf"k_\mathrm{{min}} = k_\mathrm{{max}} = {k_min:.4g}\;\mu\,Gy\quad (b^2 \approx a\,c)")
            else:
                st.latex(r"b^2 \le a\,c\quad\text{(no middle-term dominance interval)}")
        else:
            st.info("Dominance interval bounds could not be determined.")

        # Plot σ² components
        try:
            k_arr = np.array(kerma_vals, dtype=float)
            y_arr = np.array(cached_sd.get("sd2", []), dtype=float)
            a_, b_, c_ = np.array(cached_sd["coeffs"], dtype=float)

            k_smooth = np.linspace(k_arr.min(), k_arr.max(), 200)
            y_fit_smooth = a_ * k_smooth**2 + b_ * k_smooth + c_

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.scatter(k_arr, y_arr, label='σ² data', color='black', s=50, zorder=5)
            ax3.plot(k_smooth, y_fit_smooth, color='C3', linewidth=2, label='Total: $a·k² + b·k + c$')
            ax3.plot(k_smooth, a_ * k_smooth**2, '--', color='C0', linewidth=1.5, label=f'Structural: ${a_:.4g}·k²$')
            ax3.plot(k_smooth, b_ * k_smooth, '--', color='C2', linewidth=1.5, label=f'Quantum: ${b_:.4g}·k$')
            ax3.axhline(c_, linestyle='--', color='C1', linewidth=1.5, label=f'Electronic: ${c_:.4g}$')
            ax3.set_xlabel(r"$k$ (μGy)", fontsize=12)
            ax3.set_ylabel(r"$\sigma²$", fontsize=12)
            ax3.set_title(f"σ² vs Kerma — {cached_sd.get('label', '')}")
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Could not display cached SD fit: {e}")

        # Histogram of moving-ROI local variances
        if cached_sd.get("moving_roi") and cached_sd.get("moving_roi_distributions"):
            st.write("#### Local σ² Distribution (Moving-ROI)")
            dists = cached_sd["moving_roi_distributions"]
            n_files = len(dists)
            n_cols_hist = min(n_files, 4)
            n_rows_hist = (n_files + n_cols_hist - 1) // n_cols_hist
            fig_hist, axes = plt.subplots(
                n_rows_hist, n_cols_hist,
                figsize=(5 * n_cols_hist, 4 * n_rows_hist), squeeze=False
            )
            for i, dist_info in enumerate(dists):
                ax = axes[i // n_cols_hist, i % n_cols_hist]
                vals = np.array(dist_info["sd2_values"])
                ax.hist(vals, bins='auto', edgecolor='black', alpha=0.7, color='C0')
                ax.axvline(dist_info["mean_sd2"], color='red', linestyle='--',
                           linewidth=1.5, label=f'mean = {dist_info["mean_sd2"]:.4g}')
                std_val = float(np.std(vals))
                ax.axvline(dist_info["mean_sd2"] - std_val, color='orange',
                           linestyle=':', linewidth=1, label=f'±1σ ({std_val:.3g})')
                ax.axvline(dist_info["mean_sd2"] + std_val, color='orange',
                           linestyle=':', linewidth=1)
                ax.set_title(dist_info["filename"], fontsize=9)
                ax.set_xlabel("σ² (per ROI)")
                ax.set_ylabel("Count")
                ax.legend(fontsize=7)
                n_rois = dist_info["n_rois"]
                grid = dist_info["grid_shape"]
                ax.text(0.95, 0.95, f"N={n_rois} ({grid[0]}×{grid[1]})",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            for i in range(n_files, n_rows_hist * n_cols_hist):
                axes[i // n_cols_hist, i % n_cols_hist].set_visible(False)
            fig_hist.suptitle(
                "Per-ROI variance distribution (non-overlapping 100×100, 80% central area)",
                fontsize=11
            )
            fig_hist.tight_layout()
            st.pyplot(fig_hist)

    # =====================================================================
    # Explicit Noise Decomposition (Monnin et al. 2014)
    # =====================================================================
    st.markdown("---")
    st.write("### Explicit Noise Decomposition (Monnin et al. 2014)")
    st.caption(
        "Upload N images acquired at the **same kerma level** to separate "
        "total noise into stochastic (quantum + electronic) and "
        "fixed-pattern (structural) components via the explicit method.  "
        "Requires the Detector Response Curve fit above."
    )

    explicit_files = st.file_uploader(
        "Upload same-kerma images (RAW/STD)",
        type=["raw", "RAW", "std", "STD"],
        accept_multiple_files=True,
        key="explicit_decomp_uploader",
    )
    explicit_kerma = st.number_input(
        "Kerma level k\u2080 (µGy) for the uploaded images",
        value=0.0, format="%.4f", key="explicit_k0",
    )

    col_ed1, col_ed2 = st.columns(2)
    with col_ed1:
        explicit_detrend = st.checkbox(
            "Detrend each frame (heel / dome removal)",
            value=True, key="explicit_detrend",
        )
    with col_ed2:
        explicit_detrend_area = 80
        if explicit_detrend:
            explicit_detrend_area = st.slider(
                "Detrending area (%)",
                min_value=10, max_value=100, value=80, step=5,
                key="explicit_detrend_area",
            )
    explicit_detrend_frac = explicit_detrend_area / 100.0

    if st.button("Run explicit noise decomposition", key="run_explicit_decomp"):
        conv = dc_state.get("fit")
        if not (isinstance(conv, dict) and conv.get("coeffs")):
            st.error("Run the Detector Response Curve fit first.")
        elif not explicit_files or len(explicit_files) < 2:
            st.error("Upload at least 2 images at the same kerma level.")
        elif explicit_kerma <= 0:
            st.error("Kerma level k\u2080 must be > 0.")
        else:
            try:
                inv_fn = _build_inverse_fn(conv)

                # Load all same-kerma images → kerma-domain ROI stack
                frames = []
                for ef in explicit_files:
                    ef_name, ef_bytes = file_name_and_bytes(ef)
                    arr_e, _, _ = _read_raw_as_square(
                        ef_bytes, dtype,
                        little_endian=default_little_endian,
                        auto_endian_from_dicom=True,
                    )
                    kerma_frame = np.asarray(inv_fn(arr_e), dtype=float)
                    kerma_frame[~np.isfinite(kerma_frame)] = np.nan
                    frames.append(kerma_frame)

                roi_stack = np.stack(frames, axis=0)  # (N, H, W)
                st.info(f"Loaded {len(frames)} frames at k\u2080 = {explicit_kerma:.4g} \u00b5Gy")

                decomp = _explicit_noise_decomposition(
                    roi_stack, explicit_kerma,
                    detrend=explicit_detrend,
                    area_fraction=explicit_detrend_frac,
                )

                # --- Electronic noise: use polynomial sd2_fit stochastic data ---
                cached_sd = dc_state.get("sd2_fit")
                se2, beta, r2_e = None, None, None
                sd2_stoch_at_levels = None

                if isinstance(cached_sd, dict) and cached_sd.get("sd2"):
                    # Predict structural variance at every kerma level using
                    # the k² scaling anchored at k0
                    k0 = decomp["k0"]
                    all_kerma = np.array(kerma_vals, dtype=float)
                    all_sd2_total = np.array(
                        [v if v is not None else np.nan for v in cached_sd["sd2"]],
                        dtype=float,
                    )

                    fp_ratio = decomp["sd2_fp"] / (k0**2) if k0 > 0 else 0.0
                    sd2_fp_predicted = fp_ratio * all_kerma**2
                    sd2_stoch_at_levels = all_sd2_total - sd2_fp_predicted

                    # Combine with the explicit measurement at k0
                    stoch_kerma = np.append(all_kerma, k0)
                    stoch_vals = np.append(sd2_stoch_at_levels, decomp["sd2_stoch"])

                    try:
                        se2, beta, r2_e = _extrapolate_electronic_noise(
                            stoch_kerma, stoch_vals,
                        )
                    except ValueError as ve:
                        st.warning(f"Electronic noise extrapolation failed: {ve}")

                # Store results
                dc_state["explicit_decomp"] = {
                    **decomp,
                    "detrended": explicit_detrend,
                    "detrend_area_pct": explicit_detrend_area if explicit_detrend else None,
                    "sd2_electronic": float(se2) if se2 is not None else None,
                    "beta_quantum": float(beta) if beta is not None else None,
                    "r2_electronic_fit": float(r2_e) if r2_e is not None and not np.isnan(r2_e) else None,
                    "sd2_fp_predicted": sd2_fp_predicted.tolist() if sd2_stoch_at_levels is not None else None,
                    "sd2_stoch_at_levels": sd2_stoch_at_levels.tolist() if sd2_stoch_at_levels is not None else None,
                    "kerma_levels": kerma_vals,
                }
                st.session_state["detector_conversion"] = dc_state
            except Exception as e:
                st.error(f"Explicit decomposition failed: {e}")

    # --- Render cached explicit decomposition ---
    cached_ex = dc_state.get("explicit_decomp")
    if isinstance(cached_ex, dict) and cached_ex.get("sd2_total") is not None:
        k0 = cached_ex["k0"]
        N = cached_ex["n_frames"]
        sd2_div = cached_ex.get("sd2_div", 0.0)
        sd2_stoch_rel = cached_ex.get("sd2_stoch_rel", 0.0)
        mean_sig = cached_ex.get("mean_signal", 0.0)
        st.write(f"**Results at k\u2080 = {k0:.4g} \u00b5Gy  (N = {N} frames, \u27e8p\u27e9 = {mean_sig:.4g} \u00b5Gy)**")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("S\u00b2 total", f"{cached_ex['sd2_total']:.4g}")
        with col_r2:
            st.metric("S\u00b2 div (relative)", f"{sd2_div:.4g}")
        with col_r3:
            st.metric("S\u00b2 stochastic", f"{cached_ex['sd2_stoch']:.4g}")
        with col_r4:
            st.metric("S\u00b2 fixed-pattern", f"{cached_ex['sd2_fp']:.4g}")

        st.caption("Step-by-step conversion from division variance to absolute stochastic variance:")
        st.latex(
            rf"\overline{{S^{{2}}_{{\mathrm{{div}}}}}} = \frac{{1}}{{N}}\sum_{{n=1}}^{{{N}}} "
            rf"S^{{2}}_{{\mathrm{{div}},n}} = {sd2_div:.4g}"
        )
        st.latex(
            rf"S^{{2}}_{{\mathrm{{stoch,rel}}}} = \frac{{N}}{{N-1}}\,\overline{{S^{{2}}_{{\mathrm{{div}}}}}} "
            rf"= \frac{{{N}}}{{{N-1}}}\times {sd2_div:.4g} = {sd2_stoch_rel:.4g}"
        )
        st.latex(
            rf"S^{{2}}_{{\mathrm{{stoch}}}} = S^{{2}}_{{\mathrm{{stoch,rel}}}} \times \langle p \rangle^{{2}} "
            rf"= {sd2_stoch_rel:.4g} \times {mean_sig:.4g}^{{2}} = {cached_ex['sd2_stoch']:.4g}"
        )
        st.latex(
            rf"S^{{2}}_{{\mathrm{{fp}}}} = S^{{2}} - S^{{2}}_{{\mathrm{{stoch}}}} "
            rf"= {cached_ex['sd2_total']:.4g} - {cached_ex['sd2_stoch']:.4g} "
            rf"= {cached_ex['sd2_fp']:.4g}"
        )

        # Electronic noise results
        se2 = cached_ex.get("sd2_electronic")
        beta = cached_ex.get("beta_quantum")
        r2_e = cached_ex.get("r2_electronic_fit")
        if se2 is not None:
            st.write("---")
            st.write("**Electronic noise (extrapolation to k \u2192 0)**")
            st.latex(
                rf"S^{{2}}_{{\mathrm{{stoch}}}}(k) = {beta:.4g}\,k + {se2:.4g}"
            )
            if r2_e is not None:
                st.write(f"R\u00b2 = {r2_e:.4f}")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.metric("S\u00b2 electronic", f"{se2:.4g}")
            with col_e2:
                sd2_q_at_k0 = cached_ex["sd2_stoch"] - se2
                st.metric(f"S\u00b2 quantum (at k\u2080={k0:.4g})", f"{sd2_q_at_k0:.4g}")

        # --- Comprehensive decomposition plot ---
        sd2_fp_pred = cached_ex.get("sd2_fp_predicted")
        sd2_stoch_levels = cached_ex.get("sd2_stoch_at_levels")
        k_levels = cached_ex.get("kerma_levels")
        cached_sd = dc_state.get("sd2_fit")

        if (sd2_fp_pred is not None and sd2_stoch_levels is not None
                and k_levels is not None
                and isinstance(cached_sd, dict) and cached_sd.get("sd2")):
            k_all = np.array(k_levels, dtype=float)
            sd2_total_all = np.array(
                [v if v is not None else np.nan for v in cached_sd["sd2"]],
                dtype=float,
            )
            fp_arr = np.array(sd2_fp_pred, dtype=float)
            stoch_arr = np.array(sd2_stoch_levels, dtype=float)

            k_smooth = np.linspace(max(k_all.min(), 0.01), k_all.max(), 200)
            fp_ratio = cached_ex["sd2_fp"] / (k0**2) if k0 > 0 else 0.0
            fp_smooth = fp_ratio * k_smooth**2

            fig_dec, ax_dec = plt.subplots(figsize=(10, 6))
            # Total
            ax_dec.scatter(k_all, sd2_total_all, color='black', s=50, zorder=5, label='S\u00b2 total (data)')
            # Structural
            ax_dec.plot(k_smooth, fp_smooth, '--', color='C0', linewidth=1.5,
                        label=f'S\u00b2 structural (k\u00b2 scaling from k\u2080)')
            ax_dec.scatter(k_all, fp_arr, marker='s', color='C0', s=30, zorder=4, alpha=0.7)
            ax_dec.scatter([k0], [cached_ex['sd2_fp']], marker='D', color='C0',
                           s=80, zorder=6, edgecolors='black',
                           label=f'S\u00b2 fp measured (k\u2080={k0:.4g})')
            # Stochastic
            ax_dec.scatter(k_all, stoch_arr, marker='^', color='C2', s=30, zorder=4,
                           alpha=0.7, label='S\u00b2 stochastic (by subtraction)')
            ax_dec.scatter([k0], [cached_ex['sd2_stoch']], marker='D', color='C2',
                           s=80, zorder=6, edgecolors='black',
                           label=f'S\u00b2 stoch measured (k\u2080={k0:.4g})')

            if se2 is not None and beta is not None:
                stoch_smooth = beta * k_smooth + se2
                ax_dec.plot(k_smooth, stoch_smooth, '--', color='C2', linewidth=1.5,
                            label=f'S\u00b2 stoch fit: {beta:.4g}\u00b7k + {se2:.4g}')
                ax_dec.axhline(se2, linestyle=':', color='C1', linewidth=1.5,
                               label=f'S\u00b2 electronic = {se2:.4g}')
                q_smooth = beta * k_smooth
                ax_dec.plot(k_smooth, q_smooth, '-.', color='C3', linewidth=1.5,
                            label=f'S\u00b2 quantum: {beta:.4g}\u00b7k')

            ax_dec.set_xlabel(r"$k$ (\u00b5Gy)", fontsize=12)
            ax_dec.set_ylabel(r"$S^{2}$", fontsize=12)
            ax_dec.set_title("Explicit Noise Decomposition (Monnin et al. 2014)")
            ax_dec.legend(loc='best', fontsize=8)
            ax_dec.grid(True, alpha=0.3)
            st.pyplot(fig_dec)

        # Per-frame variance table
        with st.expander("Per-frame variance details"):
            frame_data = {
                "Frame": list(range(1, N + 1)),
                "S\u00b2 total": [f"{v:.4g}" for v in cached_ex["sd2_per_frame"]],
                "S\u00b2 div (relative)": [f"{v:.4g}" for v in cached_ex["sd2_div_per_frame"]],
            }
            st.table(frame_data)

    # CSV export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['filename', 'kerma', 'mpv', 'sd', 'ei'])
    for r in results["files"]:
        writer.writerow([r['filename'], r['kerma'], r['mpv'], r['sd'], r.get('ei')])
    st.download_button('Download results CSV', data=output.getvalue(), file_name='detector_conversion_results.csv')

    # Persist latest structured file summary each run
    dc_state["results"] = {
        "files": [{"filename": r["filename"], "kerma": r["kerma"], "mpv": r["mpv"],
                   "sd": r["sd"], "ei": r.get('ei')} for r in results["files"]]
    }
    st.session_state["detector_conversion"] = dc_state
    
    # Return structured results if detector curve was fitted this run
    if detector_curve_fitted:
        cached_fit = dc_state.get("fit", {})
        cached_ei = dc_state.get("ei_fit", {})
        returned = {
            "files": dc_state["results"]["files"],
            "fit": {
                "method": cached_fit.get("method"), "formula": cached_fit.get("formula"),
                "r2": cached_fit.get("r2"), "coeffs": cached_fit.get("coeffs"),
                "kerma": kerma_vals, "mpv": mpv_vals,
                "ei_fit": {"formula": cached_ei.get("formula"), "r2": cached_ei.get("r2"),
                          "coeffs": cached_ei.get("coeffs")}
            }
        }
        dc_state["results"] = returned
        st.session_state["detector_conversion"] = dc_state
        return returned
    return None
