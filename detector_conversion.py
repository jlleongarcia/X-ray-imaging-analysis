import streamlit as st
import numpy as np
import io
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import least_squares
from raw_endian import frombuffer_with_endian

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


def _bootstrap_variance(detrended_roi, n_bootstrap=500):
    """Estimate variance of detrended ROI using bootstrap resampling.
    
    Treats the pure noise ROI as a "bag of marbles" and repeatedly samples
    with replacement to robustly estimate the variance.
    
    Args:
        detrended_roi: 2D array of detrended (pure noise) pixel values
        n_bootstrap: Number of bootstrap iterations (default 500)
    
    Returns:
        Bootstrap variance estimate (median of bootstrap variances)
    """
    # Flatten and remove invalid values
    pixels = detrended_roi.flatten()
    valid_pixels = pixels[np.isfinite(pixels)]
    
    if len(valid_pixels) < 2:
        return np.nan
    
    n = len(valid_pixels)
    bootstrap_variances = []
    
    # Bootstrap resampling
    for _ in range(n_bootstrap):
        # Sample n pixels with replacement
        sample = np.random.choice(valid_pixels, size=n, replace=True)
        # Calculate variance of this sample
        bootstrap_variances.append(np.var(sample, ddof=1))
    
    # Return variance of bootstrap variances (accurate measure of uncertainty)
    return float(np.var(bootstrap_variances, ddof=1))


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
    """Compute percentage deviations and return list with max absolute deviation."""
    actual, fitted = np.array(actual, dtype=float), np.array(fitted, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        devs = 100.0 * (actual - fitted) / fitted
    dev_list = [None if np.isnan(x) else float(x) for x in devs]
    abs_devs = [abs(d) for d in dev_list if d is not None]
    return dev_list, (max(abs_devs) if abs_devs else None)

def _constrained_weighted_fit(k_valid, y_valid, weights, n_starts=4):
    """Perform constrained weighted least squares fit with multi-start verification.
    
    Args:
        k_valid: Valid kerma values
        y_valid: Valid y values (σ²)
        weights: Weights for each point
        n_starts: Number of initial guesses (default 4)
    
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
            loss='soft_l1', f_scale=1.0
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

def _get_detector_conversion_state() -> dict:
    """Get unified detector conversion state container from session.

    Structure:
      detector_conversion = {
        'fit': {...},
        'results': {...} | None,
        'ei_fit': {...},
        'sd2_fit': {...}
      }
    """
    state = st.session_state.get("detector_conversion")
    if not isinstance(state, dict):
        state = {}
    state.setdefault("fit", {})
    state.setdefault("results", None)
    state.setdefault("ei_fit", {})
    state.setdefault("sd2_fit", {})
    st.session_state["detector_conversion"] = state
    return state


def _file_name_and_bytes(file_obj):
    """Return (filename, bytes) for Streamlit UploadedFile or preloaded payload dict."""
    if isinstance(file_obj, dict):
        return file_obj.get('name', 'unknown'), file_obj.get('bytes', b'')
    return getattr(file_obj, 'name', 'unknown'), file_obj.getvalue()


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
        st.write(f"Maximum absolute deviation from fit: {max_dev:.3f}%")
    
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

def display_detector_conversion_section(uploaded_files=None):
    st.subheader("Detector conversion: MPV vs Kerma")
    st.write("Assign kerma value to each uploaded RAW/STD file, compute central 100x100 MPV and σ. Use the buttons below to run fits when ready.")

    uploaded = uploaded_files if uploaded_files else st.file_uploader(
        "Upload RAW or STD files", type=["raw", "RAW", "std", "STD"], accept_multiple_files=True
    )
    if not uploaded:
        st.info("Upload one or more RAW/STD files to begin")
        return None

    dc_state = _get_detector_conversion_state()

    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
    dtype_str = st.selectbox("Pixel dtype", options=list(dtype_map.keys()), index=1)
    dtype = dtype_map[dtype_str]
    default_little_endian = bool(st.session_state.get("raw_little_endian_default", True))

    st.markdown("---")
    st.write("Enter kerma and EI values for each file")
    kerma_vals, results = [], {"files": []}
    uploaded_exts = []
    for f in uploaded:
        fname, _ = _file_name_and_bytes(f)
        uploaded_exts.append((fname.split('.')[-1] if '.' in fname else '').lower())
    
    for f in uploaded:
        fname, fbytes = _file_name_and_bytes(f)
        col_a, col_b = st.columns(2)
        with col_a:
            kerma_val = st.number_input(f"Kerma (μGy) — {fname}", value=0.0, format="%.4f", key=f"kerma_{fname}")
        with col_b:
            ei_val = st.number_input(f"Exposition Index (EI) — {fname}", value=0.0, format="%.2f", key=f"ei_{fname}")
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
                "mpv": float(mpv), "sd": float(sd), "roi": roi
            })
            st.write(f"MPV: {mpv:.3f}, σ: {sd:.3f}")
            st.caption(f"Endian used: {'little' if endian_used else 'big'} ({endian_source})")
        except Exception as e:
            st.error(f"Failed to process {fname}: {e}")
            return None
    
    st.markdown("---")
    # --- Detector Response Curve (MPV vs Kerma) ---
    st.write("### Detector Response Curve")
    
    all_std = len(uploaded_exts) > 0 and all(ext == 'std' for ext in uploaded_exts)
    fit_method = st.selectbox(
        "Fit method", options=['linear', 'log', 'poly'], index=1 if all_std else 0,
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
    
    if st.button("Run fit: σ² vs Kerma", key="run_fit_sd2"):
        conv = dc_state.get("fit")
        if not (isinstance(conv, dict) and conv.get("coeffs")):
            st.error("Run the Detector Response Curve fit first.")
        else:
            try:
                inv_fn = _build_inverse_fn(conv)
                sd2_vals = []
                bootstrap_vars = []  # Store bootstrap variance estimates for weights
                
                for rec in results["files"]:
                    if rec.get("roi") is None:
                        st.error(f"Missing ROI data for {rec['filename']}")
                        return None
                    kerma_img = np.asarray(inv_fn(rec["roi"]), dtype=float)
                    kerma_img[~np.isfinite(kerma_img)] = np.nan
                    
                    # Extract central 100x100 ROI from kerma image
                    _, _, roi_kerma = _central_roi_stats(kerma_img)
                    
                    # Detrend to remove heel effect and geometric dome (keep only noise)
                    roi_detrended = _detrend_roi(roi_kerma)
                    
                    # Bootstrap variance estimation for robust weight calculation
                    boot_var = _bootstrap_variance(roi_detrended, n_bootstrap=500)
                    bootstrap_vars.append(boot_var)
                    
                    # Compute std of detrended (pure noise) ROI
                    std_kerma = float(np.nanstd(roi_detrended))
                    sd2_vals.append(std_kerma**2)
                
                k_arr, y_arr = np.array(kerma_vals, dtype=float), np.array(sd2_vals, dtype=float)
                boot_var_arr = np.array(bootstrap_vars, dtype=float)
                
                mask = np.isfinite(k_arr) & np.isfinite(y_arr) & np.isfinite(boot_var_arr) & (k_arr > 0) & (boot_var_arr > 0)
                if mask.sum() < 3:
                    st.error("Need at least 3 valid points to fit a quadratic.")
                else:
                    k_valid, y_valid = k_arr[mask], y_arr[mask]
                    boot_var_valid = boot_var_arr[mask]
                    weights = 1.0 / boot_var_valid
                    
                    st.info("Using bootstrap variance estimates (n=500) with non-negative constrained fitting.")
                    
                    # Perform constrained weighted fit with multi-start verification
                    a_, b_, c_, success_msg, warning_msg = _constrained_weighted_fit(k_valid, y_valid, weights)
                    
                    if a_ is None:
                        st.error(warning_msg)
                        return None
                    
                    # Display convergence messages
                    if warning_msg:
                        st.warning(warning_msg)
                    if success_msg:
                        st.success(success_msg)
                    
                    # Compute fitted values and R²
                    y_fit = a_ * k_arr**2 + b_ * k_arr + c_
                    ss_res = np.nansum((y_arr - y_fit)**2)
                    ss_tot = np.nansum((y_arr - np.nanmean(y_arr))**2)
                    r2_sd = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    
                    # Compute dominance interval
                    interval_info = _compute_dominance_interval(a_, b_, c_)
                    
                    # Store results
                    dc_state["sd2_fit"] = {
                        "coeffs": [float(a_), float(b_), float(c_)],
                        "formula": f"σ² = {a_:.4g}·k² + {b_:.4g}·k + {c_:.4g}",
                        "latex_formula": rf"\sigma² = {a_:.4g}\,k² + {b_:.4g}\,k + {c_:.4g}",
                        "r2": float(r2_sd) if not np.isnan(r2_sd) else None,
                        "sd2": [None if not np.isfinite(v) else float(v) for v in y_arr],
                        **interval_info
                    }
                    st.session_state["detector_conversion"] = dc_state
            except NotImplementedError as nie:
                st.error(str(nie))
            except Exception as e:
                st.error(f"SD_norm fit failed: {e}")

    # Render cached SD² fit
    cached_sd = dc_state.get("sd2_fit")
    if isinstance(cached_sd, dict) and cached_sd.get("coeffs"):
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
            k_arr, y_arr = np.array(kerma_vals, dtype=float), np.array(cached_sd.get("sd2", []), dtype=float)
            a_, b_, c_ = np.array(cached_sd["coeffs"], dtype=float)
            
            # Generate smooth curve for better visualization
            k_smooth = np.linspace(k_arr.min(), k_arr.max(), 200)
            y_fit_smooth = a_ * k_smooth**2 + b_ * k_smooth + c_
            structural_smooth = a_ * k_smooth**2
            quantum_smooth = b_ * k_smooth
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.scatter(k_arr, y_arr, label='σ² data', color='black', s=50, zorder=5)
            ax3.plot(k_smooth, y_fit_smooth, color='C3', linewidth=2, label='Total: $a·k² + b·k + c$')
            ax3.plot(k_smooth, structural_smooth, '--', color='C0', linewidth=1.5, label=f'Structural: ${a_:.4g}·k²$')
            ax3.plot(k_smooth, quantum_smooth, '--', color='C2', linewidth=1.5, label=f'Quantum: ${b_:.4g}·k$')
            ax3.axhline(c_, linestyle='--', color='C1', linewidth=1.5, label=f'Electronic: ${c_:.4g}$')
            ax3.set_xlabel(r"$k$ (μGy)", fontsize=12)
            ax3.set_ylabel(r"$\sigma²$", fontsize=12)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Could not display cached SD fit: {e}")
    
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
