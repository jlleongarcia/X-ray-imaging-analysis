import streamlit as st
import numpy as np
import io
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

"""
Detector conversion feature: Upload RAW/STD files, assign kerma values, extract ROI stats,
fit MPV vs Kerma curves (linear/log/poly), and analyze noise characteristics.
"""

# ==================== HELPER FUNCTIONS ====================

def _read_raw_as_square(raw_bytes, dtype):
    """Read raw bytes and reshape into largest possible square array."""
    pixels = np.frombuffer(raw_bytes, dtype=dtype)
    side = int(np.floor(np.sqrt(pixels.size)))
    if side < 100:
        raise ValueError(f"Inferred square size {side} < 100; cannot extract 100x100 ROI")
    return pixels[:side*side].reshape((side, side))


def _central_roi_stats(img_array, roi_h=100, roi_w=100):
    """Extract central ROI and compute mean pixel value (MPV) and standard deviation."""
    H, W = img_array.shape
    if roi_h > H or roi_w > W:
        raise ValueError("Image smaller than requested ROI size")
    y0, x0 = (H - roi_h) // 2, (W - roi_w) // 2
    roi = img_array[y0:y0+roi_h, x0:x0+roi_w]
    return float(np.mean(roi)), float(np.std(roi)), roi


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

def _render_cached_fit(cache_key, title, x_label, y_label):
    """Render a cached fit with formula, R², deviations, and plot."""
    cached = st.session_state.get(cache_key)
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
        ax.scatter(cached["x_data"], cached["y_data"], label='data')
        ax.plot(cached["x_data"], cached["y_fit"], color='C1', label='fit')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        st.pyplot(fig)

def display_detector_conversion_section(uploaded_files=None):
    st.subheader("Detector conversion: MPV vs Kerma")
    st.write("Assign kerma value to each uploaded RAW/STD file, compute central 100x100 MPV and SD. Use the buttons below to run fits when ready.")

    uploaded = uploaded_files if uploaded_files else st.file_uploader(
        "Upload RAW or STD files", type=["raw", "RAW", "std", "STD"], accept_multiple_files=True
    )
    if not uploaded:
        st.info("Upload one or more RAW/STD files to begin")
        return None

    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
    dtype_str = st.selectbox("Pixel dtype", options=list(dtype_map.keys()), index=1)
    dtype = dtype_map[dtype_str]

    st.markdown("---")
    st.write("Enter kerma and EI values for each file")
    kerma_vals, results = [], {"files": []}
    uploaded_exts = [(f.name.split('.')[-1] if '.' in f.name else '').lower() for f in uploaded]
    
    for f in uploaded:
        fname = f.name
        col_a, col_b = st.columns(2)
        with col_a:
            kerma_val = st.number_input(f"Kerma (μGy) — {fname}", value=0.0, format="%.4f", key=f"kerma_{fname}")
        with col_b:
            ei_val = st.number_input(f"Exposition Index (EI) — {fname}", value=0.0, format="%.2f", key=f"ei_{fname}")
        kerma_vals.append(kerma_val)
        
        try:
            arr = _read_raw_as_square(f.getvalue(), dtype)
            mpv, sd, roi = _central_roi_stats(arr)
            results["files"].append({
                "filename": fname, "kerma": float(kerma_val), "ei": float(ei_val),
                "mpv": float(mpv), "sd": float(sd), "roi": roi
            })
            st.write(f"MPV: {mpv:.3f}, SD: {sd:.3f}")
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")
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
            
            st.session_state["detector_conversion"] = {
                "method": fit_method, "coeffs": p.tolist(), 
                "poly_degree": int(poly_degree) if fit_method == 'poly' else None,
                "formula": formula, "latex_formula": _latex_formula(fit_method, p, poly_degree),
                "r2": float(r2) if not np.isnan(r2) else None, "predict_mpv": pred_fn,
                "x_data": kerma_vals, "y_data": mpv_vals, "y_fit": fit_vals.tolist(),
                "max_deviation": max_dev
            }
            detector_curve_fitted = True
        except Exception as e:
            st.error(f"Fit failed: {e}")

    _render_cached_fit("detector_conversion", "Detector Response", r"$k$", "MPV")

    # --- Exposition Index (EI) vs Kerma ---
    st.write("### Exposition Index (EI) vs Kerma Fit")
    if st.button("Run fit: EI vs Kerma", key="run_fit_ei_curve_out"):
        if any(v is None for v in ei_vals):
            st.error("EI values missing for one or more files; cannot fit EI vs Kerma.")
        else:
            try:
                fit_vals_ei, formula_ei, r2_ei, p_ei = _fit_mpv_vs_kerma(kerma_vals, ei_vals, 'linear', 1)
                deviations_ei, max_dev_ei = _compute_deviations(ei_vals, fit_vals_ei)
                st.session_state["detector_ei_fit"] = {
                    "method": "linear", "coeffs": p_ei.tolist(), "formula": formula_ei,
                    "latex_formula": rf"EI = {p_ei[0]:.4g}\,k + {p_ei[1]:.4g}",
                    "r2": float(r2_ei) if not np.isnan(r2_ei) else None,
                    "x_data": kerma_vals, "y_data": ei_vals, "y_fit": fit_vals_ei.tolist(),
                    "max_deviation": max_dev_ei
                }
            except Exception as e:
                st.error(f"EI fit failed: {e}")
    
    _render_cached_fit("detector_ei_fit", "EI vs Kerma", r"$k$", "Exposition Index (EI)")

    # --- Noise: SD² vs Kerma ---
    st.write("### Noise: SD² vs Kerma")
    st.caption("Compute SD on linearized (kerma-domain) ROI, square it (SD²), then fit SD² = a·k² + b·k + c.")
    
    if st.button("Run fit: SD² vs Kerma", key="run_fit_sd2"):
        conv = st.session_state.get("detector_conversion")
        if not (isinstance(conv, dict) and conv.get("coeffs")):
            st.error("Run the Detector Response Curve fit first.")
        else:
            try:
                inv_fn = _build_inverse_fn(conv)
                sd2_vals = []
                for rec in results["files"]:
                    if rec.get("roi") is None:
                        st.error(f"Missing ROI data for {rec['filename']}")
                        return None
                    kerma_img = np.asarray(inv_fn(rec["roi"]), dtype=float)
                    kerma_img[~np.isfinite(kerma_img)] = np.nan
                    _, std_kerma, _ = _central_roi_stats(kerma_img)
                    sd2_vals.append(float(std_kerma)**2)
                
                k_arr, y_arr = np.array(kerma_vals, dtype=float), np.array(sd2_vals, dtype=float)
                mask = np.isfinite(k_arr) & np.isfinite(y_arr) & (k_arr > 0)
                if mask.sum() < 3:
                    st.error("Need at least 3 valid points to fit a quadratic.")
                else:
                    # Weighted RLM fit: SD² = a·k² + b·k + c with weights = 1/k²
                    k_valid, y_valid = k_arr[mask], y_arr[mask]
                    
                    # Design matrix: [k², k, 1]
                    X = np.column_stack([k_valid**2, k_valid, np.ones_like(k_valid)])
                    
                    # Weights: 1/k²
                    weights = 1.0 / (k_valid**2)
                    
                    # Fit Weighted RLM
                    rlm_model = sm.RLM(y_valid, X, M=sm.robust.norms.HuberT(), weights=weights)
                    rlm_results = rlm_model.fit(scale_est=sm.robust.scale.HuberScale())
                    
                    # Extract coefficients: [a, b, c] for SD² = a·k² + b·k + c
                    a_, b_, c_ = rlm_results.params
                    
                    # Compute fitted values for all points (including masked)
                    X_all = np.column_stack([k_arr**2, k_arr, np.ones_like(k_arr)])
                    y_fit = X_all @ rlm_results.params
                    
                    # R² calculation
                    ss_res = np.nansum((y_arr - y_fit)**2)
                    ss_tot = np.nansum((y_arr - np.nanmean(y_arr))**2)
                    r2_sd = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    
                    # Dominance interval computation
                    abc_positive = (a_ > 0) and (b_ > 0) and (c_ > 0)
                    k_min = c_ / b_ if (abc_positive and b_ != 0) else None
                    k_max = b_ / a_ if (abc_positive and a_ != 0) else None
                    interval_exists, interval_degenerate = None, None
                    if abc_positive and k_min and k_max:
                        delta = b_**2 - a_*c_
                        if np.isclose(delta, 0.0, rtol=1e-10, atol=1e-12):
                            interval_exists, interval_degenerate = False, True
                        elif delta > 0:
                            interval_exists, interval_degenerate = True, False
                        else:
                            interval_exists, interval_degenerate = False, False
                    
                    st.session_state["detector_sd2_fit"] = {
                        "coeffs": [float(a_), float(b_), float(c_)], "formula": f"SD² = {a_:.4g}·k² + {b_:.4g}·k + {c_:.4g}",
                        "latex_formula": rf"SD² = {a_:.4g}\,k² + {b_:.4g}\,k + {c_:.4g}",
                        "r2": float(r2_sd) if not np.isnan(r2_sd) else None,
                        "sd2": [None if not np.isfinite(v) else float(v) for v in y_arr],
                        "abc_positive": bool(abc_positive), "upper_limit_k": (float(k_max) if k_max else None),
                        "k_min": (float(k_min) if k_min else None), "k_max": (float(k_max) if k_max else None),
                        "dominance_interval_exists": (bool(interval_exists) if interval_exists is not None else None),
                        "dominance_interval_degenerate": (bool(interval_degenerate) if interval_degenerate is not None else None),
                    }
            except NotImplementedError as nie:
                st.error(str(nie))
            except Exception as e:
                st.error(f"SD_norm fit failed: {e}")

    # Render cached SD² fit
    cached_sd = st.session_state.get("detector_sd2_fit")
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
        
        # Data table
        st.write("**Data from uploaded images:**")
        df_table = pd.DataFrame([{
            "Kerma (μGy)": float(r.get("kerma", 0)),
            "Mean Value": float(r.get("mpv", 0)),
            "SD Value": float(r.get("sd", 0))
        } for r in results["files"]])
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Plot SD² components
        try:
            conv = st.session_state.get("detector_conversion")
            inv_fn = _build_inverse_fn(conv)
            sd2_vals = [float(np.nanstd(np.where(np.isfinite(x:=np.asarray(inv_fn(rec["roi"]), dtype=float)), x, np.nan)))**2 
                       for rec in results["files"]]
            
            k_arr, y_arr = np.array(kerma_vals, dtype=float), np.array(sd2_vals, dtype=float)
            a_, b_, c_ = np.array(cached_sd["coeffs"], dtype=float)
            y_fit = np.polyval([a_, b_, c_], k_arr)
            order = np.argsort(k_arr)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.scatter(k_arr, y_arr, label='SD² data', color='black', s=50, zorder=5)
            ax3.plot(k_arr[order], y_fit[order], color='C3', linewidth=2, label='Total: $a·k² + b·k + c$')
            ax3.plot(k_arr[order], (a_*k_arr**2)[order], '--', color='C0', linewidth=1.5, label=f'Structural: ${a_:.4g}·k²$')
            ax3.plot(k_arr[order], (b_*k_arr)[order], '--', color='C2', linewidth=1.5, label=f'Quantum: ${b_:.4g}·k$')
            ax3.axhline(c_, linestyle='--', color='C1', linewidth=1.5, label=f'Electronic: ${c_:.4g}$')
            ax3.set_xlabel(r"$k$ (μGy)", fontsize=12)
            ax3.set_ylabel(r"$SD²$", fontsize=12)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            
            # Normalized noise plot: SD²/k_real² vs Kerma
            st.write("**Normalized noise: SD²/k_real² vs Kerma**")
            st.caption("k_real: kerma from MPV using detector response curve.")
            try:
                cached = st.session_state.get("detector_conversion")
                if not cached or not cached.get("predict_mpv"):
                    st.warning("Detector response curve not available.")
                else:
                    kerma_real_arr = np.array([inv_fn(np.array([r["mpv"]]))[0] for r in results["files"]], dtype=float)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_normalized = y_arr / (kerma_real_arr**2)
                    
                    valid_mask = np.isfinite(y_normalized) & (k_arr > 0) & (kerma_real_arr > 0)
                    k_valid, y_norm_valid = k_arr[valid_mask], y_normalized[valid_mask]
                    
                    if len(k_valid) > 0:
                        k_inv = 1.0 / k_valid
                        p_norm = np.polyfit(k_inv, y_norm_valid, 2)
                        y_norm_fit = np.polyval(p_norm, k_inv)
                        c_coeff, b_coeff, a_coeff = p_norm  # c/k², b/k, a
                        
                        r2_norm = 1 - np.sum((y_norm_valid - y_norm_fit)**2) / np.sum((y_norm_valid - np.mean(y_norm_valid))**2)
                        
                        fig4, ax4 = plt.subplots(figsize=(10, 6))
                        ax4.scatter(k_valid, y_norm_valid, label='SD²/k_real² data', color='black', s=50, zorder=5)
                        order_norm = np.argsort(k_valid)
                        ax4.plot(k_valid[order_norm], y_norm_fit[order_norm], color='C3', linewidth=2, label='Fit: $a + b/k + c/k²$')
                        ax4.set_xlabel(r"$k$ (μGy)", fontsize=12)
                        ax4.set_ylabel(r"$SD²/k_{\mathrm{real}}²$", fontsize=12)
                        ax4.legend(loc='best', fontsize=10)
                        ax4.grid(True, alpha=0.3)
                        
                        st.latex(rf"SD²/k_{{\mathrm{{real}}}}² = {a_coeff:.4g} + {b_coeff:.4g}/k + {c_coeff:.4g}/k²")
                        st.write(f"R² = {r2_norm:.4f}")
                        
                        # Dominance interval
                        if (a_coeff > 0) and (b_coeff > 0) and (c_coeff > 0):
                            k_min_norm = c_coeff / b_coeff if b_coeff != 0 else None
                            k_max_norm = b_coeff / a_coeff if a_coeff != 0 else None
                            if k_min_norm and k_max_norm:
                                delta_norm = b_coeff**2 - a_coeff*c_coeff
                                if np.isclose(delta_norm, 0.0, rtol=1e-10, atol=1e-12):
                                    st.latex(rf"k_\mathrm{{min}} = k_\mathrm{{max}} = {k_min_norm:.4g}\;\mu\,Gy\quad (b^2 \approx a\cdot c)")
                                elif delta_norm > 0:
                                    st.write("Quantum noise dominance interval (normalized):")
                                    st.latex(rf"({k_min_norm:.4g},\,{k_max_norm:.4g})\;\mu\,Gy")
                                else:
                                    st.latex(r"b^2 \le a\cdot c\quad\text{(no dominance)}")
                        st.pyplot(fig4)
            except Exception as e:
                st.warning(f"Could not display normalized plot: {e}")
        except Exception as e:
            st.warning(f"Could not display cached SD fit: {e}")
    
    # CSV export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['filename', 'kerma', 'mpv', 'sd', 'ei'])
    for r in results["files"]:
        writer.writerow([r['filename'], r['kerma'], r['mpv'], r['sd'], r.get('ei')])
    st.download_button('Download results CSV', data=output.getvalue(), file_name='detector_conversion_results.csv')
    
    # Return structured results if detector curve was fitted this run
    if detector_curve_fitted:
        cached = st.session_state.get("detector_conversion", {})
        cached_ei = st.session_state.get("detector_ei_fit", {})
        return {
            "files": [{"filename": r["filename"], "kerma": r["kerma"], "mpv": r["mpv"], 
                      "sd": r["sd"], "ei": r.get('ei')} for r in results["files"]],
            "fit": {
                "method": cached.get("method"), "formula": cached.get("formula"),
                "r2": cached.get("r2"), "coeffs": cached.get("coeffs"),
                "kerma": kerma_vals, "mpv": mpv_vals,
                "ei_fit": {"formula": cached_ei.get("formula"), "r2": cached_ei.get("r2"), 
                          "coeffs": cached_ei.get("coeffs")}
            }
        }
    return None
