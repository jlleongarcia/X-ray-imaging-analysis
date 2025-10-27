import streamlit as st
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

"""
Detector conversion feature
- Upload multiple RAW files (simple RAW binary with known dtype and dimensions)
- User enters kerma value (mGy or any unit) per file via sidebar inputs
- For each file: extract a 100x100 central ROI, compute Mean Pixel Value (MPV) and SD
- Fit MPV(kerma) using one of: linear, logarithmic (MPV vs log(kerma)), polynomial (degree 2/3)
- Show fit parameters, R^2 and plot with data points and fit curve
"""


def _read_raw_bytes_to_array(raw_bytes, dtype, height, width):
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    if arr.size != height * width:
        raise ValueError(f"Raw data size mismatch: expected {height*width} elements, got {arr.size}")
    return arr.reshape((height, width))


def _central_roi_stats(img_array, roi_h=100, roi_w=100):
    H, W = img_array.shape
    if roi_h > H or roi_w > W:
        raise ValueError("Image smaller than requested ROI size")
    y0 = (H - roi_h) // 2
    x0 = (W - roi_w) // 2
    roi = img_array[y0:y0+roi_h, x0:x0+roi_w]
    mpv = float(np.mean(roi))
    sd = float(np.std(roi))
    return mpv, sd, roi


def _fit_mpv_vs_kerma(kerma_vals, mpv_vals, method, poly_degree=2):
    # Convert to numpy arrays
    k = np.array(kerma_vals, dtype=float)
    m = np.array(mpv_vals, dtype=float)

    if method == 'linear':
        # fit m = a*k + b
        p = np.polyfit(k, m, 1)
        fit_vals = np.polyval(p, k)
        formula = f"m = {p[0]:.4g} * k + {p[1]:.4g}"
    elif method == 'log':
        # fit m = a*log(k) + b  (use natural log); ignore non-positive kerma
        if np.any(k <= 0):
            raise ValueError("Kerma values must be positive for logarithmic fit")
        kk = np.log(k)
        p = np.polyfit(kk, m, 1)
        fit_vals = np.polyval(p, kk)
        formula = f"m = {p[0]:.4g} * ln(k) + {p[1]:.4g}"
    elif method == 'poly':
        p = np.polyfit(k, m, poly_degree)
        fit_vals = np.polyval(p, k)
        formula = "m = " + " + ".join([f"{coef:.4g}*k^{deg}" for deg, coef in enumerate(p[::-1])])
    else:
        raise ValueError("Unknown fit method")

    # R^2
    ss_res = np.sum((m - fit_vals) ** 2)
    ss_tot = np.sum((m - np.mean(m)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return fit_vals, formula, r2, p


def display_detector_conversion_section(uploaded_files=None):
    st.subheader("Detector conversion: MPV vs Kerma")
    st.write("Assign kerma value to each uploaded RAW/STD file, compute central 100x100 MPV and SD. Use the buttons below to run fits when ready.")

    # If caller passed uploaded_files (from the sidebar), use them. Otherwise show a file uploader.
    if uploaded_files is None:
        # Accept RAW and STD files (treat STD as RAW for reading)
        uploaded = st.file_uploader(
            "Upload RAW or STD files",
            type=["raw", "RAW", "std", "STD"],
            accept_multiple_files=True,
        )
    else:
        uploaded = uploaded_files

    if not uploaded:
        st.info("Upload one or more RAW/STD files to begin")
        return None

    # Simple RAW param inputs
    dtype_str = st.selectbox("Pixel dtype", options=["uint8", "uint16", "float32"], index=1)
    dtype = np.uint16 if dtype_str == 'uint16' else (np.uint8 if dtype_str == 'uint8' else np.float32)

    # Kerma inputs per file
    st.markdown("---")
    st.write("Enter kerma and EI values for each file")
    kerma_vals = []
    results = {"files": []}
    # Track extensions to decide default fit later
    uploaded_exts = []
    for f in uploaded:
        fname = f.name
        uploaded_exts.append((fname.split('.')[-1] if '.' in fname else '').lower())
        # show kerma and new EI inputs side-by-side (unique keys per file)
        col_a, col_b = st.columns(2)
        with col_a:
            kerma_val = st.number_input(f"Kerma (μGy) — {fname}", value=0.0, format="%.4f", key=f"kerma_{fname}")
        with col_b:
            ei_val = st.number_input(f"Exposition Index (EI) — {fname}", value=0.0, format="%.0f", key=f"ei_{fname}")

        kerma_vals.append(kerma_val)

        try:
            raw_bytes = f.getvalue()
            # Infer shape: try to form the largest square possible from the number of pixels
            pixels = np.frombuffer(raw_bytes, dtype=dtype)
            n_pixels = pixels.size

            if n_pixels < 100 * 100:
                raise ValueError(f"File {f.name} contains only {n_pixels} pixels (<10000); cannot extract 100x100 ROI")

            # Largest square side <= sqrt(n_pixels)
            side = int(np.floor(np.sqrt(n_pixels)))
            # Use only side*side pixels to form a square array
            use_pixels = pixels[: side * side ]
            arr = use_pixels.reshape((side, side))

            # If the square is smaller than 100x100 (shouldn't happen because we checked n_pixels), raise
            if side < 100:
                raise ValueError(f"Inferred square size {side} is smaller than 100; cannot extract 100x100 ROI")

            # Compute ROI stats
            mpv, sd, roi = _central_roi_stats(arr, roi_h=100, roi_w=100)
            results["files"].append({
                "filename": fname,
                "kerma": float(kerma_val),
                "ei": float(ei_val),
                "mpv": float(mpv),
                "sd": float(sd),
                # Store ROI pixels for later inverse-conversion based analyses
                "roi": roi,
            })
            st.write(f"MPV: {mpv:.3f}, SD: {sd:.3f}")
            # Proceed to next file; fit UI is shown after processing all files
            continue
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")
            return None
    # ------------------- Fit controls (deferred until button click) -------------------
    st.markdown("---")
        # --- Detector Response Curve (MPV vs Kerma) ---
    st.write("### Detector Response Curve")

        # Default behavior: STD files typically fit better with log; RAW with linear.
    # If all uploaded files have .std extension -> default to 'log'; else 'linear'. User can override.
    all_std = (len(uploaded_exts) > 0 and all(ext == 'std' for ext in uploaded_exts))
    default_index = 1 if all_std else 0  # 0: linear, 1: log, 2: poly
    fit_method = st.selectbox(
        "Fit method",
        options=['linear', 'log', 'poly'],
        index=default_index,
        help="RAW defaults to linear; STD defaults to log. You can change this if needed. EI vs Kerma is always linear.",
        key="fit_method_detector_curve",
    )
    poly_degree = 2
    if fit_method == 'poly':
        poly_degree = st.selectbox("Polynomial degree", options=[2, 3], index=0, key="poly_degree_detector_curve")

    # Prepare data arrays
    mpv_vals = [r['mpv'] for r in results["files"]]
    ei_vals = [r.get('ei', None) for r in results["files"]]
    try:
        ei_vals = [float(x) for x in ei_vals]
    except Exception:
        ei_vals = [None for _ in ei_vals]

    # Placeholders for fit results (to use for CSV/export if buttons not clicked)
    fit_vals = None
    formula = None
    r2 = None
    p = None
    deviations_list = [None] * len(results["files"])
    max_deviation_pct = None

    fit_vals_ei = None
    formula_ei = None
    r2_ei = None
    p_ei = None
    deviations_list_ei = [None] * len(results["files"])
    max_deviation_pct_ei = None

    if st.button("Run fit: Detector Response Curve", key="run_fit_detector_curve_out"):
        try:
            fit_vals, formula, r2, p = _fit_mpv_vs_kerma(kerma_vals, mpv_vals, fit_method, poly_degree)
            # Cache conversion function for later use
            def _build_mpv_from_kerma_fn(method, coeffs, poly_deg=2):
                coeffs = np.array(coeffs, dtype=float)
                if method == 'linear':
                    a, b = coeffs
                    return lambda k: a * np.asarray(k, dtype=float) + b
                elif method == 'log':
                    a, b = coeffs
                    def f(k):
                        k = np.asarray(k, dtype=float)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            return a * np.log(k) + b
                    return f
                else:
                    def f(k):
                        return np.polyval(coeffs, np.asarray(k, dtype=float))
                    return f

            mpv_from_kerma_fn = _build_mpv_from_kerma_fn(fit_method, p, poly_degree)
            st.session_state["detector_conversion"] = {
                "method": fit_method,
                "coeffs": np.array(p).tolist() if p is not None else None,
                "poly_degree": int(poly_degree) if fit_method == 'poly' else None,
                "formula": formula,
                "r2": float(r2) if (r2 is not None and not np.isnan(r2)) else None,
                "predict_mpv": mpv_from_kerma_fn,
            }
        except Exception as e:
            st.error(f"Fit failed: {e}")

    # Render cached detector response fit (if available)
    cached = st.session_state.get("detector_conversion")
    if isinstance(cached, dict) and cached.get("coeffs") is not None:
        st.caption("Fitted using: " + cached.get("method", "?"))
        st.write(cached.get("formula", ""))
        r2_val = cached.get("r2")
        if r2_val is not None:
            st.write(f"R^2 = {r2_val:.4f}")

        # Predictions and deviations
        try:
            pred_fn = cached.get("predict_mpv")
            if pred_fn is not None:
                fit_vals_disp = pred_fn(kerma_vals)
            else:
                coeffs = np.array(cached.get("coeffs"), dtype=float)
                method = cached.get("method")
                if method == 'log':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fit_vals_disp = coeffs[0] * np.log(np.asarray(kerma_vals, dtype=float)) + coeffs[1]
                elif method == 'linear':
                    fit_vals_disp = np.polyval(coeffs, np.asarray(kerma_vals, dtype=float))
                else:
                    fit_vals_disp = np.polyval(coeffs, np.asarray(kerma_vals, dtype=float))

            fit_vals_arr = np.array(fit_vals_disp, dtype=float)
            mpv_vals_arr = np.array(mpv_vals, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                deviations = 100.0 * (mpv_vals_arr - fit_vals_arr) / fit_vals_arr
            deviations_list = [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in deviations]
            abs_devs = [abs(d) for d in deviations_list if d is not None]
            max_deviation_pct = float(max(abs_devs)) if abs_devs else None
            if max_deviation_pct is not None:
                st.write(f"Maximum absolute deviation from fit: {max_deviation_pct:.3f}%")

            # Plot stored fit and data
            fig, ax = plt.subplots()
            ax.scatter(kerma_vals, mpv_vals, label='data')
            ax.plot(kerma_vals, fit_vals_arr, color='C1', label='fit')
            ax.set_xlabel('Kerma')
            ax.set_ylabel('MPV')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display cached detector fit: {e}")

    # --- Exposition Index (EI) vs Kerma ---
    st.write("### Exposition Index (EI) vs Kerma Fit")
    if st.button("Run fit: EI vs Kerma", key="run_fit_ei_curve_out"):
        # Only proceed if EI values are all numeric
        if any(v is None for v in ei_vals):
            st.error("EI values are not numeric or missing for one or more files; cannot fit EI vs Kerma.")
        else:
            try:
                fit_vals_ei, formula_ei, r2_ei, p_ei = _fit_mpv_vs_kerma(kerma_vals, ei_vals, 'linear', 1)
                # Cache EI fit (no conversion function needed)
                st.session_state["detector_ei_fit"] = {
                    "coeffs": np.array(p_ei).tolist() if p_ei is not None else None,
                    "formula": formula_ei,
                    "r2": float(r2_ei) if (r2_ei is not None and not np.isnan(r2_ei)) else None,
                }
            except Exception as e:
                st.error(f"EI fit failed: {e}")

    # Render cached EI vs Kerma fit (if available)
    cached_ei = st.session_state.get("detector_ei_fit")
    if isinstance(cached_ei, dict) and cached_ei.get("coeffs") is not None:
        st.caption("Fitted using: linear")
        st.write(cached_ei.get("formula", ""))
        r2_val_ei = cached_ei.get("r2")
        if r2_val_ei is not None:
            st.write(f"R^2 = {r2_val_ei:.4f}")

        try:
            coeffs_ei = np.array(cached_ei.get("coeffs"), dtype=float)
            fit_vals_arr_ei = np.polyval(coeffs_ei, np.asarray(kerma_vals, dtype=float))
            if all(v is not None for v in ei_vals):
                ei_vals_arr = np.array(ei_vals, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    deviations_ei = 100.0 * (ei_vals_arr - fit_vals_arr_ei) / fit_vals_arr_ei
                deviations_list_ei = [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in deviations_ei]
                abs_devs_ei = [abs(d) for d in deviations_list_ei if d is not None]
                max_deviation_pct_ei = float(max(abs_devs_ei)) if abs_devs_ei else None
                if max_deviation_pct_ei is not None:
                    st.write(f"Maximum absolute deviation from EI fit: {max_deviation_pct_ei:.3f}%")

            # Plot
            fig2, ax2 = plt.subplots()
            ax2.scatter(kerma_vals, ei_vals, label='data')
            ax2.plot(kerma_vals, fit_vals_arr_ei, color='C2', label='fit')
            ax2.set_xlabel('Kerma')
            ax2.set_ylabel('Exposition Index (EI)')
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not display cached EI fit: {e}")

    # --- Noise: SD^2 vs Kerma (uses inverse conversion) ---
    st.write("### Noise: SD^2 vs Kerma")
    st.caption("Compute SD on linearized (kerma-domain) ROI, square it (SD^2), then fit SD^2 = a*k^2 + b*k + c.")

    def _build_kerma_from_m_fn(conv: dict):
        method = conv.get("method")
        coeffs = np.array(conv.get("coeffs"), dtype=float)
        if method == 'linear':
            # m = a*k + b -> k = (m - b)/a
            a, b = coeffs
            if a == 0:
                raise ValueError("Inverse conversion undefined for a=0 in linear fit")
            return lambda m: (np.asarray(m, dtype=float) - b) / a
        elif method == 'log':
            # m = a*ln(k) + b -> k = exp((m - b)/a)
            a, b = coeffs
            if a == 0:
                raise ValueError("Inverse conversion undefined for a=0 in log fit")
            def f(m):
                m = np.asarray(m, dtype=float)
                with np.errstate(over='ignore', invalid='ignore'):
                    return np.exp((m - b) / a)
            return f
        else:
            # Inversion for polynomial fits is non-trivial; not implemented yet.
            raise NotImplementedError("Inverse conversion for 'poly' fit is not supported yet. Please use linear or log.")

    if st.button("Run fit: SD^2 vs Kerma", key="run_fit_sd2"):
        conv = st.session_state.get("detector_conversion")
        if not isinstance(conv, dict) or conv.get("coeffs") is None:
            st.error("Run the Detector Response Curve fit first to obtain the conversion function.")
        else:
            try:
                inv_fn = _build_kerma_from_m_fn(conv)
                sd2_vals = []
                for rec in results["files"]:
                    roi_px = rec.get("roi")
                    if roi_px is None:
                        st.error(f"Missing ROI data for {rec['filename']}")
                        return None
                    kerma_img = inv_fn(roi_px)
                    # Clean any invalids
                    kerma_img = np.asarray(kerma_img, dtype=float)
                    kerma_img[~np.isfinite(kerma_img)] = np.nan
                    # SD over finite values
                    sd_lin = float(np.nanstd(kerma_img))
                    k_in = float(rec["kerma"]) if rec.get("kerma") is not None else np.nan
                    sd2 = (sd_lin**2)
                    sd2_vals.append(sd2)

                # Remove NaNs for fitting; fit SD^2
                k_arr = np.array([float(x) for x in kerma_vals], dtype=float)
                y_arr = np.array(sd2_vals, dtype=float)
                mask = np.isfinite(k_arr) & np.isfinite(y_arr)
                if mask.sum() < 3:
                    st.error("Not enough valid points to fit a quadratic (need at least 3).")
                else:
                    p_sd = np.polyfit(k_arr[mask], y_arr[mask], 2)  # [a, b, c] for SD^2
                    y_fit = np.polyval(p_sd, k_arr)
                    a_, b_, c_ = p_sd
                    formula_sd = f"SD^2 = {a_:.4g}*k^2 + {b_:.4g}*k + {c_:.4g}"
                    # R^2
                    ss_res = np.nansum((y_arr - y_fit) ** 2)
                    ss_tot = np.nansum((y_arr - np.nanmean(y_arr)) ** 2)
                    r2_sd = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

                    # Dominance interval for middle term b*k over both a*k^2 and c
                    # Only compute if all coefficients are positive as requested
                    abc_positive = (a_ > 0) and (b_ > 0) and (c_ > 0)
                    k_min = None  # where b*k > c -> k > c/b
                    k_max = None  # where b*k > a*k^2 -> k < b/a
                    interval_exists = None
                    interval_degenerate = None
                    if abc_positive:
                        # Compute bounds
                        k_min = c_ / b_ if b_ != 0 else None
                        k_max = b_ / a_ if a_ != 0 else None
                        # Check if non-empty interval (c/b, b/a) exists: b^2 > a*c
                        if (k_min is not None) and (k_max is not None):
                            delta = (b_**2) - (a_*c_)
                            if np.isclose(delta, 0.0, rtol=1e-10, atol=1e-12):
                                interval_exists = False
                                interval_degenerate = True
                            elif delta > 0:
                                interval_exists = True
                                interval_degenerate = False
                            else:
                                interval_exists = False
                                interval_degenerate = False
                    # For backward compatibility keep upper_limit_k == k_max
                    upper_limit_k = k_max

                    # Cache results
                    st.session_state["detector_sd2_fit"] = {
                        "coeffs": p_sd.tolist(),
                        "formula": formula_sd,
                        "r2": float(r2_sd) if not np.isnan(r2_sd) else None,
                        "sd2": [None if not np.isfinite(v) else float(v) for v in y_arr],
                        "abc_positive": bool(abc_positive),
                        "upper_limit_k": (float(upper_limit_k) if upper_limit_k is not None else None),
                        "k_min": (float(k_min) if k_min is not None else None),
                        "k_max": (float(k_max) if k_max is not None else None),
                        "dominance_interval_exists": (bool(interval_exists) if interval_exists is not None else None),
                        "dominance_interval_degenerate": (bool(interval_degenerate) if interval_degenerate is not None else None),
                    }
            except NotImplementedError as nie:
                st.error(str(nie))
            except Exception as e:
                st.error(f"SD_norm fit failed: {e}")

    # Render cached SD^2 fit (if available)
    cached_sd = st.session_state.get("detector_sd2_fit")
    if isinstance(cached_sd, dict) and cached_sd.get("coeffs") is not None:
        st.write(cached_sd.get("formula", ""))
        r2_sd = cached_sd.get("r2")
        if r2_sd is not None:
            st.write(f"R^2 = {r2_sd:.4f}")

        # Retrieve dominance interval details from cache
        abc_positive = cached_sd.get("abc_positive", None)
        k_min = cached_sd.get("k_min", None)
        k_max = cached_sd.get("k_max", None)
        interval_exists = cached_sd.get("dominance_interval_exists", None)
        interval_degenerate = cached_sd.get("dominance_interval_degenerate", None)

        # Immediate user feedback for the dominance interval
        if not abc_positive:
            st.info("Coefficients a, b, and c are not all positive; dominance interval of the quantum noise is not computed.")
        else:
            if (k_min is not None and k_max is not None and k_min > 0 and k_max > 0):
                if interval_exists:
                    st.write(f"Quantum noise dominance interval (over both structural and electronic noise): (k_min, k_max) = ({k_min:.4g}, {k_max:.4g}) μGy")
                elif interval_degenerate:
                    st.info(f"No dominance interval (degenerate): k_min = k_max = {k_min:.4g} μGy (b^2 ≈ a*c)")
                else:
                    st.info("No kerma value where the quantum noise term dominates both other terms (b^2 ≤ a*c).")
            else:
                st.info("Dominance interval bounds could not be determined (non-positive or undefined).")

        # Recompute SD_norm scatter with current data for visualization
        conv = st.session_state.get("detector_conversion")
        try:
            inv_fn = _build_kerma_from_m_fn(conv)
            sd2_vals = []
            for rec in results["files"]:
                kerma_img = inv_fn(rec.get("roi"))
                kerma_img = np.asarray(kerma_img, dtype=float)
                kerma_img[~np.isfinite(kerma_img)] = np.nan
                sd_lin = float(np.nanstd(kerma_img))
                sd2 = (sd_lin**2)
                sd2_vals.append(sd2)

            k_arr = np.array([float(x) for x in kerma_vals], dtype=float)
            y_arr = np.array(sd2_vals, dtype=float)
            coeffs_sd = np.array(cached_sd.get("coeffs"), dtype=float)
            y_fit = np.polyval(coeffs_sd, k_arr)

            fig3, ax3 = plt.subplots()
            ax3.scatter(k_arr, y_arr, label='SD^2 data')
            # Sort kerma for smooth curve
            order = np.argsort(k_arr)
            ax3.plot(k_arr[order], y_fit[order], color='C3', label='quadratic fit')
            ax3.set_xlabel('Kerma')
            ax3.set_ylabel('SD^2')
            ax3.legend()
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Could not display cached SD_norm fit: {e}")

    # Offer CSV download of results
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['filename', 'kerma', 'mpv', 'sd', 'deviation_pct_mpv', 'ei', 'deviation_pct_ei'])
    for k, r, dev_mpv, dev_ei in zip(kerma_vals, results["files"], deviations_list, deviations_list_ei):
        writer.writerow([r['filename'], k, r['mpv'], r['sd'], dev_mpv, r.get('ei'), dev_ei])
    st.download_button('Download results CSV', data=output.getvalue(), file_name='detector_conversion_results.csv')

    # Build a structured results object to return so callers can persist it to session state
    results_obj = {
        "files": [
            {"filename": r["filename"], "kerma": float(k), "mpv": float(r["mpv"]), "sd": float(r["sd"]), "ei": float(r.get('ei')) if r.get('ei') is not None else None, "deviation_pct_mpv": (float(dmpv) if dmpv is not None else None), "deviation_pct_ei": (float(dei) if dei is not None else None)}
            for k, r, dmpv, dei in zip(kerma_vals, results["files"], deviations_list, deviations_list_ei)
        ],
        "fit": {
            "method": fit_method,
            "formula": formula if formula is not None else None,
            "r2": (float(r2) if (r2 is not None and not np.isnan(r2)) else None),
            "coeffs": (np.array(p).tolist() if p is not None else None),
            "kerma": [float(x) for x in kerma_vals],
            "mpv": [float(x) for x in mpv_vals],
            "fit_vals": ([float(x) for x in fit_vals] if fit_vals is not None else None),
            "deviations_pct_mpv": [ (float(x) if x is not None else None) for x in deviations_list ],
            "max_deviation_pct_mpv": max_deviation_pct,
            "ei_fit": {
                "formula": formula_ei if formula_ei is not None else None,
                "r2": (float(r2_ei) if (r2_ei is not None and not np.isnan(r2_ei)) else None),
                "coeffs": (np.array(p_ei).tolist() if p_ei is not None else None),
                "ei": ([float(x) for x in ei_vals] if all(v is not None for v in ei_vals) else None),
                "fit_vals_ei": ([float(x) for x in fit_vals_ei] if fit_vals_ei is not None else None),
                "deviations_pct_ei": [ (float(x) if x is not None else None) for x in deviations_list_ei ],
                "max_deviation_pct_ei": max_deviation_pct_ei
            }
        }
    }

    return results_obj
