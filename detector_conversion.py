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
