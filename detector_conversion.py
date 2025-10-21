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
    st.write("Assign kerma value to each uploaded RAW file, compute central 100x100 MPV and SD, then fit MPV(kerma).")

    # If caller passed uploaded_files (from the sidebar), use them. Otherwise show a file uploader.
    if uploaded_files is None:
        uploaded = st.file_uploader("Upload RAW files", type=["raw"], accept_multiple_files=True)
    else:
        uploaded = uploaded_files

    if not uploaded:
        st.info("Upload one or more RAW files to begin")
        return None

    # Simple RAW param inputs
    dtype_str = st.selectbox("Pixel dtype", options=["uint8", "uint16", "float32"], index=1)
    dtype = np.uint16 if dtype_str == 'uint16' else (np.uint8 if dtype_str == 'uint8' else np.float32)

    # Kerma inputs per file
    st.markdown("---")
    st.write("Enter kerma  and EI values for each RAW file")
    kerma_vals = []
    results = {"files": []}
    for f in uploaded:
        fname = f.name
        # show kerma and new EI inputs side-by-side (unique keys per file)
        col_a, col_b = st.columns(2)
        with col_a:
            kerma_val = st.number_input(f"Kerma (μGy) — {fname}", value=0.0, key=f"kerma_{fname}")
        with col_b:
            ei_val = st.number_input(f"Exposition Index (EI) — {fname}", value=0.0, key=f"ei_{fname}")

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

            mpv, sd, roi = _central_roi_stats(arr, roi_h=100, roi_w=100)
            results["files"].append({
                "filename": fname,
                "kerma": float(kerma_val),
                "ei": float(ei_val),
                "mpv": float(mpv),
                "sd": float(sd),
            })
            st.write(f"MPV: {mpv:.3f}, SD: {sd:.3f}")
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")
            return None

    # Fit selection
    st.markdown("---")
    fit_method = st.selectbox("Fit method (applies to Detector Response Curve: MPV vs Kerma)", options=['linear', 'log', 'poly'], index=0, help="Select the fitting method for the Detector Response Curve (MPV vs Kerma). EI vs Kerma uses a linear fit.")
    poly_degree = 2
    if fit_method == 'poly':
        poly_degree = st.selectbox("Polynomial degree", options=[2, 3], index=0)

    # Prepare data
    mpv_vals = [r['mpv'] for r in results["files"]]

    try:
        fit_vals, formula, r2, p = _fit_mpv_vs_kerma(kerma_vals, mpv_vals, fit_method, poly_degree)
    except Exception as e:
        st.error(f"Fit failed: {e}")
        return None

    st.write("### Detector Response Curve")
    st.write(formula)
    st.write(f"R^2 = {r2:.4f}")

    # Compute percentage deviations of each measured MPV from the fitted value (reference = fitted point)
    fit_vals_arr = np.array(fit_vals, dtype=float)
    mpv_vals_arr = np.array(mpv_vals, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        deviations = 100.0 * (mpv_vals_arr - fit_vals_arr) / fit_vals_arr
    # Convert NaNs to None for nicer JSON/CSV handling and compute max absolute deviation
    deviations_list = [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in deviations]
    abs_devs = [abs(d) for d in deviations_list if d is not None]
    max_deviation_pct = float(max(abs_devs)) if abs_devs else None
    if max_deviation_pct is not None:
        st.write(f"Maximum absolute deviation from fit: {max_deviation_pct:.3f}%")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(kerma_vals, mpv_vals, label='data')
    ax.plot(kerma_vals, fit_vals, color='C1', label='fit')
    ax.set_xlabel('Kerma')
    ax.set_ylabel('MPV')
    ax.legend()
    st.pyplot(fig)

    # ----- EI vs Kerma fitting (same method) -----
    ei_vals = [r.get('ei', None) for r in results["files"]]
    # Ensure EI values are numeric
    try:
        ei_vals = [float(x) for x in ei_vals]
    except Exception:
        st.error("EI values are not numeric or missing for one or more files; cannot fit EI vs Kerma.")
        return None

    # EI vs Kerma uses a linear fit regardless of the MPV fit method
    try:
        fit_vals_ei, formula_ei, r2_ei, p_ei = _fit_mpv_vs_kerma(kerma_vals, ei_vals, 'linear', 1)
    except Exception as e:
        st.error(f"EI fit failed: {e}")
        return None

    st.write("### Exposition Index (EI) vs Kerma Fit")
    st.write(formula_ei)
    st.write(f"R^2 = {r2_ei:.4f}")

    # Compute deviations for EI (do this before plotting so we can display the max deviation above the graph)
    fit_vals_arr_ei = np.array(fit_vals_ei, dtype=float)
    ei_vals_arr = np.array(ei_vals, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        deviations_ei = 100.0 * (ei_vals_arr - fit_vals_arr_ei) / fit_vals_arr_ei
    deviations_list_ei = [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in deviations_ei]
    abs_devs_ei = [abs(d) for d in deviations_list_ei if d is not None]
    max_deviation_pct_ei = float(max(abs_devs_ei)) if abs_devs_ei else None
    if max_deviation_pct_ei is not None:
        st.write(f"Maximum absolute deviation from EI fit: {max_deviation_pct_ei:.3f}%")

    # Plot EI fit
    fig2, ax2 = plt.subplots()
    ax2.scatter(kerma_vals, ei_vals, label='data')
    ax2.plot(kerma_vals, fit_vals_ei, color='C2', label='fit')
    ax2.set_xlabel('Kerma')
    ax2.set_ylabel('Exposition Index (EI)')
    ax2.legend()
    st.pyplot(fig2)

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
            "formula": formula,
            "r2": float(r2) if not np.isnan(r2) else None,
            "coeffs": np.array(p).tolist() if p is not None else None,
            "kerma": [float(x) for x in kerma_vals],
            "mpv": [float(x) for x in mpv_vals],
            "fit_vals": [float(x) for x in fit_vals],
            "deviations_pct_mpv": [ (float(x) if x is not None else None) for x in deviations_list ],
            "max_deviation_pct_mpv": max_deviation_pct,
            "ei_fit": {
                "formula": formula_ei,
                "r2": float(r2_ei) if not np.isnan(r2_ei) else None,
                "coeffs": np.array(p_ei).tolist() if p_ei is not None else None,
                "ei": [float(x) for x in ei_vals],
                "fit_vals_ei": [float(x) for x in fit_vals_ei],
                "deviations_pct_ei": [ (float(x) if x is not None else None) for x in deviations_list_ei ],
                "max_deviation_pct_ei": max_deviation_pct_ei
            }
        }
    }

    return results_obj
