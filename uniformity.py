import numpy as np
import streamlit as st
import pydicom
from PIL import Image # Used for displaying images in Streamlit
import os
import tempfile

def _calculate_uniformity_term(val_i, val_mean):
    """
    Calculates the term abs(val_i - val_mean) / abs(val_mean) for uniformity.
    Assumes val_mean is non-negative (typical for pixel values and standard deviations).

    Args:
        val_i (float): The individual ROI value (e.g., PV_i or SD_i).
        val_mean (float): The reference mean value (e.g., MeanPV, PV_8n, MeanSD, SD_8n).

    Returns:
        float: The calculated uniformity term. Can be 0.0, a positive float, or np.inf.
    """

    # Both val_i and val_mean are finite at this point
    abs_diff = np.abs(val_i - val_mean)
    denominator = np.abs(val_mean)

    # Guard against zero or non-finite denominators which would make the term undefined
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan

    return abs_diff / denominator

def calculate_xray_uniformity_metrics(image_array, pixel_spacing_row, pixel_spacing_col):
    """
    Calculates uniformity metrics for an X-ray image.

    The process involves:
    1. Defining a central ROI (80% of total image area).
    2. Calculating Mean Pixel Value (MeanPV_central) and Standard Deviation (MeanSD_central) of this central ROI.
    3. Performing a sliding window analysis with a 30mm x 30mm ROI moving in 15mm steps within the central ROI.
    4. For each moving ROI, its local mean (PV_i) and local standard deviation (SD_i) are found.
    5. Calculating:
        - GU_PV: Max(abs(PV_i - MeanPV_central) / abs(MeanPV_central))
        - LU_PV: Max(abs(PV_i - PV_8n) / abs(PV_8n)) (PV_8n is mean of 8 neighbors' PVs)
        - GU_SNR: Max(abs(SNR_i - MeanSNR_central) / abs(MeanSNR_central)), where SNR = PV/SD
        - LU_SNR: Max(abs(SNR_i - SNR_8n) / abs(SNR_8n)) (SNR_8n is mean of 8 neighbors' SNRs)

    Args:
        image_array (np.ndarray): The 2D NumPy array representing the X-ray image pixels.
        pixel_spacing_row (float): Pixel spacing in mm/pixel for the row dimension (height).
        pixel_spacing_col (float): Pixel spacing in mm/pixel for the column dimension (width).

    Returns:
        dict: A dictionary containing the calculated metrics:
            "GU_PV", "LU_PV", "GU_SNR", "LU_SNR",
            "MeanPV_central", "MeanSD_central",
            "MeanSNR_central", (calculated as MeanPV_central / MeanSD_central)
            "central_roi_coords" (tuple: y_start, x_start, y_end, x_end),
            "moving_roi_pvs" (np.ndarray): Grid of mean pixel values for each moving ROI.
            "moving_roi_sds" (np.ndarray): Grid of standard deviations for each moving ROI.
            "num_moving_rois", "moving_roi_grid_shape" (tuple: rows, cols).
            Metrics can be np.nan if prerequisites are not met (e.g., image too small).
    """
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise ValueError("image_array must be a 2D NumPy array.")
    if pixel_spacing_row <= 0 or pixel_spacing_col <= 0:
        raise ValueError("Pixel spacing values must be positive.")

    # --- 1. Define central ROI (80% of total area) ---
    H_orig, W_orig = image_array.shape

    scale_factor = np.sqrt(0.8) # To get 80% area, scale dimensions by sqrt(0.8)
    new_H = int(round(H_orig * scale_factor))
    new_W = int(round(W_orig * scale_factor))

    # Default results for cases where processing isn't feasible
    nan_results = {
        "GU_PV": np.nan, "LU_PV": np.nan, "GU_SNR": np.nan, "LU_SNR": np.nan,
        "MeanPV_central": np.nan, "MeanSD_central": np.nan, "MeanSNR_central": np.nan,
        "central_roi_coords": None, "num_moving_rois": 0, "moving_roi_grid_shape": (0,0),
        "central_roi_percent": 0.0,
        "num_invalid_rois": 0,
        "num_valid_rois": 0,
        "moving_roi_pvs": np.array([]).reshape(0,0), 
        "moving_roi_sds": np.array([]).reshape(0,0)
    }

    if new_H < 1 or new_W < 1:
        # Central ROI is too small (effectively zero area)
        return nan_results

    start_row_central = (H_orig - new_H) // 2
    end_row_central = start_row_central + new_H
    start_col_central = (W_orig - new_W) // 2
    end_col_central = start_col_central + new_W

    central_roi_data = image_array[start_row_central:end_row_central, start_col_central:end_col_central]
    central_roi_coords = (start_row_central, start_col_central, end_row_central, end_col_central)

    # Percentage of the image area taken by the central ROI
    central_roi_area = new_H * new_W
    total_image_area = H_orig * W_orig
    central_roi_percent = 100.0 * (central_roi_area / total_image_area) if total_image_area > 0 else 0.0

    if central_roi_data.size == 0:
        nan_results["central_roi_coords"] = central_roi_coords
        return nan_results

    # --- 2. Placeholder central means ---
    # These will be calculated from the grid of moving-ROI averages below
    MeanPV_central = np.nan
    MeanSD_central = np.nan
    MeanSNR_central = np.nan

    # --- 3. Define moving ROI parameters (30mm x 30mm, step 15mm) ---
    roi_size_mm = 30.0
    step_size_mm = 15.0

    roi_h_px = int(round(roi_size_mm / pixel_spacing_row))
    roi_w_px = int(round(roi_size_mm / pixel_spacing_col))
    step_h_px = int(round(step_size_mm / pixel_spacing_row))
    step_w_px = int(round(step_size_mm / pixel_spacing_col))

    # Ensure step sizes are at least 1 pixel
    step_h_px = max(1, step_h_px)
    step_w_px = max(1, step_w_px)

    base_results = {
        "MeanPV_central": MeanPV_central, "MeanSD_central": MeanSD_central,
        "MeanSNR_central": MeanSNR_central,
        "central_roi_coords": central_roi_coords, "num_moving_rois": 0,
        "moving_roi_grid_shape": (0,0),
        "moving_roi_pvs": np.array([]).reshape(0,0), "moving_roi_sds": np.array([]).reshape(0,0),
        "central_roi_percent": central_roi_percent
    }

    if roi_h_px < 1 or roi_w_px < 1:
        # Moving ROI is too small in pixels (e.g., pixel spacing very large)
        return {**nan_results, **base_results, "GU_PV": 0.0, "GU_SNR": 0.0} # No variation if no sub-ROIs

    # --- 4. Sliding window analysis within the central_roi_data ---
    central_roi_h, central_roi_w = central_roi_data.shape

    if central_roi_h < roi_h_px or central_roi_w < roi_w_px:
        # Central ROI cannot fit even one moving ROI
        return {**nan_results, **base_results, "GU_PV": 0.0, "GU_SNR": 0.0}

    y_coords = list(range(0, central_roi_h - roi_h_px + 1, step_h_px))
    x_coords = list(range(0, central_roi_w - roi_w_px + 1, step_w_px))

    num_rois_y = len(y_coords)
    num_rois_x = len(x_coords)

    if num_rois_y == 0 or num_rois_x == 0:
        # No moving ROIs can be formed
        return {**nan_results, **base_results, "GU_PV": 0.0, "GU_SNR": 0.0}

    # Recalculate central_roi_percent using the actual area covered by the moving-ROI grid
    # y_coords/x_coords are in coordinates within the central ROI (0-based). The covered
    # vertical span is from the first y to the last y + roi_h_px. Same for horizontal.
    try:
        used_h_px = (y_coords[-1] + roi_h_px) - y_coords[0]
        used_w_px = (x_coords[-1] + roi_w_px) - x_coords[0]
        used_area = max(0, used_h_px) * max(0, used_w_px)
        central_roi_percent = 100.0 * (used_area / total_image_area) if total_image_area > 0 else 0.0
    except Exception:
        # Fallback to previously computed central_roi_percent (from central ROI geometry)
        pass

    # Store PV_i and SD_i for each moving ROI in grids
    pv_grid = np.full((num_rois_y, num_rois_x), np.nan, dtype=float)
    sd_grid = np.full((num_rois_y, num_rois_x), np.nan, dtype=float)
    snr_grid = np.full((num_rois_y, num_rois_x), np.nan, dtype=float)

    for r_idx, y in enumerate(y_coords):
        for c_idx, x in enumerate(x_coords):
            moving_roi = central_roi_data[y : y + roi_h_px, x : x + roi_w_px]
            if moving_roi.size == 0: continue # Should not happen due to prior checks
            
            local_pv = np.mean(moving_roi)
            local_sd = np.std(moving_roi)

            pv_grid[r_idx, c_idx] = local_pv
            sd_grid[r_idx, c_idx] = local_sd

            # Compute SNR robustly: if local_sd is zero or not finite, set SNR to NaN
            if not np.isfinite(local_sd) or np.isclose(local_sd, 0.0):
                snr_grid[r_idx, c_idx] = np.nan
            else:
                snr_grid[r_idx, c_idx] = local_pv / local_sd

    # Count invalid/valid ROIs (invalid if PV or SNR is not finite)
    total_rois = num_rois_y * num_rois_x
    num_invalid_rois = int(np.count_nonzero(~np.isfinite(pv_grid) | ~np.isfinite(snr_grid)))
    num_valid_rois = int(total_rois - num_invalid_rois)

    # --- 5. Calculate uniformity metrics ---
    gu_pv_terms = []
    gu_snr_terms = []
    lu_pv_terms = []
    lu_snr_terms = []

    # Compute central means from the moving ROI grids
    # Use nanmean to ignore invalid/missing ROIs
    if np.isfinite(pv_grid).any():
        MeanPV_central = np.nanmean(pv_grid)
    else:
        MeanPV_central = np.nan

    if np.isfinite(sd_grid).any():
        MeanSD_central = np.nanmean(sd_grid)
    else:
        MeanSD_central = np.nan

    if np.isfinite(snr_grid).any():
        MeanSNR_central = np.nanmean(snr_grid)
    else:
        MeanSNR_central = np.nan

    for r_idx in range(num_rois_y):
        for c_idx in range(num_rois_x):
            pv_i = pv_grid[r_idx, c_idx]
            snr_i = snr_grid[r_idx, c_idx]  # Use calculated SNR_i

            # Skip ROIs where PV or SNR is undefined
            if not np.isfinite(pv_i) or not np.isfinite(snr_i):
                continue

            # Global Uniformity terms (append only if finite)
            gu_pv_term = _calculate_uniformity_term(pv_i, MeanPV_central)
            if np.isfinite(gu_pv_term):
                gu_pv_terms.append(gu_pv_term)

            gu_snr_term = _calculate_uniformity_term(snr_i, MeanSNR_central)
            if np.isfinite(gu_snr_term):
                gu_snr_terms.append(gu_snr_term)

            # Local Uniformity terms (only for ROIs with 8 valid neighbors)
            if 0 < r_idx < num_rois_y - 1 and 0 < c_idx < num_rois_x - 1:
                neighbor_pvs = []
                neighbor_snrs = []
                valid_neighbors = True
                for dr_neighbor in [-1, 0, 1]:
                    for dc_neighbor in [-1, 0, 1]:
                        if dr_neighbor == 0 and dc_neighbor == 0:
                            continue  # Skip the central ROI itself

                        nr, nc = r_idx + dr_neighbor, c_idx + dc_neighbor

                        if not np.isfinite(pv_grid[nr, nc]) or not np.isfinite(snr_grid[nr, nc]):
                            valid_neighbors = False  # A neighbor has invalid data
                            break
                        neighbor_pvs.append(pv_grid[nr, nc])
                        neighbor_snrs.append(snr_grid[nr, nc])
                    if not valid_neighbors:
                        break

                if valid_neighbors and len(neighbor_pvs) == 8:  # Ensure all 8 neighbors were valid and collected
                    pv_8n = np.mean(neighbor_pvs)
                    lu_pv_term = _calculate_uniformity_term(pv_i, pv_8n)
                    if np.isfinite(lu_pv_term):
                        lu_pv_terms.append(lu_pv_term)

                    snr_8n = np.mean(neighbor_snrs)  # Mean of neighbor SNRs
                    lu_snr_term = _calculate_uniformity_term(snr_i, snr_8n)
                    if np.isfinite(lu_snr_term):
                        lu_snr_terms.append(lu_snr_term)
    
    # Final metrics: Max of the calculated terms.
    # Using np.nanmax to ignore NaNs if any slip through, though robust _calculate_uniformity_term aims to prevent them.
    GU_PV = np.nanmax(gu_pv_terms) * 100 if gu_pv_terms else np.nan
    LU_PV = np.nanmax(lu_pv_terms) * 100 if lu_pv_terms else np.nan
    GU_SNR = np.nanmax(gu_snr_terms) * 100 if gu_snr_terms else np.nan
    LU_SNR = np.nanmax(lu_snr_terms) * 100 if lu_snr_terms else np.nan
    
    return {
        "GU_PV": GU_PV,
        "LU_PV": LU_PV,
        "GU_SNR": GU_SNR,
        "LU_SNR": LU_SNR,
        "MeanPV_central": abs(MeanPV_central),
        "MeanSD_central": MeanSD_central,
        "MeanSNR_central": abs(MeanSNR_central),
        "central_roi_coords": central_roi_coords,
        "num_moving_rois": num_rois_y * num_rois_x,
        "central_roi_percent": central_roi_percent,
        "num_invalid_rois": num_invalid_rois,
        "num_valid_rois": num_valid_rois,
        "moving_roi_grid_shape": (num_rois_y, num_rois_x)
    }

def display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    """
    Runs the Streamlit application for X-ray Uniformity Analysis.
    """

    # This function is now called by the main app (exe_analyzer.py)
    # It assumes image_array, pixel_spacing_row, and pixel_spacing_col are already available.

    st.subheader("Uniformity Analysis")
    st.write("""
    Calculates uniformity metrics (Global/Local Pixel Value Uniformity, Global/Local SNR Uniformity)
    within a central 80% area ROI using a sliding 30mm x 30mm window.
    """)

    if pixel_spacing_row is not None and pixel_spacing_col is not None:
        if st.button("Run Uniformity Analysis"):
            with st.spinner("Calculating uniformity metrics..."):
                try:
                    results = calculate_xray_uniformity_metrics(image_array, pixel_spacing_row, pixel_spacing_col)
                    st.success("Uniformity Analysis Complete!") # Display results as a JSON object for clarity
                    
                    st.subheader("Calculated Uniformity Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Size of central ROI", f"{results['central_roi_percent']:.2f}% of image area")
                        st.metric("Moving ROI grid (rows x cols)", f"{results['moving_roi_grid_shape'][0]} x {results['moving_roi_grid_shape'][1]}")
                        st.metric("Global Uniformity (PV)", f"{results['GU_PV']:.2f}%")
                        st.metric("Local Uniformity (PV)", f"{results['LU_PV']:.2f}%")
                    with col2:
                        st.metric("Number of moving ROIs", f"{results['num_moving_rois']:.0f}")
                        st.metric("Invalid ROIs (PV or SNR NaN)", f"{results['num_invalid_rois']:.0f}")
                        st.metric("Global Uniformity (SNR)", f"{results['GU_SNR']:.2f}%")
                        st.metric("Local Uniformity (SNR)", f"{results['LU_SNR']:.2f}%")

                except ValueError as ve:
                    st.error(f"Uniformity analysis failed due to invalid input: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during uniformity analysis: {e}")
    else:
        st.warning("Pixel spacing information is required for Uniformity Analysis. Ensure the DICOM file has the PixelSpacing tag (0028,0030).")
