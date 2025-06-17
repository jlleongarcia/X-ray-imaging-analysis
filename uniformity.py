import numpy as np
import streamlit as st
import pydicom
from PIL import Image # Used for displaying images in Streamlit
import os
import tempfile

def _calculate_uniformity_term(val_i, val_mean, epsilon=1e-9):
    """
    Calculates the term abs(val_i - val_mean) / val_mean for uniformity.
    Handles cases where val_mean is zero or close to zero.
    Assumes val_mean is non-negative (typical for pixel values and standard deviations).

    Args:
        val_i (float): The individual ROI value (e.g., PV_i or SD_i).
        val_mean (float): The reference mean value (e.g., MeanPV, PV_8n, MeanSD, SD_8n).
        epsilon (float): A small number to handle near-zero denominators.

    Returns:
        float: The calculated uniformity term. Can be 0.0, a positive float, or np.inf.
    """
    abs_diff = np.abs(val_i - val_mean)

    if val_mean < epsilon:  # If val_mean is effectively zero
        if abs_diff < epsilon:  # And val_i is also effectively equal to val_mean (i.e., zero)
            return 0.0
        else:  # val_i is different from zero, val_mean is zero (infinite non-uniformity)
            return np.inf
    else:  # val_mean is significantly non-zero
        return abs_diff / val_mean

def calculate_xray_uniformity_metrics(image_array, pixel_spacing_row, pixel_spacing_col):
    """
    Calculates uniformity metrics for an X-ray image.

    The process involves:
    1. Defining a central ROI (80% of total image area).
    2. Calculating Mean Pixel Value (MeanPV_central) and Standard Deviation (MeanSD_central) of this central ROI.
    3. Performing a sliding window analysis with a 30mm x 30mm ROI moving in 15mm steps within the central ROI.
    4. For each moving ROI, its local mean (PV_i) and local standard deviation (SD_i) are found.
    5. Calculating:
        - GU_PV: Max(abs(PV_i - MeanPV_central) / MeanPV_central)
        - LU_PV: Max(abs(PV_i - PV_8n) / PV_8n) (PV_8n is mean of 8 neighbors)
        - GU_SNR: Max(abs(SD_i - MeanSD_central) / MeanSD_central)
        - LU_SNR: Max(abs(SD_i - SD_8n) / SD_8n) (SD_8n is mean of 8 neighbors' SDs)

    Args:
        image_array (np.ndarray): The 2D NumPy array representing the X-ray image pixels.
        pixel_spacing_row (float): Pixel spacing in mm/pixel for the row dimension (height).
        pixel_spacing_col (float): Pixel spacing in mm/pixel for the column dimension (width).

    Returns:
        dict: A dictionary containing the calculated metrics:
            "GU_PV", "LU_PV", "GU_SNR", "LU_SNR",
            "MeanPV_central", "MeanSD_central",
            "central_roi_coords" (tuple: y_start, x_start, y_end, x_end),
            "num_moving_rois", "moving_roi_grid_shape" (tuple: rows, cols).
            Metrics can be np.nan if prerequisites are not met (e.g., image too small).
    """
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise ValueError("image_array must be a 2D NumPy array.")
    if pixel_spacing_row <= 0 or pixel_spacing_col <= 0:
        raise ValueError("Pixel spacing values must be positive.")

    epsilon = 1e-9  # Small value for safe division and zero-checks

    # --- 1. Define central ROI (80% of total area) ---
    H_orig, W_orig = image_array.shape

    scale_factor = np.sqrt(0.8) # To get 80% area, scale dimensions by sqrt(0.8)
    new_H = int(round(H_orig * scale_factor))
    new_W = int(round(W_orig * scale_factor))

    # Default results for cases where processing isn't feasible
    nan_results = {
        "GU_PV": np.nan, "LU_PV": np.nan, "GU_SNR": np.nan, "LU_SNR": np.nan,
        "MeanPV_central": np.nan, "MeanSD_central": np.nan,
        "central_roi_coords": None, "num_moving_rois": 0, "moving_roi_grid_shape": (0,0)
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

    if central_roi_data.size == 0:
        nan_results["central_roi_coords"] = central_roi_coords
        return nan_results

    # --- 2. Calculate MeanPV and MeanSD for the central ROI ---
    MeanPV_central = np.mean(central_roi_data)
    MeanSD_central = np.std(central_roi_data)

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
        "central_roi_coords": central_roi_coords, "num_moving_rois": 0,
        "moving_roi_grid_shape": (0,0)
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

    # Store PV_i and SD_i for each moving ROI in grids
    pv_grid = np.full((num_rois_y, num_rois_x), np.nan)
    sd_grid = np.full((num_rois_y, num_rois_x), np.nan)

    for r_idx, y in enumerate(y_coords):
        for c_idx, x in enumerate(x_coords):
            moving_roi = central_roi_data[y : y + roi_h_px, x : x + roi_w_px]
            if moving_roi.size == 0: continue # Should not happen due to prior checks
            
            pv_grid[r_idx, c_idx] = np.mean(moving_roi)
            sd_grid[r_idx, c_idx] = np.std(moving_roi)

    # --- 5. Calculate uniformity metrics ---
    gu_pv_terms = []
    gu_snr_terms = []
    lu_pv_terms = []
    lu_snr_terms = []

    for r_idx in range(num_rois_y):
        for c_idx in range(num_rois_x):
            pv_i = pv_grid[r_idx, c_idx]
            sd_i = sd_grid[r_idx, c_idx]

            if np.isnan(pv_i) or np.isnan(sd_i):
                continue # Should not occur if grids populated correctly

            # Global Uniformity terms
            gu_pv_terms.append(_calculate_uniformity_term(pv_i, MeanPV_central, epsilon))
            gu_snr_terms.append(_calculate_uniformity_term(sd_i, MeanSD_central, epsilon))

            # Local Uniformity terms (only for ROIs with 8 valid neighbors)
            if 0 < r_idx < num_rois_y - 1 and 0 < c_idx < num_rois_x - 1:
                neighbor_pvs = []
                neighbor_sds = []
                valid_neighbors = True
                for dr_neighbor in [-1, 0, 1]:
                    for dc_neighbor in [-1, 0, 1]:
                        if dr_neighbor == 0 and dc_neighbor == 0:
                            continue # Skip the central ROI itself
                        
                        nr, nc = r_idx + dr_neighbor, c_idx + dc_neighbor
                        
                        if np.isnan(pv_grid[nr, nc]) or np.isnan(sd_grid[nr, nc]):
                            valid_neighbors = False # A neighbor has NaN data
                            break
                        neighbor_pvs.append(pv_grid[nr, nc])
                        neighbor_sds.append(sd_grid[nr, nc])
                    if not valid_neighbors:
                        break
                
                if valid_neighbors and len(neighbor_pvs) == 8: # Ensure all 8 neighbors were valid and collected
                    pv_8n = np.mean(neighbor_pvs)
                    lu_pv_terms.append(_calculate_uniformity_term(pv_i, pv_8n, epsilon))

                    sd_8n = np.mean(neighbor_sds) # Mean of neighbor SDs
                    lu_snr_terms.append(_calculate_uniformity_term(sd_i, sd_8n, epsilon))
    
    # Final metrics: Max of the calculated terms.
    # If a list of terms is empty (e.g., no ROIs for LU), result is np.nan.
    # If all terms are 0, result is 0. If np.inf is present, it becomes the max.
    GU_PV = np.max(gu_pv_terms) if gu_pv_terms else np.nan
    LU_PV = np.max(lu_pv_terms) if lu_pv_terms else np.nan
    GU_SNR = np.max(gu_snr_terms) if gu_snr_terms else np.nan
    LU_SNR = np.max(lu_snr_terms) if lu_snr_terms else np.nan
    
    return {
        "GU_PV": GU_PV,
        "LU_PV": LU_PV,
        "GU_SNR": GU_SNR,
        "LU_SNR": LU_SNR,
        "MeanPV_central": MeanPV_central,
        "MeanSD_central": MeanSD_central,
        "central_roi_coords": central_roi_coords,
        "num_moving_rois": num_rois_y * num_rois_x,
        "moving_roi_grid_shape": (num_rois_y, num_rois_x)
    }

def run_streamlit_app():
    """
    Runs the Streamlit application for X-ray Uniformity Analysis.
    """
    st.set_page_config(layout="wide") # Use wide layout for better image display

    st.title("X-ray Image Uniformity Analysis")

    st.write("""
    Upload a DICOM X-ray image to calculate uniformity metrics
    (Global/Local Pixel Value Uniformity, Global/Local SNR Uniformity)
    within a central 80% area ROI using a sliding 30mm x 30mm window.
    """)

    uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm", "dicom"])

    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.info(f"Processing file: {uploaded_file.name}")
            
            dicom_dataset = pydicom.dcmread(tmp_file_path)
            
            image_array = dicom_dataset.pixel_array if 'PixelData' in dicom_dataset else None
            
            pixel_spacing_row, pixel_spacing_col = None, None
            if 'PixelSpacing' in dicom_dataset:
                pixel_spacing = dicom_dataset.PixelSpacing
                if len(pixel_spacing) == 2:
                    pixel_spacing_row = float(pixel_spacing[0])
                    pixel_spacing_col = float(pixel_spacing[1])
                    st.write(f"Detected Pixel Spacing: {pixel_spacing_row:.2f} mm (row) x {pixel_spacing_col:.2f} mm (col)")
                else:
                    st.warning(f"Pixel Spacing tag (0028,0030) has unexpected format: {pixel_spacing}.")
            else:
                st.warning("Pixel Spacing tag (0028,0030) not found. Cannot perform mm-based calculations.")

        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            image_array = None
            pixel_spacing_row, pixel_spacing_col = None, None
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

        if image_array is not None:
            st.subheader("Original Image")
            display_array = image_array.astype(np.float32)
            if np.max(display_array) > np.min(display_array):
                display_array = (display_array - np.min(display_array)) / (np.max(display_array) - np.min(display_array)) * 255.0
            else:
                display_array = np.zeros_like(display_array)
            display_array = display_array.astype(np.uint8)

            if len(display_array.shape) == 2:
                img_pil = Image.fromarray(display_array, mode='L')
                st.image(img_pil, caption="Original Image (Normalized for Display)", use_column_width=True)
            else:
                st.warning(f"Image has unexpected shape {image_array.shape}. Cannot display directly.")

            st.subheader("Uniformity Analysis")
            if pixel_spacing_row is not None and pixel_spacing_col is not None:
                if st.button("Run Uniformity Analysis"):
                    with st.spinner("Calculating uniformity metrics..."):
                        try:
                            results = calculate_xray_uniformity_metrics(image_array, pixel_spacing_row, pixel_spacing_col)
                            st.success("Analysis Complete!")
                            st.json(results) # Display results as a JSON object for clarity
                        except ValueError as ve:
                            st.error(f"Analysis failed due to invalid input: {ve}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during analysis: {e}")
            else:
                st.warning("Pixel spacing is required. Upload a DICOM with Pixel Spacing tag (0028,0030).")
        else:
            st.error("Could not load image data.")

if __name__ == '__main__':
    run_streamlit_app()
