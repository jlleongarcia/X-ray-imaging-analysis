import streamlit as st
import numpy as np
import pandas as pd

def get_freq_at_mtf_level(target_mtf_level: float, mtf_values_np: np.ndarray, frequencies_np: np.ndarray) -> float:
    """
    Finds the spatial frequency at which the MTF curve reaches a certain target_mtf_level.
    Assumes MTF values are generally decreasing with increasing frequency.

    Args:
        target_mtf_level (float): The target MTF value (e.g., 0.5 for MTF50).
        mtf_values_np (np.ndarray): 1D array of MTF values.
        frequencies_np (np.ndarray): 1D array of corresponding spatial frequencies.

    Returns:
        float: The interpolated spatial frequency, or np.nan if not found/applicable.
    """
    if not (isinstance(mtf_values_np, np.ndarray) and isinstance(frequencies_np, np.ndarray) and
            mtf_values_np.ndim == 1 and frequencies_np.ndim == 1 and
            mtf_values_np.size == frequencies_np.size and mtf_values_np.size >= 2):
        return np.nan

    valid_indices = ~np.isnan(mtf_values_np) & ~np.isnan(frequencies_np)
    mtf_clean = mtf_values_np[valid_indices]
    freq_clean = frequencies_np[valid_indices]

    if mtf_clean.size < 2:
        return np.nan
    if target_mtf_level > mtf_clean[0] + 1e-6 or target_mtf_level < mtf_clean[-1] - 1e-6:
        return np.nan
    return float(np.interp(target_mtf_level, mtf_clean[::-1], freq_clean[::-1]))

def calculate_mtf_metrics(image_array, pixel_spacing_col):
    """
    Calculates MTF from a given 1D profile.
    For this initial version, it uses the central row of the image_array as a simplified LSF.

    Args:
        image_array (np.ndarray): The input 2D image data.
        pixel_spacing_col (float): Pixel spacing for columns (e.g., mm/pixel), used for dpmm.

    Returns:
        dict: A dictionary containing MTF results.
    """
    if image_array is None or image_array.ndim != 2 or image_array.shape[0] < 1 or image_array.shape[1] < 2:
        # Need at least 2 pixels for a profile
        st.error("Valid 2D image_array with at least 2 columns is required for MTF calculation.")
        return {"MTF_Status": "Error: Invalid image array"}

    if pixel_spacing_col is not None and pixel_spacing_col <= 0:
        st.warning("Pixel spacing (col) is not valid (<=0). MTF spatial frequencies will be in cycles/pixel.")
        dpmm = None
        x_axis_unit = "cycles/pixel"
    elif pixel_spacing_col is None:
        st.warning("Pixel spacing (col) is not available. MTF spatial frequencies will be in cycles/pixel.")
        dpmm = None
        x_axis_unit = "cycles/pixel"
    else:
        dpmm = 1.0 / pixel_spacing_col
        x_axis_unit = "lp/mm"

    # Extract the central row as a simplified LSF.
    profile_values = image_array[image_array.shape[0] // 2, :].astype(float)
    
    if profile_values.size < 2: # MTF calculation needs at least 2 points
        st.error("Extracted profile for MTF has fewer than 2 points.")
        return {"MTF_Status": "Error: Profile too short"}

    try:
        # --- MTF Calculation from Scratch ---
        N = profile_values.size

        # 1. Compute FFT of the LSF
        lsf_fft = np.fft.fft(profile_values)

        # 2. Compute MTF (magnitude of FFT, positive frequencies only)
        mtf_raw = np.abs(lsf_fft[:N//2])

        # 3. Normalize MTF
        if mtf_raw[0] < 1e-9: # Avoid division by zero or near-zero
            st.warning("MTF at zero frequency is near zero. Cannot normalize.")
            mtf_normalized = np.full_like(mtf_raw, np.nan) # Or mtf_raw, or handle error
        else:
            mtf_normalized = mtf_raw / mtf_raw[0]

        # 4. Generate Frequency Axis
        # Frequencies from fftfreq are in cycles/sample (cycles/pixel)
        frequencies_cycles_per_pixel = np.fft.fftfreq(N)[:N//2]
        if dpmm is not None: # Convert to lp/mm if dpmm is available
            frequencies_for_mtf = frequencies_cycles_per_pixel * dpmm
        else:
            frequencies_for_mtf = frequencies_cycles_per_pixel

        frequencies_list = frequencies_for_mtf.tolist()
        mtf_list = mtf_normalized.tolist()

        # Prepare data for st.line_chart
        mtf_data_for_chart = np.array([frequencies_list, mtf_list]).T

        results = {
            "frequencies": frequencies_list,
            "mtf_values": mtf_list,
            "mtf_chart_data": mtf_data_for_chart,
            "x_axis_unit": x_axis_unit,
            "source_profile": "Central Row (Simplified LSF)"
        }

        # Calculate MTF at specific percentages (MTF50%, MTF10%)
        results["MTF50"] = get_freq_at_mtf_level(0.5, mtf_normalized, frequencies_for_mtf)
        results["MTF10"] = get_freq_at_mtf_level(0.1, mtf_normalized, frequencies_for_mtf)
        # Convert NaN to "N/A" for display
        results["MTF50"] = "N/A" if np.isnan(results["MTF50"]) else results["MTF50"]
        results["MTF10"] = "N/A" if np.isnan(results["MTF10"]) else results["MTF10"]

        return results

    except Exception as e:
        st.error(f"Error during MTF calculation: {e}")
        return {"MTF_Status": f"Error: {e}"}

def display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    st.subheader("Modulation Transfer Function (MTF) Analysis")
    st.markdown("""
    **Note:** This initial MTF calculation uses the **central row** of the image as a simplified Line Spread Function (LSF).
    For accurate MTF, a well-defined edge or slit from a phantom image is typically required.
    """)

    if st.button("Run MTF Analysis (using central row)"):
        if image_array is None:
            st.error("Please upload an image first.")
            return

        with st.spinner("Calculating MTF..."):
            mtf_results_dict = calculate_mtf_metrics(image_array, pixel_spacing_col)
            
            if "MTF_Status" in mtf_results_dict and "Error" in mtf_results_dict["MTF_Status"]:
                st.error(f"MTF Calculation Failed: {mtf_results_dict['MTF_Status']}")
            elif mtf_results_dict and "mtf_chart_data" in mtf_results_dict:
                st.success("MTF Analysis Complete!")
                st.write(f"MTF calculated from: **{mtf_results_dict.get('source_profile', 'N/A')}**")
                
                st.subheader(f"MTF Curve ({mtf_results_dict['x_axis_unit']})")
                df_mtf = pd.DataFrame(mtf_results_dict["mtf_chart_data"], columns=[mtf_results_dict['x_axis_unit'], 'MTF'])
                st.line_chart(df_mtf.set_index(mtf_results_dict['x_axis_unit']))

                st.write(f"**MTF50% ({mtf_results_dict.get('x_axis_unit','')}):** {mtf_results_dict.get('MTF50', 'N/A')}")
                st.write(f"**MTF10% ({mtf_results_dict.get('x_axis_unit','')}):** {mtf_results_dict.get('MTF10', 'N/A')}")
                
                st.json(mtf_results_dict) # Display all results for inspection
            else:
                st.error("MTF calculation did not return expected results.")
