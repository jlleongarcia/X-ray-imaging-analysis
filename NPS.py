import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, average_power
import pandas as pd

def calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col, **kwargs):
    """
    Calculates NPS metrics from an image array.

    Args:
        image_array (np.ndarray): The input image data.
        pixel_spacing_row (float): Pixel spacing for rows.
        pixel_spacing_col (float): Pixel spacing for columns.
        **kwargs: Additional parameters specific to NPS.

    Returns:
        dict: A dictionary containing NPS results.
    """
    if image_array is None or image_array.ndim != 2 or image_array.size == 0:
        st.error("Valid 2D image_array is required for NPS calculation.")
        return {"NPS_Status": "Error: Invalid image array"}

    if pixel_spacing_row is None or pixel_spacing_col is None or pixel_spacing_row <= 0 or pixel_spacing_col <= 0:
        st.warning("Pixel spacing is not valid or available. NPS spatial frequencies will be in cycles/pixel.")
        pixel_spacing_row_safe = 1.0
        pixel_spacing_col_safe = 1.0
        x_axis_unit_nps = "cycles/pixel"
    else:
        pixel_spacing_row_safe = pixel_spacing_row
        pixel_spacing_col_safe = pixel_spacing_col
        x_axis_unit_nps = "lp/mm"

    pixel_spacing_avg = (pixel_spacing_row_safe + pixel_spacing_col_safe) / 2

    # The noise_power_spectrum_2d function expects an iterable of ROIs.
    rois_list = [image_array]

    try:
        nps_2d_result = noise_power_spectrum_2d(pixel_size=pixel_spacing_avg, rois=rois_list)
        
        # NNPS definition: NNPS(fx, fy) = NPS(fx, fy) / NPS(0)
        # NPS(0) is the value at the center of the 2D NPS array.
        center_y, center_x = nps_2d_result.shape[0] // 2, nps_2d_result.shape[1] // 2
        nps_at_zero_freq = nps_2d_result[center_y, center_x]
        
        if nps_at_zero_freq == 0:
            st.warning("NPS at zero frequency is zero. NNPS cannot be normalized.")
            nnps_2d = np.full_like(nps_2d_result, np.nan)
        else:
            nnps_2d = nps_2d_result / nps_at_zero_freq

        # Calculate 1D NPS and NNPS from their 2D counterparts
        nps_1d_result = noise_power_spectrum_1d(spectrum_2d=nps_2d_result)
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d)
        avg_power_result = average_power(nps1d=nps_1d_result)

        # Calculate NNPS at specific frequencies (0.5 mm^-1, 2.0 mm^-1)
        # Spatial frequency axes
        fy_axis_shifted = np.fft.fftshift(np.fft.fftfreq(image_array.shape[0], d=pixel_spacing_row_safe))
        fx_axis_shifted = np.fft.fftshift(np.fft.fftfreq(image_array.shape[1], d=pixel_spacing_col_safe))

        target_fx = 0.5  # mm^-1
        target_fy = 2.0  # mm^-1

        # Find the indices of the closest frequencies
        idx_fx = np.argmin(np.abs(fx_axis_shifted - target_fx))
        idx_fy = np.argmin(np.abs(fy_axis_shifted - target_fy))

        # Get the actual frequency values at these indices
        actual_fx_value = fx_axis_shifted[idx_fx]
        actual_fy_value = fy_axis_shifted[idx_fy]

        # Get the NNPS value at these (closest) frequencies
        nnps_at_target_freq = nnps_2d[idx_fy, idx_fx]

        # Calculate frequencies for 1D NNPS plot
        # The 1D NPS is a radial average. The frequency axis goes from 0 to Nyquist.
        # Nyquist frequency is 0.5 / pixel_spacing.
        # Use the maximum Nyquist frequency from both dimensions for the x-axis range.
        nyquist_freq_row = 0.5 / pixel_spacing_row_safe
        nyquist_freq_col = 0.5 / pixel_spacing_col_safe
        max_nyquist_freq = max(nyquist_freq_row, nyquist_freq_col)

        # Generate frequencies for the 1D NNPS plot
        # The number of points in the frequency axis should match the length of nnps_1d_result
        frequencies_nps = np.linspace(0, max_nyquist_freq, len(nnps_1d_result))
        nnps_data_for_chart = np.array([frequencies_nps, nnps_1d_result]).T

        return {
            "NNPS_at_target_fx_fy": {
                "target_fx": float(target_fx),
                "target_fy": float(target_fy),
                "actual_fx": float(actual_fx_value),
                "actual_fy": float(actual_fy_value),
                "value": float(nnps_at_target_freq) if not np.isnan(nnps_at_target_freq) else np.nan
            },
            "NNPS_1D_chart_data": nnps_data_for_chart,
            "x_axis_unit_nps": x_axis_unit_nps,
            "Average_Power": float(avg_power_result),
        }
    except Exception as e:
        st.error(f"Error during NPS calculation: {e}")
        return {"NPS_Status": f"Error: {e}"}

def display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    st.subheader("Noise Power Spectrum (NPS) Analysis")

    # Initialize session state for current NPS results
    # if 'current_nps_results' not in st.session_state:
    #     st.session_state['current_nps_results'] = None

    if st.button("Run NPS Analysis"):
        with st.spinner("Calculating NPS..."):
            nps_results_dict = calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col)
            st.success("NPS Analysis Complete!")
            
            if "NPS_Status" in nps_results_dict and "Error" in nps_results_dict["NPS_Status"]:
                st.error(f"NPS Calculation Failed: {nps_results_dict['NPS_Status']}")
            elif nps_results_dict and "NNPS_1D_chart_data" in nps_results_dict:
                st.success("NPS Analysis Complete!")
                st.subheader("1D Normalized Noise Power Spectrum")
                
                nnps_chart_data = nps_results_dict["NNPS_1D_chart_data"]
                x_axis_unit_nps = nps_results_dict["x_axis_unit_nps"]
                df_nnps = pd.DataFrame(nnps_chart_data, columns=[x_axis_unit_nps, 'NNPS'])
                st.line_chart(df_nnps.set_index(x_axis_unit_nps))
                
                # Display the NNPS at target frequency if available
                if "NNPS_at_target_fx_fy" in nps_results_dict:
                    st.subheader("NNPS at Target Frequencies")
                    target_info = nps_results_dict["NNPS_at_target_fx_fy"]
                    nnps_value_display = f"{target_info['value']:.4e}" if not np.isnan(target_info['value']) else "N/A"
                    
                    st.write(f"Target (fx, fy): ({target_info['target_fx']:.2f} mm⁻¹, {target_info['target_fy']:.2f} mm⁻¹)")
                    st.write(f"Actual (fx, fy): ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹)")
                    st.write(f"NNPS ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹): {nnps_value_display}")

                st.session_state['nnps_data'] = nnps_chart_data
            else:
                st.error("NPS calculation did not return expected results.")
