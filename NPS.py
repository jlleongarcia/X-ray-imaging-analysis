import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, average_power
import pandas as pd

def calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col):
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

    try:
        # Edge effects arising from taking the Fourier transform of an image of finite dimensions need to be considered,
        # as these will lead to a large peak at ω = 0. To avoid this, a difference image, calculated from two images recorded 
        # under the identical conditions, is taken to calculate NPS. 
        # Such a difference image will have a mean of zero, negating edge effects, and thus the resulting NPS only needs to be halved to compensate.

        # Define the list of allowed power-of-two values for the large central ROI and for the small ROIs within
        allowed_big = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        allowed_small = [8, 16, 32, 64, 128, 256, 512]
        selected_big = st.select_slider(
            label="Slide to select the pixel size for the large central ROI:",
            options=allowed_big,
            value=1024
        )
        selected_small = st.select_slider(
            label="Slide to select the pixel size for the large central ROI:",
            options=allowed_small,
            value=128
        )
        nps_2d_raw, mean_pv = noise_power_spectrum_2d(
            image_array,
            pixel_size=pixel_spacing_avg,
            big_roi_size=selected_big,
            small_roi_size=selected_small
        )

        nps_2d_result = nps_2d_raw / 2
        
        # NNPS definition: NNPS(fx, fy) = NPS(fx, fy) / (Mean Pixel Value of Largest ROI)^2
        nnps_2d = nps_2d_result / (mean_pv ** 2)

        # Separate both axis of NNPS to plot them and compare to radial average
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d)
        nnps_x = np.fft.fftshift(nnps_2d.mean(axis=0))
        nnps_y = np.fft.fftshift(nnps_2d.mean(axis=1))

        # Calculate NNPS at specific frequencies (0.5 mm^-1, 2.0 mm^-1)
        # Spatial frequency axes
        f1_axis_shifted = np.fft.fftshift(np.fft.fftfreq(selected_small, d=pixel_spacing_row_safe))
        f2_axis_shifted = np.fft.fftshift(np.fft.fftfreq(selected_small, d=pixel_spacing_col_safe))

        target_f1 = 0.5  # mm^-1
        target_f2 = 2.0  # mm^-1

        # Find the indices of the closest frequencies
        idx_f1 = np.argmin(np.abs(f1_axis_shifted - target_f1))
        idx_f2 = np.argmin(np.abs(f2_axis_shifted - target_f2))

        # Get the actual frequency values at these indices
        actual_f1_value = f1_axis_shifted[idx_f1]
        actual_f2_value = f2_axis_shifted[idx_f2]

        # Get the NNPS values at these (closest) frequencies
        nnps_x_at_f1 = nnps_x[idx_f1]
        nnps_x_at_f2 = nnps_x[idx_f2]
        nnps_y_at_f1 = nnps_y[idx_f1]
        nnps_y_at_f2 = nnps_y[idx_f2]

        # Calculate frequencies for 1D NNPS plot
        # The 1D NPS is a radial average. The frequency axis goes from 0 to Nyquist.
        # Nyquist frequency is 0.5 / pixel_spacing.
        # Use the maximum Nyquist frequency from both dimensions for the x-axis range.
        nyquist_freq_row = 0.5 / pixel_spacing_row_safe
        nyquist_freq_col = 0.5 / pixel_spacing_col_safe
        max_nyquist_freq = max(nyquist_freq_row, nyquist_freq_col)

        # Generate frequencies for the 1D NNPS plot
        # The number of points in the frequency axis should match the length of nnps_1d_result
        frequencies_nps = np.linspace(0, max_nyquist_freq, len(nnps_x))
        nnps_data_for_chart = np.array([frequencies_nps, nnps_x, nnps_y]).T
        # nnps_x_data_for_chart = np.array([frequencies_nps, nnps_x]).T
        # nnps_y_data_for_chart = np.array([frequencies_nps, nnps_y]).T

        return {
            "NNPS_at_target_fx_fy": {
                "target_f1": float(target_f1),
                "target_f2": float(target_f2),
                "actual_f1": float(actual_f1_value),
                "actual_f2": float(actual_f2_value),
                "value_1x": float(nnps_x_at_f1),
                "value_1y": float(nnps_y_at_f1),
                "value_2x": float(nnps_x_at_f2),
                "value_2y": float(nnps_y_at_f2),
            },
            "NNPS_1D_chart_data": nnps_data_for_chart,
            # "NNPS_x_chart_data": nnps_x_data_for_chart,
            # "NNPS_y_chart_data": nnps_y_data_for_chart,
            "x_axis_unit_nps": x_axis_unit_nps,
        }
    except Exception as e:
        st.error(f"Error during NPS calculation: {e}")
        return {"NPS_Status": f"Error: {e}"}

def display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    st.subheader("Noise Power Spectrum (NPS) Analysis")

    if st.button("Run NPS Analysis"):
        with st.spinner("Calculating NPS..."):
            nps_results_dict = calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col)
            
            if "NPS_Status" in nps_results_dict and "Error" in nps_results_dict["NPS_Status"]:
                st.error(f"NPS Calculation Failed: {nps_results_dict['NPS_Status']}")
            elif nps_results_dict and "NNPS_1D_chart_data" in nps_results_dict:
                st.success("NPS Analysis Complete!")
                st.subheader("Normalized Noise Power Spectrum")
                
                nnps_chart_data = nps_results_dict["NNPS_1D_chart_data"]
                # nnps_x_chart_data = nps_results_dict["NNPS_x_chart_data"]
                # nnps_y_chart_data = nps_results_dict["NNPS_y_chart_data"]
                x_axis_unit_nps = nps_results_dict["x_axis_unit_nps"]
                df_nnps = pd.DataFrame(nnps_chart_data, columns=[x_axis_unit_nps, 'NNPS_x', 'NNPS_y'])
                st.line_chart(df_nnps.set_index(x_axis_unit_nps), x_label=x_axis_unit_nps, y_label='NNPS')
                
                # Display the NNPS at target frequency if available
                if "NNPS_at_target_fx_fy" in nps_results_dict:
                    st.subheader("NNPS at Target Frequencies")
                    target_info = nps_results_dict["NNPS_at_target_fx_fy"]
                    nnps_value_display = f"{target_info['value']:.4e}" if not np.isnan(target_info['value']) else "N/A"
                    
                    st.write(f"Target (fx, fy): ({target_info['target_fx']:.2f} mm⁻¹, {target_info['target_fy']:.2f} mm⁻¹)")
                    st.write(f"Actual (fx, fy): ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹)")
                    st.write(f"NNPS ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹): {nnps_value_display}")

                st.session_state['nnps_data'] = nnps1d_chart_data
            else:
                st.error("NPS calculation did not return expected results.")
