import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, average_power
import pandas as pd

def calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col, **kwargs):
    """
    Placeholder for your actual NPS calculation logic.

    Args:
        image_array (np.ndarray): The input image data.
        pixel_spacing_row (float): Pixel spacing for rows.
        pixel_spacing_col (float): Pixel spacing for columns.
        **kwargs: Additional parameters specific to NPS.

    Returns:
        dict: A dictionary containing NPS results.
    """
    st.write(f"NPS calculation using image of shape: {image_array.shape} and pixel spacing: ({pixel_spacing_row}, {pixel_spacing_col})")

    # Example usage of pylinac functions (adjust parameters as needed)
    # Note: I'm assuming pixel_spacing is a single value for both dimensions for simplicity.
    # If they're different, you might need to adjust how you pass them.
    pixel_spacing = (pixel_spacing_row + pixel_spacing_col) / 2 # Average spacing

    # The noise_power_spectrum_2d function expects an iterable of ROIs.
    # If the whole image_array is considered as one ROI:
    rois_list = [image_array]

    try:
        nps_2d_result = noise_power_spectrum_2d(pixel_size=pixel_spacing, rois=rois_list)
        
        # Definition of NNPS:
        # NNPS(fx, fy) = NPS(fx, fy) / (pixel_spacing_col * pixel_spacing_row * image_array.shape[1] * image_array.shape[0])
        N_rows, N_cols = image_array.shape
        nnps_normalization_factor = (pixel_spacing_row * pixel_spacing_col * N_rows * N_cols)
        
        if nnps_normalization_factor == 0:
            st.error("Normalization factor for NNPS is zero. Check pixel spacings and image dimensions.")
            nnps_2d = np.full_like(nps_2d_result, np.nan)
        else:
            nnps_2d = nps_2d_result / nnps_normalization_factor

        nps_1d_result = noise_power_spectrum_1d(spectrum_2d=nps_2d_result)
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d)
        avg_power_result = average_power(nps1d=nps_1d_result)

        # Calculate NNPS at specific frequencies (0.5 mm^-1, 2.0 mm^-1)
        # Spatial frequency axes
        fy_axis = np.fft.fftshift(np.fft.fftfreq(N_rows, d=pixel_spacing_row))
        fx_axis = np.fft.fftshift(np.fft.fftfreq(N_cols, d=pixel_spacing_col))

        target_fx = 0.5  # mm^-1
        target_fy = 2.0  # mm^-1

        # Find the indices of the closest frequencies
        idx_fx = np.argmin(np.abs(fx_axis - target_fx))
        idx_fy = np.argmin(np.abs(fy_axis - target_fy))

        # Get the actual frequency values at these indices
        actual_fx_value = fx_axis[idx_fx]
        actual_fy_value = fy_axis[idx_fy]

        # Get the NNPS value at these (closest) frequencies
        # nnps_2d is already fftshifted as it's derived from nps_2d_result from pylinac
        nnps_at_target_freq = nnps_2d[idx_fy, idx_fx]

        # Calculate frequencies for 1D NNPS plot
        # The 1D NPS is a radial average. The frequency axis goes from 0 to Nyquist.
        # Nyquist frequency is 0.5 / pixel_spacing.
        # Use the maximum Nyquist frequency from both dimensions for the x-axis range.
        nyquist_freq_row = 0.5 / pixel_spacing_row
        nyquist_freq_col = 0.5 / pixel_spacing_col
        max_nyquist_freq = max(nyquist_freq_row, nyquist_freq_col)

        # Generate frequencies for the 1D NNPS plot
        # The number of points in the frequency axis should match the length of nnps_1d_result
        frequencies_nps = np.linspace(0, max_nyquist_freq, len(nnps_1d_result))
        nnps_data_for_chart = np.array([frequencies_nps, nnps_1d_result]).T
        x_axis_unit_nps = "lp/mm" # Assuming lp/mm as the standard unit for spatial frequency

        return {
            "NNPS_at_target_fx_fy": {
                "target_fx": float(target_fx),
                "target_fy": float(target_fy),
                "actual_fx": float(actual_fx_value),
                "actual_fy": float(actual_fy_value),
                "value": float(nnps_at_target_freq) if not np.isnan(nnps_at_target_freq) else np.nan
            },
            "NNPS_1D_values": nnps_1d_result.tolist(), # Keep original for potential other uses
            "NNPS_1D_chart_data": nnps_data_for_chart, # New data for plotting
            "x_axis_unit_nps": x_axis_unit_nps,
            "Average_Power": float(avg_power_result),
        }
    except Exception as e:
        st.error(f"Error during NPS calculation: {e}")
        return {"NPS_Status": f"Error: {e}"}

def display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    st.subheader("Noise Power Spectrum (NPS) Analysis")

    # Initialize session state for current NPS results
    if 'current_nps_results' not in st.session_state:
        st.session_state['current_nps_results'] = None

    if st.button("Run NPS Analysis"):
        with st.spinner("Calculating NPS..."):
            nps_results_dict = calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col)
            st.success("NPS Analysis Complete!")
            
            if nps_results_dict:
                st.subheader("1D Normalized Noise Power Spectrum")
                
                nnps_chart_data = nps_results_dict["NNPS_1D_chart_data"]
                x_axis_unit_nps = nps_results_dict["x_axis_unit_nps"]
                df_nnps = pd.DataFrame(nnps_chart_data, columns=[x_axis_unit_nps, 'NNPS'])
                st.line_chart(df_nnps.set_index(x_axis_unit_nps))
                
                # Display the NNPS at target frequency if available
                if "NNPS_at_target_fx_fy" in nps_results_dict:
                    st.subheader("NNPS at Target Frequencies")
                    target_info = nps_results_dict["NNPS_at_target_fx_fy"]
                    st.write(f"Target (fx, fy): ({target_info['target_fx']:.2f} mm⁻¹, {target_info['target_fy']:.2f} mm⁻¹)")
                    st.write(f"Actual (fx, fy): ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹)")
                    st.write(f"NNPS ({target_info['actual_fx']:.2f} mm⁻¹, {target_info['actual_fy']:.2f} mm⁻¹) : {target_info['value']:.4e}")

                st.session_state['nnps_data'] = nnps_chart_data
