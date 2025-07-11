import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, radial_average
import pandas as pd
import altair as alt

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
        x_axis_unit_nps = "cycles/pixel"
    else:
        x_axis_unit_nps = "lp/mm"

    pixel_spacing_avg = (pixel_spacing_row + pixel_spacing_col) / 2

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

        # We are taking into account we are analyzing a difference image, so NPS must be halved.
        nps_2d_result = nps_2d_raw / 2
        
        # NNPS definition: NNPS(fx, fy) = NPS(fx, fy) / (Mean Pixel Value of Largest ROI)^2
        nnps_2d = nps_2d_result / (mean_pv ** 2)

        # --- 1D NNPS Calculation ---
        # Radially average the 2D NNPS to get the 1D NNPS
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d)

        # To get the corresponding frequency axis, we do the same radial average
        # on a grid of radial frequencies. This is the most accurate method.
        # 1. Create the frequency axes for the 2D grid.
        freqs = np.fft.fftshift(np.fft.fftfreq(selected_small, d=pixel_spacing_avg))

        # 2. Create the 2D frequency grid (meshgrid).
        fx, fy = np.meshgrid(freqs, freqs)
        f_grid = np.sqrt(fx**2 + fy**2)

        # 3. Apply the same radial average to get the 1D frecuency axis
        freqs_nnps1d = radial_average(f_grid)

        # 4. Combine into a single array for charting and interpolation.
        nnps_data_for_chart = np.array([freqs_nnps1d, nnps_1d_result]).T

        # --- NNPS at specific frecuencies ---
        target_f1 = 0.5  # lp/mm
        target_f2 = 2.0  # lp/mm

        # Interpolate to find the NNPS value at the exact target frequencies.
        # This is more accurate than finding the nearest index.
        nnps_1d_at_f1 = np.interp(target_f1, nnps_data_for_chart[:,0], nnps_data_for_chart[:,1], left=np.nan, right=np.nan)
        nnps_1d_at_f2 = np.interp(target_f2, nnps_data_for_chart[:,0], nnps_data_for_chart[:,1], left=np.nan, right=np.nan)

        # Calculate frequencies for 1D NNPS plot
        # The 1D NPS is a radial average. The frequency axis goes from 0 to Nyquist.
        # Nyquist frequency is 0.5 / pixel_spacing.
        # Use the maximum Nyquist frequency from both dimensions for the x-axis range.
        nyquist_freq = 0.5 / pixel_spacing_avg

        return {
            "Nyquist": nyquist_freq,
            "NNPS_at_target_f": {
                "target_f1": float(target_f1),
                "value_1": float(nnps_1d_at_f1),
                "target_f2": float(target_f2),
                "value_2": float(nnps_1d_at_f2),
            },
            "NNPS_1D_chart_data": nnps_data_for_chart,
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
                
                nyquist_freq = nps_results_dict["Nyquist"]
                nnps_chart_data = nps_results_dict["NNPS_1D_chart_data"]
                x_axis_unit_nps = nps_results_dict["x_axis_unit_nps"]
                df_nnps = pd.DataFrame(nnps_chart_data, columns=[x_axis_unit_nps, 'NNPS_1D'])

                # ---- Create the Altair chart ----
                chart = alt.Chart(df_nnps).mark_line(clip=True).encode(
                    x=alt.X(x_axis_unit_nps, scale=alt.Scale(domainMax=nyquist_freq)),
                    y='NNPS_1D'
                ).properties(
                    title='Radial Average of Normalized Noise Power Spectrum (NNPS)'
                ).interactive()

                # Create a selection that finds the nearest point based on X axis
                # 'on="mouseover"' makes it activate when the mouse hovers
                # 'empty="none"' ensures the selection is cleared when mouse leaves the chart
                nearest_selection = alt.selection_point(
                    fields=[x_axis_unit_nps],
                    nearest=True,
                    on='mouseover',
                    empty='none',
                    clear='mouseout' # Clear when mouse leaves the chart area
                )

                # Transparent selectors to enable the nearest selection across the entire chart width
                selectors = alt.Chart(df_nnps).mark_point().encode(
                    x=x_axis_unit_nps,
                    opacity=alt.value(0),
                ).add_params(nearest_selection)

                # Text labels to display the x and y values, near the mouse's x-coordinate.
                # First, create a text source for the text label:
                text_source = chart.transform_calculate(
                    hover_text=f"'NNPS: ' + format(datum.NNPS_1D, '.3f')"
                )
                text = text_source.mark_text(align='left', dx=7, dy=-7, fontSize=14, fontWeight="normal", stroke='white', strokeWidth=1).encode(
                    text=alt.when(nearest_selection).then(
                        alt.Text('hover_text:N')
                    ).otherwise(
                        alt.value('')
                    ),
                )

                # Optionally, small circles to highlight the nearest data point on the line
                points = chart.mark_circle().encode(
                    opacity=alt.when(nearest_selection).then(alt.value(1)).otherwise(alt.value(0)),
                ).add_params(nearest_selection)

                # Layer all these components together
                final_chart = alt.layer(
                    chart, selectors, points, text
                )

                st.altair_chart(final_chart, use_container_width=True)
                
                # Display the NNPS at target frequency if available
                if "NNPS_at_target_f" in nps_results_dict:
                    st.subheader("NNPS at Target Frequencies")
                    target_info = nps_results_dict["NNPS_at_target_f"]
                    nnps_value_1 = f"{target_info['value_1']:.3f}" if not np.isnan(target_info['value_1']) else "N/A" # .e for scientific notation
                    nnps_value_2 = f"{target_info['value_2']:.3f}" if not np.isnan(target_info['value_2']) else "N/A"
                    
                    st.write(f"**NNPS at {target_info['target_f1']:.2f} {x_axis_unit_nps}**: {nnps_value_1}")
                    st.write(f"**NNPS at {target_info['target_f2']:.2f} {x_axis_unit_nps}**: {nnps_value_2}")


                st.session_state['nnps_data'] = nnps_chart_data
            else:
                st.error("NPS calculation did not return expected results.")
