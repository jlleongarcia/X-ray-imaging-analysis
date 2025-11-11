import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import Optional

try:
    from pylinac.core.mtf import EdgeMTF
except ImportError:
    EdgeMTF = None


def _bump_mtf_refresh():
    """Callback to force a Streamlit rerun when MTF inputs change."""
    st.session_state['mtf_refresh'] = st.session_state.get('mtf_refresh', 0) + 1


def calculate_mtf_metrics(
    edge_roi: np.ndarray,
    pixel_spacing: float,
) -> dict:
    """
    Calculate MTF using the slanted edge method (IEC 62220-1-1:2015) with full compliance corrections.

    Parameters
    ----------
    edge_roi : np.ndarray
        2D array containing the edge region for MTF analysis.
    pixel_spacing : float
        Physical pixel size in mm (average of row/col spacing).

    Returns
    -------
    dict
        Dictionary containing MTF results including frequencies, values, ESF, LSF, and spatial resolution metrics.
        
    IEC Corrections Applied
    -----------------------
    1. Spectral smoothing correction for finite-element differentiation (Section 5.1.4)
    2. Frequency axis scaling correction (1/cos α) for oblique projection
    """
    if edge_roi is None or not isinstance(edge_roi, np.ndarray) or edge_roi.ndim != 2:
        st.error("Valid 2D edge ROI is required for MTF calculation.")
        return {"MTF_Status": "Error: Invalid edge ROI"}

    if edge_roi.size < 100:  # Need sufficient pixels for meaningful edge analysis
        st.error("Edge ROI is too small. Select a larger region containing the edge.")
        return {"MTF_Status": "Error: ROI too small"}

    if pixel_spacing is None or pixel_spacing <= 0:
        st.warning("Pixel spacing is not valid. MTF will be calculated but spatial frequencies may be incorrect.")
        pixel_spacing = 0.1  # Default fallback
        x_axis_unit = "cycles/mm (approx)"
    else:
        x_axis_unit = "cycles/mm"

    if EdgeMTF is None:
        st.error("EdgeMTF class not available. Cannot perform IEC-compliant MTF analysis.")
        return {"MTF_Status": "Error: EdgeMTF not available"}

    try:
        # Initialize EdgeMTF without edge smoothing (not in IEC standard)
        edge_mtf = EdgeMTF(
            edge_data=edge_roi,
            pixel_size=pixel_spacing,
            edge_smoothing=0.0,  # No pre-smoothing per IEC
        )

        # Extract base results from EdgeMTF
        frequencies_raw = edge_mtf.frequencies
        mtf_values_raw = edge_mtf.mtf_values
        esf_positions = edge_mtf.esf_positions
        esf = edge_mtf.esf
        lsf = edge_mtf.lsf
        edge_angle_rad = edge_mtf.edge_angle
        edge_angle_deg = np.degrees(edge_angle_rad)

        # --- IEC Correction 1: Frequency axis scaling for oblique projection (1/cos α) ---
        # The edge angle causes frequency axis compression; correct by dividing by cos(α)
        cos_alpha = np.cos(edge_angle_rad)
        if abs(cos_alpha) < 1e-6:
            st.warning("Edge angle near 90°; frequency scaling may be unreliable.")
            freq_scale_factor = 1.0
        else:
            freq_scale_factor = 1.0 / cos_alpha
        
        frequencies_corrected = frequencies_raw * freq_scale_factor

        # --- IEC Correction 2: Spectral smoothing correction for finite-element differentiation ---
        # Differentiation (np.gradient) acts as a low-pass filter on MTF.
        # Correct by dividing MTF by the sinc function: sinc(π·f·Δx)
        # Δx is the ESF sampling interval (mean spacing between ESF positions)
        if len(esf_positions) > 1:
            delta_x = np.mean(np.diff(esf_positions)) * pixel_spacing  # in mm
        else:
            delta_x = pixel_spacing
        
        # Build sinc correction: sinc(π·f·Δx)
        # Avoid division by zero at f=0 and near-zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_arg = np.pi * frequencies_corrected * delta_x
            sinc_correction = np.sinc(sinc_arg / np.pi)  # np.sinc uses sinc(x) = sin(πx)/(πx)
            # Avoid division by very small sinc values (near zeros)
            sinc_correction = np.where(np.abs(sinc_correction) < 1e-6, 1.0, sinc_correction)
            mtf_values_corrected = mtf_values_raw / sinc_correction
        
        # Clip MTF to valid range [0, 1] after correction
        mtf_values_corrected = np.clip(mtf_values_corrected, 0.0, 1.0)

        # Calculate MTF50 and MTF10 from corrected MTF
        try:
            # Interpolate to find frequency at MTF = 0.5
            if np.any(mtf_values_corrected <= 0.5):
                mtf50 = np.interp(0.5, mtf_values_corrected[::-1], frequencies_corrected[::-1])
            else:
                mtf50 = np.nan
        except Exception:
            mtf50 = np.nan

        try:
            # Interpolate to find frequency at MTF = 0.1
            if np.any(mtf_values_corrected <= 0.1):
                mtf10 = np.interp(0.1, mtf_values_corrected[::-1], frequencies_corrected[::-1])
            else:
                mtf10 = np.nan
        except Exception:
            mtf10 = np.nan

        # Prepare chart data with corrected values
        mtf_chart_data = np.column_stack([frequencies_corrected, mtf_values_corrected])
        esf_chart_data = np.column_stack([esf_positions, esf])
        lsf_chart_data = np.column_stack([np.arange(len(lsf)), lsf])

        return {
            "frequencies": frequencies_corrected.tolist(),
            "mtf_values": mtf_values_corrected.tolist(),
            "mtf_chart_data": mtf_chart_data,
            "esf_chart_data": esf_chart_data,
            "lsf_chart_data": lsf_chart_data,
            "x_axis_unit": x_axis_unit,
            "MTF50": float(mtf50) if np.isfinite(mtf50) else "N/A",
            "MTF10": float(mtf10) if np.isfinite(mtf10) else "N/A",
            "edge_angle_deg": float(edge_angle_deg),
            "is_vertical": bool(edge_mtf.is_vertical),
            "nyquist_freq": float(frequencies_corrected[-1]) if len(frequencies_corrected) > 0 else np.nan,
            "freq_scale_factor": float(freq_scale_factor),
            "iec_corrections_applied": True,
        }

    except Exception as e:
        st.error(f"Error during MTF calculation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"MTF_Status": f"Error: {e}"}


def display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    """Display the MTF analysis UI with ROI selection and IEC-compliant edge method."""
    st.subheader("Modulation Transfer Function (MTF) Analysis")
    
    st.markdown("""
    **IEC 62220-1-1:2015 Slanted Edge Method**
    
    This analysis calculates MTF using the slanted edge technique:
    1. Select a rectangular ROI containing a sharp edge (contrast phantom)
    2. The edge should be slanted 3-5° from vertical or horizontal
    3. Edge Spread Function (ESF) is extracted perpendicular to the edge
    4. Line Spread Function (LSF) is derived by differentiating the ESF
    5. MTF is calculated via Fourier Transform of the LSF
    
    **Requirements:**
    - Edge phantom with sharp transition (high contrast)
    - Edge angle 3-5° from vertical/horizontal for optimal accuracy
    - Sufficient ROI size (at least 100x100 pixels recommended)
    """)

    # Initialize session state
    if 'mtf_roi_center_x' not in st.session_state:
        st.session_state['mtf_roi_center_x'] = 50
    if 'mtf_roi_center_y' not in st.session_state:
        st.session_state['mtf_roi_center_y'] = 50
    if 'mtf_roi_width' not in st.session_state:
        st.session_state['mtf_roi_width'] = 20
    if 'mtf_roi_height' not in st.session_state:
        st.session_state['mtf_roi_height'] = 20

    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        st.warning("Please upload a valid 2D image first.")
        return

    h, w = image_array.shape

    # ROI Selection Controls
    st.markdown("### Edge ROI Selection")
    st.caption("Define the rectangular region containing the edge for MTF analysis.")

    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "ROI Center X (%)",
            min_value=0,
            max_value=100,
            key='mtf_roi_center_x',
            on_change=_bump_mtf_refresh,
        )
        st.slider(
            "ROI Width (%)",
            min_value=5,
            max_value=100,
            key='mtf_roi_width',
            on_change=_bump_mtf_refresh,
        )

    with col2:
        st.slider(
            "ROI Center Y (%)",
            min_value=0,
            max_value=100,
            key='mtf_roi_center_y',
            on_change=_bump_mtf_refresh,
        )
        st.slider(
            "ROI Height (%)",
            min_value=5,
            max_value=100,
            key='mtf_roi_height',
            on_change=_bump_mtf_refresh,
        )

    # Extract ROI from image
    center_x_pct = st.session_state['mtf_roi_center_x']
    center_y_pct = st.session_state['mtf_roi_center_y']
    width_pct = st.session_state['mtf_roi_width']
    height_pct = st.session_state['mtf_roi_height']

    center_x_px = int(w * center_x_pct / 100)
    center_y_px = int(h * center_y_pct / 100)
    width_px = max(10, int(w * width_pct / 100))
    height_px = max(10, int(h * height_pct / 100))

    x0 = max(0, center_x_px - width_px // 2)
    x1 = min(w, center_x_px + width_px // 2)
    y0 = max(0, center_y_px - height_px // 2)
    y1 = min(h, center_y_px + height_px // 2)

    edge_roi = image_array[y0:y1, x0:x1]

    st.caption(f"ROI: [{y0}:{y1}, {x0}:{x1}] → {edge_roi.shape[0]}×{edge_roi.shape[1]} pixels")

    # Show ROI preview
    with st.expander("Preview Edge ROI"):
        roi_norm = (edge_roi - edge_roi.min()) / (edge_roi.max() - edge_roi.min() + 1e-9)
        st.image(roi_norm, caption="Selected Edge ROI (normalized)", use_container_width=True)

    # Pixel spacing
    if pixel_spacing_row is not None and pixel_spacing_col is not None and pixel_spacing_row > 0 and pixel_spacing_col > 0:
        pixel_spacing_avg = (pixel_spacing_row + pixel_spacing_col) / 2.0
    else:
        pixel_spacing_avg = 0.1
        st.warning("Pixel spacing unavailable or invalid; using default 0.1 mm/pixel.")

    # Add button to trigger MTF calculation
    st.markdown("---")
    if not st.button("Calculate MTF", key="mtf_calculate_button"):
        st.info("Click 'Calculate MTF' button to compute the Modulation Transfer Function.")
        return

    # Calculate MTF
    with st.spinner("Calculating IEC-compliant MTF with spectral and angle corrections..."):
        mtf_results = calculate_mtf_metrics(
            edge_roi=edge_roi,
            pixel_spacing=pixel_spacing_avg,
        )

    if "MTF_Status" in mtf_results and "Error" in mtf_results["MTF_Status"]:
        st.error(f"MTF Calculation Failed: {mtf_results['MTF_Status']}")
        return

    if not mtf_results or "mtf_chart_data" not in mtf_results:
        st.error("MTF calculation did not return expected results.")
        return

    st.success("MTF Analysis Complete!")

    # Display edge detection info
    edge_angle = mtf_results.get("edge_angle_deg", np.nan)
    is_vertical = mtf_results.get("is_vertical", False)
    orientation = "vertical" if is_vertical else "horizontal"
    freq_scale = mtf_results.get("freq_scale_factor", 1.0)
    
    st.info(f"**Edge detected:** {edge_angle:.2f}° from {orientation} | Orientation: {'Vertical' if is_vertical else 'Horizontal'}")
    
    # Show IEC corrections applied
    if mtf_results.get("iec_corrections_applied"):
        st.caption(f"✓ IEC corrections applied: Frequency scaling (1/cos α = {freq_scale:.4f}) | Spectral smoothing correction")
    
    if abs(edge_angle) > 10:
        st.warning(f"Edge angle ({edge_angle:.1f}°) is outside the recommended 3-5° range. Consider adjusting ROI or phantom alignment.")

    # MTF Curve
    st.subheader(f"MTF Curve")
    
    mtf_chart_data_np = mtf_results["mtf_chart_data"]
    df_mtf = pd.DataFrame(mtf_chart_data_np, columns=["Frequency", "MTF"])
    
    chart = alt.Chart(df_mtf).mark_line(clip=True, color='steelblue').encode(
        x=alt.X('Frequency:Q', title=f'Spatial Frequency ({mtf_results["x_axis_unit"]})'),
        y=alt.Y('MTF:Q', title='MTF', scale=alt.Scale(domain=[0, 1.05]))
    ).properties(
        title='Modulation Transfer Function (IEC 62220-1-1:2015)',
        height=400
    ).interactive()

    # Add hover interaction
    nearest = alt.selection_point(
        fields=['Frequency'],
        nearest=True,
        on='mouseover',
        empty=False,
        clear='mouseout'
    )

    selectors = alt.Chart(df_mtf).mark_point().encode(
        x='Frequency:Q',
        opacity=alt.value(0),
    ).add_params(nearest)

    points = chart.mark_circle(size=80).encode(
        opacity=alt.when(nearest).then(alt.value(1)).otherwise(alt.value(0)),
    )

    text = chart.mark_text(align='left', dx=7, dy=-7, fontSize=12, stroke='white', strokeWidth=1).encode(
        text=alt.when(nearest).then(alt.Text('MTF:Q', format='.3f')).otherwise(alt.value('')),
    )

    final_chart = alt.layer(chart, selectors, points, text)
    st.altair_chart(final_chart, use_container_width=True)

    # MTF Metrics
    st.subheader("MTF Spatial Resolution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mtf50_val = mtf_results.get('MTF50', 'N/A')
        mtf50_str = f"{mtf50_val:.3f}" if isinstance(mtf50_val, (int, float)) else mtf50_val
        st.metric("MTF 50%", f"{mtf50_str} {mtf_results['x_axis_unit']}")
    
    with col2:
        mtf10_val = mtf_results.get('MTF10', 'N/A')
        mtf10_str = f"{mtf10_val:.3f}" if isinstance(mtf10_val, (int, float)) else mtf10_val
        st.metric("MTF 10%", f"{mtf10_str} {mtf_results['x_axis_unit']}")
    
    with col3:
        nyquist = mtf_results.get('nyquist_freq', np.nan)
        nyq_str = f"{nyquist:.3f}" if np.isfinite(nyquist) else "N/A"
        st.metric("Nyquist Frequency", f"{nyq_str} {mtf_results['x_axis_unit']}")

    # ESF and LSF plots
    with st.expander("View Edge Spread Function (ESF) & Line Spread Function (LSF)"):
        col_esf, col_lsf = st.columns(2)
        
        with col_esf:
            st.markdown("**Edge Spread Function**")
            esf_data = mtf_results["esf_chart_data"]
            df_esf = pd.DataFrame(esf_data, columns=["Position", "ESF"])
            esf_chart = alt.Chart(df_esf).mark_line().encode(
                x=alt.X('Position:Q', title='Position (pixels)'),
                y=alt.Y('ESF:Q', title='Normalized Intensity')
            ).properties(height=300)
            st.altair_chart(esf_chart, use_container_width=True)
        
        with col_lsf:
            st.markdown("**Line Spread Function**")
            lsf_data = mtf_results["lsf_chart_data"]
            df_lsf = pd.DataFrame(lsf_data, columns=["Position", "LSF"])
            lsf_chart = alt.Chart(df_lsf).mark_line(color='orange').encode(
                x=alt.X('Position:Q', title='Position (pixels)'),
                y=alt.Y('LSF:Q', title='Amplitude')
            ).properties(height=300)
            st.altair_chart(lsf_chart, use_container_width=True)

    # Store results in session state
    st.session_state['mtf_data'] = mtf_results['mtf_chart_data']
