import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import io
import csv
import time
from src.core.io.analysis_payload import ImagePayload

try:
    from pylinac.core.mtf import EdgeMTF
except ImportError:
    EdgeMTF = None


def _bump_mtf_refresh():
    """Callback to force a Streamlit rerun when MTF inputs change."""
    st.session_state['mtf_refresh'] = st.session_state.get('mtf_refresh', 0) + 1


def _compute_geometric_mean_mtf(mtf_results_list):
    """Compute geometric mean of two orthogonal MTF measurements.
    
    Args:
        mtf_results_list: List of MTF results with different orientations
    
    Returns:
        Dict with geometric mean MTF or None if requirements not met
    """
    if len(mtf_results_list) != 2:
        return None
    
    # Check if we have one vertical and one horizontal edge
    orientations = [r.get('is_vertical') for r in mtf_results_list]
    if True not in orientations or False not in orientations:
        return None  # Need one of each
    
    # Identify vertical and horizontal
    mtf_vertical = mtf_results_list[0] if mtf_results_list[0]['is_vertical'] else mtf_results_list[1]
    mtf_horizontal = mtf_results_list[0] if not mtf_results_list[0]['is_vertical'] else mtf_results_list[1]
    
    # Get frequency arrays
    freq_v = np.array(mtf_vertical['frequencies'])
    mtf_v = np.array(mtf_vertical['mtf_values'])
    freq_h = np.array(mtf_horizontal['frequencies'])
    mtf_h = np.array(mtf_horizontal['mtf_values'])
    
    # Create common frequency grid (use minimum Nyquist)
    max_freq = min(freq_v.max(), freq_h.max())
    common_freq = np.linspace(0, max_freq, 200)
    
    # Interpolate both MTFs to common grid
    mtf_v_interp = np.interp(common_freq, freq_v, mtf_v)
    mtf_h_interp = np.interp(common_freq, freq_h, mtf_h)
    
    # Compute geometric mean
    mtf_geometric_mean = np.sqrt(mtf_v_interp * mtf_h_interp)
    
    return {
        'frequencies': common_freq.tolist(),
        'mtf_values': mtf_geometric_mean.tolist(),
        'available': True,
        'mtf_vertical': {'filename': mtf_vertical['filename'], 'angle': mtf_vertical['edge_angle_deg']},
        'mtf_horizontal': {'filename': mtf_horizontal['filename'], 'angle': mtf_horizontal['edge_angle_deg']}
    }


def calculate_mtf_metrics(edge_roi: np.ndarray, pixel_spacing: float) -> dict:
    """Calculate MTF using the slanted edge method (IEC 62220-1-1:2015)."""
    if edge_roi is None or not isinstance(edge_roi, np.ndarray) or edge_roi.ndim != 2:
        st.error("Valid 2D edge ROI is required for MTF calculation.")
        return {"MTF_Status": "Error: Invalid edge ROI"}

    if edge_roi.size < 100:
        st.error("Edge ROI is too small. Select a larger region containing the edge.")
        return {"MTF_Status": "Error: ROI too small"}

    if pixel_spacing is None or pixel_spacing <= 0:
        st.warning("Pixel spacing is not valid. MTF will be calculated but spatial frequencies may be incorrect.")
        pixel_spacing = 0.1
        x_axis_unit = "cycles/mm (approx)"
    else:
        x_axis_unit = "cycles/mm"

    if EdgeMTF is None:
        st.error("EdgeMTF class not available. Cannot perform IEC-compliant MTF analysis.")
        return {"MTF_Status": "Error: EdgeMTF not available"}

    try:
        # Run EdgeMTF with built-in Hough Transform angle detection
        edge_mtf = EdgeMTF(edge_data=edge_roi, pixel_size=pixel_spacing, edge_smoothing=0.0)
        
        # Extract diagnostic information from EdgeMTF's Hough Transform detection
        edge_angle_deg = edge_mtf.edge_angle_deg
        is_vertical = edge_mtf.is_vertical
        hough_confidence = edge_mtf.hough_confidence
        edge_strength = edge_mtf.edge_strength
        edge_points_count = edge_mtf.edge_points_count
        
        # Extract MTF data (already IEC-compliant from EdgeMTF)
        frequencies = edge_mtf.frequencies
        mtf_values = edge_mtf.mtf_values
        esf_positions = edge_mtf.esf_positions
        esf = edge_mtf.esf
        lsf = edge_mtf.lsf

        # Calculate MTF50 and MTF10 using first sampled frequency at/under threshold
        # (robust against interpolation artifacts)
        freq_arr = np.asarray(frequencies, dtype=float)
        mtf_arr = np.asarray(mtf_values, dtype=float)

        try:
            mtf50 = freq_arr[np.where(mtf_arr <= 0.5)[0][0]] if np.any(mtf_arr <= 0.5) else np.nan
        except (IndexError, Exception):
            mtf50 = np.nan

        try:
            mtf10 = freq_arr[np.where(mtf_arr <= 0.1)[0][0]] if np.any(mtf_arr <= 0.1) else np.nan
        except (IndexError, Exception):
            mtf10 = np.nan

        # Prepare chart data (plot all data points without filtering)
        nyquist_freq = frequencies[-1] if len(frequencies) > 0 else np.nan
        mtf_chart_data = np.column_stack([frequencies, mtf_values])
        esf_chart_data = np.column_stack([esf_positions, esf])
        lsf_chart_data = np.column_stack([np.arange(len(lsf)), lsf])
        
        # Use full data for plotting (no limit)
        mtf_chart_data_plot = mtf_chart_data

        return {
            "frequencies": frequencies.tolist(),
            "mtf_values": mtf_values.tolist(),
            "mtf_chart_data": mtf_chart_data,
            "mtf_chart_data_plot": mtf_chart_data_plot,
            "esf_chart_data": esf_chart_data,
            "lsf_chart_data": lsf_chart_data,
            "x_axis_unit": x_axis_unit,
            "MTF50": float(mtf50) if np.isfinite(mtf50) else "N/A",
            "MTF10": float(mtf10) if np.isfinite(mtf10) else "N/A",
            "edge_angle_deg": float(edge_angle_deg),
            "is_vertical": bool(is_vertical),
            "hough_confidence": float(hough_confidence),
            "edge_strength": float(edge_strength),
            "edge_points_count": int(edge_points_count),
            "nyquist_freq": float(nyquist_freq) if np.isfinite(nyquist_freq) else np.nan,
            "iec_corrections_applied": True,
            "angle_detection_method": getattr(edge_mtf, 'angle_detection_method', 'Hough Transform'),
        }

    except Exception as e:
        st.error(f"Error during MTF calculation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {"MTF_Status": f"Error: {e}"}


def _create_mtf_chart(df_mtf, mtf_results, has_comparison):
    """Helper function to create MTF chart."""
    max_freq_in_data = df_mtf['Frequency'].max() if len(df_mtf) > 0 else 5
    
    x_encoding = alt.X('Frequency:Q', title=f'Spatial Frequency ({mtf_results["x_axis_unit"]})',
                       scale=alt.Scale(domain=[0, max_freq_in_data], nice=False, padding=0.2))
    y_encoding = alt.Y('MTF:Q', title='MTF', scale=alt.Scale(domain=[0, 1.05]))
    
    title = 'Modulation Transfer Function (IEC 62220-1-1:2015)'
    
    if has_comparison:
        # Check if geometric mean is present
        has_geom_mean = 'Geometric Mean (Isotropic)' in df_mtf['Image'].values
        if has_geom_mean:
            color_encoding = alt.Color('Image:N', legend=alt.Legend(title="Image"), 
                                       scale=alt.Scale(range=['steelblue', 'orange', 'green']))
        else:
            color_encoding = alt.Color('Image:N', legend=alt.Legend(title="Image"), 
                                       scale=alt.Scale(range=['steelblue', 'orange']))
        chart = alt.Chart(df_mtf).mark_line(clip=True).encode(x=x_encoding, y=y_encoding, color=color_encoding)
    else:
        chart = alt.Chart(df_mtf).mark_line(clip=True, color='steelblue').encode(x=x_encoding, y=y_encoding)
    
    return chart.properties(title=title, height=400).interactive()


def display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col, preloaded_files: list[ImagePayload] | None = None):
    """Display the MTF analysis UI with ROI selection and IEC-compliant edge method."""
    st.subheader("Modulation Transfer Function (MTF) Analysis")
    
    st.markdown("""
    **IEC 62220-1-1:2015 Slanted Edge Method**
    """)

    # Check if we're in comparison mode
    files_for_comparison = preloaded_files
    comparison_mode = files_for_comparison is not None and len(files_for_comparison) == 2
    
    if comparison_mode:
        images_data = []
        for idx, payload in enumerate(files_for_comparison):
            if not isinstance(payload, dict):
                st.error("Strict ingestion mode: expected preloaded payload dictionaries.")
                return
            img = payload.get('image_array')
            fname = payload.get('name', f'Image {idx + 1}')
            if not isinstance(img, np.ndarray) or img.ndim != 2:
                st.error(f"Strict ingestion mode: '{fname}' is missing decoded image_array from centralized loader.")
                return
            images_data.append((img, fname))

        if len(images_data) < 2:
            st.error("Could not load both images from preloaded payloads.")
            return

        st.success("✓ Loaded both images successfully")

        image_arrays = [img for img, _ in images_data]
        filenames = [name for _, name in images_data]
    else:
        if image_array is None or not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
            st.warning("Please upload a valid 2D image first.")
            return
        image_arrays = [image_array]
        filenames = ["Current Image"]


    # ROI Selection
    st.markdown("### Edge ROI Selection")
    roi_params = []
    if comparison_mode and len(image_arrays) == 2:
        colA, colB = st.columns(2)
        slider_cols = [colA, colB]
    else:
        slider_cols = [st.container() for _ in image_arrays]

    for idx, img in enumerate(image_arrays):
        h, w = img.shape
        suffix = f"_{idx+1}" if comparison_mode else ""
        # Unique keys for each image
        keys = {
            'center_x': f'mtf_roi_center_x{suffix}',
            'center_y': f'mtf_roi_center_y{suffix}',
            'width': f'mtf_roi_width{suffix}',
            'height': f'mtf_roi_height{suffix}'
        }
        # Initialize session state for each image
        for k, key in keys.items():
            if key not in st.session_state:
                st.session_state[key] = 50 if 'center' in k else 20
        with slider_cols[idx]:
            st.markdown(f"**ROI for {filenames[idx]}**")
            st.slider("ROI Center X (%)", 0, 100, key=keys['center_x'], on_change=_bump_mtf_refresh)
            st.slider("ROI Center Y (%)", 0, 100, key=keys['center_y'], on_change=_bump_mtf_refresh)
            st.slider("ROI Width (%)", 5, 100, key=keys['width'], on_change=_bump_mtf_refresh)
            st.slider("ROI Height (%)", 5, 100, key=keys['height'], on_change=_bump_mtf_refresh)
        # Calculate pixel coordinates for ROI
        center_x_px = int(w * st.session_state[keys['center_x']] / 100)
        center_y_px = int(h * st.session_state[keys['center_y']] / 100)
        width_px = max(10, int(w * st.session_state[keys['width']] / 100))
        height_px = max(10, int(h * st.session_state[keys['height']] / 100))
        x0, x1 = max(0, center_x_px - width_px // 2), min(w, center_x_px + width_px // 2)
        y0, y1 = max(0, center_y_px - height_px // 2), min(h, center_y_px + height_px // 2)
        roi_params.append({'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'w': w, 'h': h, 'width_px': width_px, 'height_px': height_px, 'center_x_px': center_x_px, 'center_y_px': center_y_px})

    # Pixel spacing
    pixel_spacing_avg = ((pixel_spacing_row + pixel_spacing_col) / 2.0 
                        if pixel_spacing_row and pixel_spacing_col and pixel_spacing_row > 0 
                        else 0.1)
    
    if not (pixel_spacing_row and pixel_spacing_col and pixel_spacing_row > 0):
        st.warning("Pixel spacing unavailable; using default 0.1 mm/pixel.")

    # Show ROI Preview for all images
    with st.expander("🔍 Edge ROI Selection Preview", expanded=False):
        st.caption("Preview of the selected ROI on each image where MTF will be analyzed")
        # Create columns for each image preview
        if len(image_arrays) == 1:
            preview_cols = [st.container()]
        else:
            preview_cols = st.columns(len(image_arrays))
        for idx, (img, fname) in enumerate(zip(image_arrays, filenames)):
            roi = roi_params[idx]
            h, w = roi['h'], roi['w']
            x0, x1, y0, y1 = roi['x0'], roi['x1'], roi['y0'], roi['y1']
            width_px, height_px = roi['width_px'], roi['height_px']
            with preview_cols[idx] if len(image_arrays) > 1 else preview_cols[0]:
                st.markdown(f"**{fname}**")
                # Create preview with ROI overlay
                img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-10)
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
                img_with_roi = img_rgb.copy()
                border_thickness = max(2, min(h, w) // 200)
                # Top and bottom borders
                img_with_roi[max(0, y0-border_thickness):y0, x0:x1, 0] = 1.0  # Red
                img_with_roi[max(0, y0-border_thickness):y0, x0:x1, 1:] = 0.0
                img_with_roi[y1:min(h, y1+border_thickness), x0:x1, 0] = 1.0
                img_with_roi[y1:min(h, y1+border_thickness), x0:x1, 1:] = 0.0
                # Left and right borders
                img_with_roi[y0:y1, max(0, x0-border_thickness):x0, 0] = 1.0
                img_with_roi[y0:y1, max(0, x0-border_thickness):x0, 1:] = 0.0
                img_with_roi[y0:y1, x1:min(w, x1+border_thickness), 0] = 1.0
                img_with_roi[y0:y1, x1:min(w, x1+border_thickness), 1:] = 0.0
                # Add semi-transparent red tint inside ROI
                alpha = 0.3
                img_with_roi[y0:y1, x0:x1, 0] = img_with_roi[y0:y1, x0:x1, 0] * (1 - alpha) + alpha
                st.image(img_with_roi, caption=f"ROI: {width_px}×{height_px} px", use_container_width=True)
                st.caption(f"Position: ({x0}, {y0}) to ({x1}, {y1})")


    # Calculate MTF
    st.markdown("---")
    if st.button("Calculate MTF", key="mtf_calculate_button"):
        all_mtf_results = []
        with st.spinner(f"Calculating MTF for {len(image_arrays)} image(s)..."):
            for idx, (img, fname) in enumerate(zip(image_arrays, filenames)):
                roi = roi_params[idx]
                edge_roi = img[roi['y0']:roi['y1'], roi['x0']:roi['x1']]
                mtf_result = calculate_mtf_metrics(edge_roi, pixel_spacing_avg)
                if "MTF_Status" not in mtf_result or "Error" not in mtf_result.get("MTF_Status", ""):
                    mtf_result['filename'] = fname
                    all_mtf_results.append(mtf_result)
        if not all_mtf_results:
            st.error("No MTF results were successfully calculated.")
        else:
            st.session_state['mtf_cache'] = {
                'results': all_mtf_results,
                'timestamp': time.time(),
                'mtf_geometric_mean': _compute_geometric_mean_mtf(all_mtf_results)
            }

    # Render from cache (persists across reruns like detector_conversion.py)
    if 'mtf_cache' not in st.session_state:
        st.info("Click 'Calculate MTF' to compute.")
        return

    all_mtf_results = st.session_state['mtf_cache']['results']
    st.success("✅ MTF Analysis Complete!")

    # Display edge detection information for each image
    for mtf_results in all_mtf_results:
        fname = mtf_results['filename']
        edge_angle = mtf_results.get("edge_angle_deg", np.nan)
        is_vertical = mtf_results.get("is_vertical", False)
        hough_conf = mtf_results.get("hough_confidence", np.nan)
        edge_strength = mtf_results.get("edge_strength", np.nan)
        orientation = "Vertical" if is_vertical else "Horizontal"
        
        # IEC optimal range validation
        optimal_range = (85, 87) if is_vertical else (3, 5)
        abs_angle = abs(edge_angle)
        
        if optimal_range[0] <= abs_angle <= optimal_range[1]:
            st.info(f"**{fname}:** Edge angle {abs_angle:.2f}° ({orientation}) - ✅ Within IEC optimal range {optimal_range[0]}-{optimal_range[1]}° | Confidence: {hough_conf:.1%}")
        else:
            st.warning(f"**{fname}:** Edge angle {abs_angle:.2f}° ({orientation}) - ⚠️ Outside IEC optimal range {optimal_range[0]}-{optimal_range[1]}° | Confidence: {hough_conf:.1%}")

    # MTF Curve - Combine all results
    st.subheader("MTF Curve")
    
    # Build dataframe with all MTF curves
    dfs_to_concat = []
    for mtf_results in all_mtf_results:
        mtf_chart_data_plot = mtf_results.get("mtf_chart_data_plot", mtf_results["mtf_chart_data"])
        df = pd.DataFrame(mtf_chart_data_plot, columns=["Frequency", "MTF"])
        df['Image'] = mtf_results['filename']
        dfs_to_concat.append(df)
    
    # Add geometric mean curve if available
    geom_mean = st.session_state['mtf_cache'].get('mtf_geometric_mean')
    if geom_mean and geom_mean.get('available'):
        df_geom = pd.DataFrame({
            'Frequency': geom_mean['frequencies'],
            'MTF': geom_mean['mtf_values'],
            'Image': 'Geometric Mean (Isotropic)'
        })
        dfs_to_concat.append(df_geom)
        st.success("✅ Orthogonal edges detected! Geometric mean MTF computed for DQE analysis.")
    
    df_mtf = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Create and display chart
    mtf_results = all_mtf_results[0]
    chart = _create_mtf_chart(df_mtf, mtf_results, len(all_mtf_results) > 1)
    
    # Add hover interaction
    nearest = alt.selection_point(fields=['Frequency'], nearest=True, on='mouseover', 
                                  empty=False, clear='mouseout')
    selectors = alt.Chart(df_mtf).mark_point().encode(x='Frequency:Q', opacity=alt.value(0)).add_params(nearest)
    points = chart.mark_circle(size=80).encode(
        opacity=alt.when(nearest).then(alt.value(1)).otherwise(alt.value(0)))
    text = chart.mark_text(align='left', dx=7, dy=-7, fontSize=12, stroke='white', strokeWidth=1).encode(
        text=alt.when(nearest).then(alt.Text('MTF:Q', format='.3f')).otherwise(alt.value('')))
    
    final_chart = alt.layer(chart, selectors, points, text)
    st.altair_chart(final_chart, use_container_width=True)

    # MTF Metrics - Show for each image
    for idx, mtf_results in enumerate(all_mtf_results):
        fname = mtf_results['filename']
        st.subheader(f"MTF Metrics - {fname}")
        col1, col2, col3 = st.columns(3)
        
        for col, key, label in [(col1, 'MTF50', "MTF 50%"), (col2, 'MTF10', "MTF 10%"), 
                                (col3, 'nyquist_freq', "Nyquist Freq")]:
            with col:
                val = mtf_results.get(key, 'N/A')
                val_str = f"{val:.3f}" if isinstance(val, (int, float)) and np.isfinite(val) else "N/A"
                st.metric(label, f"{val_str} {mtf_results['x_axis_unit']}")
    
    # Geometric Mean MTF Metrics
    geom_mean = st.session_state['mtf_cache'].get('mtf_geometric_mean')
    if geom_mean and geom_mean.get('available'):
        st.subheader("MTF Metrics - Geometric Mean (Isotropic)")
        
        # Calculate MTF50 and MTF10 for geometric mean
        freq_geom = np.array(geom_mean['frequencies'])
        mtf_geom = np.array(geom_mean['mtf_values'])
        
        try:
            mtf50_geom = freq_geom[np.where(mtf_geom <= 0.5)[0][0]] if np.any(mtf_geom <= 0.5) else np.nan
        except (IndexError, Exception):
            mtf50_geom = np.nan
        
        try:
            mtf10_geom = freq_geom[np.where(mtf_geom <= 0.1)[0][0]] if np.any(mtf_geom <= 0.1) else np.nan
        except (IndexError, Exception):
            mtf10_geom = np.nan
        
        nyquist_geom = freq_geom[-1] if len(freq_geom) > 0 else np.nan
        
        col1, col2, col3 = st.columns(3)
        with col1:
            val_str = f"{mtf50_geom:.3f}" if np.isfinite(mtf50_geom) else "N/A"
            st.metric("MTF 50%", f"{val_str} {all_mtf_results[0]['x_axis_unit']}")
        with col2:
            val_str = f"{mtf10_geom:.3f}" if np.isfinite(mtf10_geom) else "N/A"
            st.metric("MTF 10%", f"{val_str} {all_mtf_results[0]['x_axis_unit']}")
        with col3:
            val_str = f"{nyquist_geom:.3f}" if np.isfinite(nyquist_geom) else "N/A"
            st.metric("Nyquist Freq", f"{val_str} {all_mtf_results[0]['x_axis_unit']}")

    # ESF and LSF plots
    with st.expander("View Edge & Line Spread Functions"):
        for idx, mtf_results in enumerate(all_mtf_results):
            st.markdown(f"### {mtf_results['filename']}")
            col_esf, col_lsf = st.columns(2)
            
            with col_esf:
                st.markdown("**Edge Spread Function**")
                df_esf = pd.DataFrame(mtf_results["esf_chart_data"], columns=["Position", "ESF"])
                esf_chart = alt.Chart(df_esf).mark_line().encode(
                    x=alt.X('Position:Q', title='Position (pixels)'),
                    y=alt.Y('ESF:Q', title='Intensity')
                ).properties(height=250)
                st.altair_chart(esf_chart, use_container_width=True)
            
            with col_lsf:
                st.markdown("**Line Spread Function**")
                df_lsf = pd.DataFrame(mtf_results["lsf_chart_data"], columns=["Position", "LSF"])
                lsf_chart = alt.Chart(df_lsf).mark_line(color='orange').encode(
                    x=alt.X('Position:Q', title='Position (pixels)'),
                    y=alt.Y('LSF:Q', title='Amplitude')
                ).properties(height=250)
                st.altair_chart(lsf_chart, use_container_width=True)
            
            if idx < len(all_mtf_results) - 1:
                st.markdown("---")

    # --- CSV Export ---
    st.markdown("---")
    csv_output = io.StringIO()
    csv_writer = csv.writer(csv_output)

    # Summary section
    csv_writer.writerow(['=== MTF Analysis Summary ==='])
    csv_writer.writerow([])
    x_unit = all_mtf_results[0]['x_axis_unit']
    csv_writer.writerow(['Image', 'Edge Angle (deg)', 'Orientation', 'Hough Confidence',
                         'Edge Strength', 'Edge Points', f'MTF50 ({x_unit})',
                         f'MTF10 ({x_unit})', f'Nyquist ({x_unit})'])
    for r in all_mtf_results:
        nyq = r.get('nyquist_freq', np.nan)
        csv_writer.writerow([
            r['filename'],
            f"{r.get('edge_angle_deg', 0):.2f}",
            'Vertical' if r.get('is_vertical') else 'Horizontal',
            f"{r.get('hough_confidence', 0):.4f}",
            f"{r.get('edge_strength', 0):.2f}",
            r.get('edge_points_count', ''),
            r.get('MTF50', 'N/A'),
            r.get('MTF10', 'N/A'),
            f"{nyq:.4f}" if isinstance(nyq, (int, float)) and np.isfinite(nyq) else 'N/A',
        ])

    # Geometric mean summary
    geom_csv = st.session_state.get('mtf_cache', {}).get('mtf_geometric_mean')
    if geom_csv and geom_csv.get('available'):
        freq_g = np.array(geom_csv['frequencies'])
        mtf_g = np.array(geom_csv['mtf_values'])
        try:
            mtf50_g = float(freq_g[np.where(mtf_g <= 0.5)[0][0]])
        except (IndexError, Exception):
            mtf50_g = 'N/A'
        try:
            mtf10_g = float(freq_g[np.where(mtf_g <= 0.1)[0][0]])
        except (IndexError, Exception):
            mtf10_g = 'N/A'
        csv_writer.writerow(['Geometric Mean (Isotropic)', '', '', '', '', '',
                            mtf50_g, mtf10_g, ''])

    csv_writer.writerow([])

    # MTF Curve data section
    csv_writer.writerow(['=== MTF Curve Data ==='])
    mtf_header = []
    mtf_data_pairs = []
    for r in all_mtf_results:
        mtf_header.extend([f'Frequency ({x_unit}) - {r["filename"]}', f'MTF - {r["filename"]}'])
        mtf_data_pairs.append((np.array(r['frequencies']), np.array(r['mtf_values'])))

    if geom_csv and geom_csv.get('available'):
        mtf_header.extend([f'Frequency ({x_unit}) - Geometric Mean', 'MTF - Geometric Mean'])
        mtf_data_pairs.append((np.array(geom_csv['frequencies']), np.array(geom_csv['mtf_values'])))

    mtf_max_len = max(len(p[0]) for p in mtf_data_pairs) if mtf_data_pairs else 0
    csv_writer.writerow(mtf_header)
    for i in range(mtf_max_len):
        row = []
        for freq_arr, mtf_arr in mtf_data_pairs:
            if i < len(freq_arr):
                row.extend([f"{freq_arr[i]:.6f}", f"{mtf_arr[i]:.6f}"])
            else:
                row.extend(['', ''])
        csv_writer.writerow(row)

    csv_writer.writerow([])

    # ESF and LSF data sections
    for r in all_mtf_results:
        csv_writer.writerow([f'=== ESF - {r["filename"]} ==='])
        csv_writer.writerow(['Position (pixels)', 'ESF'])
        for row_data in r['esf_chart_data']:
            csv_writer.writerow([f"{row_data[0]:.6f}", f"{row_data[1]:.6f}"])
        csv_writer.writerow([])

        csv_writer.writerow([f'=== LSF - {r["filename"]} ==='])
        csv_writer.writerow(['Position', 'LSF'])
        for row_data in r['lsf_chart_data']:
            csv_writer.writerow([f"{row_data[0]:.6f}", f"{row_data[1]:.6f}"])
        csv_writer.writerow([])

    st.download_button(
        label="\U0001f4e5 Download MTF Results (CSV)",
        data=csv_output.getvalue(),
        file_name="mtf_analysis_results.csv",
        mime="text/csv"
    )
