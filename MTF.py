import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pydicom
import io
import time

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

        # Calculate MTF50 and MTF10 using EdgeMTF's built-in method
        try:
            mtf50 = edge_mtf.spatial_resolution(50)
        except Exception:
            mtf50 = np.nan

        try:
            mtf10 = edge_mtf.spatial_resolution(10)
        except Exception:
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


def _load_image_from_file(uploaded_file):
    """Helper function to load DICOM image from uploaded file."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    # Try DICOM parsing
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        if hasattr(ds, 'pixel_array'):
            img = ds.pixel_array
            if img.ndim == 2:
                return img, filename
    except Exception:
        pass
    
    return None, filename


def _load_raw_file(uploaded_file, dtype, height, width):
    """Load RAW file with specified dtype and dimensions (like detector_conversion.py)."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    try:
        arr = np.frombuffer(file_bytes, dtype=dtype)
        expected_size = height * width
        
        if arr.size != expected_size:
            st.warning(f"⚠️ {filename}: Size mismatch (expected {expected_size} pixels, got {arr.size})")
            return None, filename
        
        img = arr.reshape((height, width))
        return img, filename
    except Exception as e:
        st.error(f"❌ Failed to load {filename} as RAW: {e}")
        return None, filename


def _create_mtf_chart(df_mtf, mtf_results, has_comparison):
    """Helper function to create MTF chart."""
    max_freq_in_data = df_mtf['Frequency'].max() if len(df_mtf) > 0 else 5
    
    x_encoding = alt.X('Frequency:Q', title=f'Spatial Frequency ({mtf_results["x_axis_unit"]})',
                       scale=alt.Scale(domain=[0, max_freq_in_data], nice=False, padding=0.2))
    y_encoding = alt.Y('MTF:Q', title='MTF', scale=alt.Scale(domain=[0, 1.05]))
    
    title = 'Modulation Transfer Function (IEC 62220-1-1:2015)'
    
    if has_comparison:
        color_encoding = alt.Color('Image:N', legend=alt.Legend(title="Image"), 
                                   scale=alt.Scale(range=['steelblue', 'orange']))
        chart = alt.Chart(df_mtf).mark_line(clip=True).encode(x=x_encoding, y=y_encoding, color=color_encoding)
    else:
        chart = alt.Chart(df_mtf).mark_line(clip=True, color='steelblue').encode(x=x_encoding, y=y_encoding)
    
    return chart.properties(title=title, height=400).interactive()


def display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col, uploaded_files=None, raw_params=None):
    """Display the MTF analysis UI with ROI selection and IEC-compliant edge method."""
    st.subheader("Modulation Transfer Function (MTF) Analysis")
    
    st.markdown("""
    **IEC 62220-1-1:2015 Slanted Edge Method**
    """)

    # Check if we're in comparison mode
    comparison_mode = uploaded_files is not None and len(uploaded_files) == 2
    
    if comparison_mode:
        
        # For RAW files, all files use the same dtype/dimensions
        if raw_params is not None:
            images_data = []
            for uf in uploaded_files:
                img, fname = _load_raw_file(uf, raw_params['dtype'], raw_params['height'], raw_params['width'])
                if img is not None:
                    images_data.append((img, fname))
                else:
                    st.error(f"⚠️ Could not load {uf.name}")
                    return
            
            if len(images_data) == 2:
                st.success(f"✓ Loaded both RAW files successfully")
        else:
            # DICOM files or pre-loaded image_array
            if image_array is None:
                st.error("⚠️ First image not loaded.")
                return
            
            images_data = [(image_array, uploaded_files[0].name)]
            
            # Try loading second as DICOM
            img2, fname2 = _load_image_from_file(uploaded_files[1])
            
            if img2 is not None:
                images_data.append((img2, fname2))
                st.success(f"✓ Loaded both images successfully")
            else:
                st.error(f"⚠️ Could not load second image '{uploaded_files[1].name}'")
                return
        
        if len(images_data) < 2:
            st.error("Could not load both images.")
            return
            
        image_arrays = [img for img, _ in images_data]
        filenames = [name for _, name in images_data]
    else:
        if image_array is None or not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
            st.warning("Please upload a valid 2D image first.")
            return
        image_arrays = [image_array]
        filenames = ["Current Image"]

    # Initialize session state
    for key, default in [('mtf_roi_center_x', 50), ('mtf_roi_center_y', 50), 
                         ('mtf_roi_width', 20), ('mtf_roi_height', 20)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Use first image dimensions for ROI selection
    h, w = image_arrays[0].shape

    # ROI Selection
    st.markdown("### Edge ROI Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("ROI Center X (%)", 0, 100, key='mtf_roi_center_x', on_change=_bump_mtf_refresh)
        st.slider("ROI Width (%)", 5, 100, key='mtf_roi_width', on_change=_bump_mtf_refresh)
    with col2:
        st.slider("ROI Center Y (%)", 0, 100, key='mtf_roi_center_y', on_change=_bump_mtf_refresh)
        st.slider("ROI Height (%)", 5, 100, key='mtf_roi_height', on_change=_bump_mtf_refresh)

    # Extract ROI coordinates (same for all images)
    center_x_px = int(w * st.session_state['mtf_roi_center_x'] / 100)
    center_y_px = int(h * st.session_state['mtf_roi_center_y'] / 100)
    width_px = max(10, int(w * st.session_state['mtf_roi_width'] / 100))
    height_px = max(10, int(h * st.session_state['mtf_roi_height'] / 100))

    x0, x1 = max(0, center_x_px - width_px // 2), min(w, center_x_px + width_px // 2)
    y0, y1 = max(0, center_y_px - height_px // 2), min(h, center_y_px + height_px // 2)

    # Pixel spacing
    pixel_spacing_avg = ((pixel_spacing_row + pixel_spacing_col) / 2.0 
                        if pixel_spacing_row and pixel_spacing_col and pixel_spacing_row > 0 
                        else 0.1)
    
    if not (pixel_spacing_row and pixel_spacing_col and pixel_spacing_row > 0):
        st.warning("Pixel spacing unavailable; using default 0.1 mm/pixel.")

    # Calculate MTF
    st.markdown("---")
    if not st.button("Calculate MTF", key="mtf_calculate_button"):
        st.info("Click 'Calculate MTF' to compute.")
        return

    # Calculate MTF for all images
    all_mtf_results = []
    with st.spinner(f"Calculating MTF for {len(image_arrays)} image(s)..."):
        for img, fname in zip(image_arrays, filenames):
            edge_roi = img[y0:y1, x0:x1]
            mtf_result = calculate_mtf_metrics(edge_roi, pixel_spacing_avg)
            
            if "MTF_Status" not in mtf_result or "Error" not in mtf_result.get("MTF_Status", ""):
                mtf_result['filename'] = fname
                all_mtf_results.append(mtf_result)

    if not all_mtf_results:
        st.error("No MTF results were successfully calculated.")
        return

    st.success("✅ MTF Analysis Complete!")
    
    # Store results in session state cache
    st.session_state['mtf_cache'] = {
        'results': all_mtf_results,
        'timestamp': time.time(),
        'mtf_geometric_mean': _compute_geometric_mean_mtf(all_mtf_results)
    }

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
