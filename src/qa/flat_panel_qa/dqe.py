import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import Optional, Dict, Any


def _validate_mtf_cache() -> tuple[bool, Optional[str]]:
    """Validate MTF cache exists and has geometric mean.
    
    Returns:
        (is_valid, error_message)
    """
    if 'mtf_cache' not in st.session_state:
        return False, "MTF analysis not performed. Please analyze MTF first."
    
    mtf_cache = st.session_state['mtf_cache']
    
    if not isinstance(mtf_cache, dict):
        return False, "MTF cache is corrupted."
    
    geom_mean = mtf_cache.get('mtf_geometric_mean')
    if not geom_mean or not geom_mean.get('available'):
        return False, "Geometric mean MTF not available. Please analyze TWO orthogonal edges (one vertical ~85-90°, one horizontal ~0-5°)."
    
    return True, None


def _validate_nps_cache() -> tuple[bool, Optional[str]]:
    """Validate NPS cache exists and has required data.
    
    Returns:
        (is_valid, error_message)
    """
    if 'nps_cache' not in st.session_state:
        return False, "NPS analysis not performed. Please analyze NPS first."
    
    nps_cache = st.session_state['nps_cache']
    
    if not isinstance(nps_cache, dict):
        return False, "NPS cache is corrupted."
    
    if 'results' not in nps_cache:
        return False, "NPS results not found in cache."
    
    if 'kerma_ugy' not in nps_cache:
        return False, "Kerma value not found in NPS cache."
    
    results = nps_cache['results']
    if 'NNPS_1D_chart_data' not in results:
        return False, "NNPS data not found in NPS results. Please re-run NPS analysis."
    
    return True, None


def compute_dqe_from_caches() -> Optional[Dict[str, Any]]:
    """Compute DQE from cached MTF and NPS data.
    
    Returns:
        Dictionary with DQE results or None if computation fails
    """
    # Validate caches
    mtf_valid, mtf_error = _validate_mtf_cache()
    if not mtf_valid:
        st.error(f"❌ MTF Cache Error: {mtf_error}")
        return None
    
    nps_valid, nps_error = _validate_nps_cache()
    if not nps_valid:
        st.error(f"❌ NPS Cache Error: {nps_error}")
        return None
    
    try:
        # Extract MTF geometric mean
        mtf_cache = st.session_state['mtf_cache']
        geom_mean = mtf_cache['mtf_geometric_mean']
        
        mtf_freq = np.array(geom_mean['frequencies'])
        mtf_values = np.array(geom_mean['mtf_values'])
        
        # Extract NNPS data for IEC 62220-1-1 DQE
        nps_cache = st.session_state['nps_cache']
        kerma_ugy = nps_cache['kerma_ugy']
        nps_results = nps_cache['results']
        
        nnps_data = nps_results['NNPS_1D_chart_data']
        nnps_freq = np.array([row[0] for row in nnps_data])
        nnps_values_um2 = np.array([row[1] for row in nnps_data])  # NNPS in μm²
        # Convert NNPS from μm² to mm² (1 mm² = 1e6 μm²)
        nnps_values = nnps_values_um2 / 1e6  # NNPS in mm²
        
        x_axis_unit = nps_results.get('x_axis_unit_nps', 'lp/mm')
        
        # Validate frequency units match
        if x_axis_unit not in ['lp/mm', 'cycles/mm']:
            st.warning(f"⚠️ Unexpected frequency unit: {x_axis_unit}. Assuming lp/mm.")
        
        # Create common frequency grid (intersection of both ranges)
        min_freq = max(mtf_freq.min(), nnps_freq.min())
        max_freq = min(mtf_freq.max(), nnps_freq.max())
        
        if min_freq >= max_freq:
            st.error("❌ No overlapping frequency range between MTF and NNPS.")
            return None
        
        # Common frequency grid with 300 points
        common_freq = np.linspace(min_freq, max_freq, 300)
        
        # Interpolate MTF and NNPS to common grid
        mtf_interp = np.interp(common_freq, mtf_freq, mtf_values)
        nnps_interp = np.interp(common_freq, nnps_freq, nnps_values)  # mm²
        
        # IEC 62220-1-1:2015 DQE formula:
        #   DQE(f) = [MTF²(f) / NNPS(f)] × K × SNR²_in
        # where:
        #   NNPS(f) is the normalised noise power spectrum in mm²
        #   K is the air kerma in µGy
        #   SNR²_in = 30174 1/(mm² · µGy) for RQA5 beam quality
        SNR2_IN_RQA5 = 30174  # 1/(mm² · µGy)
        
        dqe_values = (mtf_interp ** 2 / nnps_interp) * kerma_ugy * SNR2_IN_RQA5
        
        # Calculate key metrics
        # DQE(0) - low frequency performance
        dqe_0 = float(dqe_values[0])
        
        # Find frequencies at 50% and 10% of DQE(0)
        dqe_50_threshold = dqe_0 * 0.5
        dqe_10_threshold = dqe_0 * 0.1
        
        # Find frequency where DQE drops to 50% of DQE(0)
        idx_50 = np.where(dqe_values <= dqe_50_threshold)[0]
        freq_at_50 = float(common_freq[idx_50[0]]) if len(idx_50) > 0 else np.nan
        
        # Find frequency where DQE drops to 10% of DQE(0)
        idx_10 = np.where(dqe_values <= dqe_10_threshold)[0]
        freq_at_10 = float(common_freq[idx_10[0]]) if len(idx_10) > 0 else np.nan
        
        # Prepare chart data
        dqe_chart_data = np.column_stack([common_freq, dqe_values])
        
        return {
            'dqe_chart_data': dqe_chart_data,
            'frequencies': common_freq.tolist(),
            'dqe_values': dqe_values.tolist(),
            'x_axis_unit': x_axis_unit,
            'dqe_0': dqe_0,
            'freq_at_50_percent': freq_at_50,
            'freq_at_10_percent': freq_at_10,
            'kerma_ugy': kerma_ugy,
            'mtf_info': {
                'vertical_file': geom_mean['mtf_vertical']['filename'],
                'vertical_angle': geom_mean['mtf_vertical']['angle'],
                'horizontal_file': geom_mean['mtf_horizontal']['filename'],
                'horizontal_angle': geom_mean['mtf_horizontal']['angle']
            },
            'nps_info': {
                'num_images': nps_results.get('used_images', 'Unknown'),
                'total_pixels': nps_results.get('total_roi_pixels', 'Unknown')
            }
        }
        
    except Exception as e:
        st.error(f"❌ DQE Computation Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def _create_dqe_chart(df: pd.DataFrame, x_axis_unit: str, dqe_0: float) -> alt.Chart:
    """Create interactive Altair chart for DQE visualization.
    
    Args:
        df: DataFrame with 'Frequency' and 'DQE' columns
        x_axis_unit: Unit for frequency axis
        dqe_0: DQE at zero frequency for reference lines
    
    Returns:
        Altair chart object
    """
    x_label = f'Frequency ({x_axis_unit})'
    
    # Main DQE curve
    base = alt.Chart(df).mark_line(color='steelblue', strokeWidth=2.5).encode(
        x=alt.X(
            'Frequency:Q',
            title=x_label,
            scale=alt.Scale(domain=[df['Frequency'].min(), df['Frequency'].max()])
        ),
        y=alt.Y(
            'DQE:Q',
            title='Detective Quantum Efficiency (DQE)'
        )
    )
    
    # Reference lines at 50% and 10% of DQE(0)
    ref_50 = pd.DataFrame({
        'y': [dqe_0 * 0.5],
        'label': ['50% DQE(0)']
    })
    ref_10 = pd.DataFrame({
        'y': [dqe_0 * 0.1],
        'label': ['10% DQE(0)']
    })
    
    line_50 = alt.Chart(ref_50).mark_rule(color='orange', strokeDash=[5, 5]).encode(
        y='y:Q'
    )
    
    line_10 = alt.Chart(ref_10).mark_rule(color='red', strokeDash=[5, 5]).encode(
        y='y:Q'
    )
    
    # Interactive hover
    nearest = alt.selection_point(
        fields=['Frequency'],
        nearest=True,
        on='mouseover',
        empty=False
    )
    
    points = alt.Chart(df).mark_circle(size=80, color='steelblue').encode(
        x='Frequency:Q',
        y='DQE:Q',
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    ).add_params(nearest)
    
    text = alt.Chart(df).mark_text(
        align='left', dx=7, dy=-7, fontSize=14, fontWeight='normal'
    ).encode(
        x='Frequency:Q',
        y='DQE:Q',
        text=alt.condition(
            nearest,
            alt.Text('DQE:Q', format='.4f'),
            alt.value('')
        )
    )
    
    chart = (base + line_50 + line_10 + points + text).properties(
        title='Detective Quantum Efficiency (DQE)',
        height=400
    ).interactive()
    
    return chart


def display_dqe_analysis_section():
    """Display DQE analysis section in Streamlit UI."""
    st.subheader("Detective Quantum Efficiency (DQE) Analysis")
    
    st.markdown("""
    **IEC 62220-1-1:2015 DQE Calculation**
    
    DQE quantifies the detector's ability to transfer signal-to-noise ratio from input to output:
    
    $$\\text{DQE}(f) = \\frac{\\text{MTF}^2(f)}{\\text{NNPS}(f)} \\times K_a \\times SNR_{in}^2$$
    
    Where:
    - $\\text{MTF}(f)$ = Modulation Transfer Function (geometric mean of orthogonal edges)
    - $\\text{NNPS}(f)$ = Normalized Noise Power Spectrum (radial average, mm²)
    - $K_a$ = Air kerma in μGy
    - $SNR_{in}^2$ = 30174 $1/(mm^2 \\cdot μGy)$ for RQA5 beam quality
    - $f$ = Spatial frequency in lp/mm
    """)
    
    # Check cache status
    st.markdown("### Cache Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mtf_valid, mtf_error = _validate_mtf_cache()
        if mtf_valid:
            mtf_cache = st.session_state['mtf_cache']
            geom = mtf_cache.get('mtf_geometric_mean', {})
            st.success("✅ MTF: Geometric mean available")
            st.caption(f"Vertical: {geom.get('mtf_vertical', {}).get('filename', 'N/A')}")
            st.caption(f"Horizontal: {geom.get('mtf_horizontal', {}).get('filename', 'N/A')}")
        else:
            st.error(f"❌ MTF: {mtf_error}")
    
    with col2:
        nps_valid, nps_error = _validate_nps_cache()
        if nps_valid:
            nps_cache = st.session_state['nps_cache']
            kerma = nps_cache.get('kerma_ugy', 'N/A')
            st.success(f"✅ NPS: Cached (K = {kerma:.2f} μGy)")
            nps_results = nps_cache.get('results', {})
            num_images = nps_results.get('used_images', 'N/A')
            st.caption(f"Images used: {num_images}")
        else:
            st.error(f"❌ NPS: {nps_error}")
    
    # Show requirements if caches not ready
    if not (mtf_valid and nps_valid):
        st.markdown("---")
        st.info("""
        **Requirements for DQE Analysis:**
        
        1. **MTF Analysis**: Upload 2 images with orthogonal edge test patterns
           - One vertical edge (~85-87°)
           - One horizontal edge (~3-5°)
           - Analyze both in the MTF tab
        
        2. **NPS Analysis**: Upload multiple flat-field images
           - Minimum 4 million pixels recommended
           - Specify air kerma value (auto-detected if detector conversion available)
           - Analyze in the NPS tab
        
        Once both analyses are complete, return here to compute DQE.
        """)
        return
    
    # Compute DQE button
    st.markdown("---")
    if not st.button("Compute DQE", key="dqe_compute_button"):
        st.info("Click 'Compute DQE' to calculate Detective Quantum Efficiency.")
        return
    
    with st.spinner("Computing DQE..."):
        dqe_results = compute_dqe_from_caches()
    
    if dqe_results is None:
        return
    
    st.success("✅ DQE Analysis Complete!")
    
    # Display DQE metrics
    st.markdown("### DQE Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("DQE(0)", f"{dqe_results['dqe_0']:.4f}")
        st.caption("Low-frequency DQE")
    
    with col2:
        freq_50 = dqe_results['freq_at_50_percent']
        if np.isfinite(freq_50):
            st.metric("Frequency at 50% DQE(0)", f"{freq_50:.2f} {dqe_results['x_axis_unit']}")
        else:
            st.metric("Frequency at 50% DQE(0)", "N/A")
        st.caption("Half-value frequency")
    
    with col3:
        freq_10 = dqe_results['freq_at_10_percent']
        if np.isfinite(freq_10):
            st.metric("Frequency at 10% DQE(0)", f"{freq_10:.2f} {dqe_results['x_axis_unit']}")
        else:
            st.metric("Frequency at 10% DQE(0)", "N/A")
        st.caption("10% threshold frequency")
    
    # Display DQE curve
    st.markdown("### DQE Curve")
    
    df_dqe = pd.DataFrame(
        dqe_results['dqe_chart_data'],
        columns=['Frequency', 'DQE']
    )
    
    chart = _create_dqe_chart(df_dqe, dqe_results['x_axis_unit'], dqe_results['dqe_0'])
    st.altair_chart(chart, use_container_width=True)
    
    # Analysis details
    st.markdown("### Analysis Details")
    
    st.markdown("**MTF Source:**")
    mtf_info = dqe_results['mtf_info']
    st.write(f"- Vertical edge: {mtf_info['vertical_file']} (angle: {mtf_info['vertical_angle']:.1f}°)")
    st.write(f"- Horizontal edge: {mtf_info['horizontal_file']} (angle: {mtf_info['horizontal_angle']:.1f}°)")
    
    st.markdown("**NPS Source:**")
    nps_info = dqe_results['nps_info']
    st.write(f"- Air kerma: {dqe_results['kerma_ugy']:.2f} μGy")
    st.write(f"- Number of images: {nps_info['num_images']}")
    
    total_pixels = nps_info['total_pixels']
    if isinstance(total_pixels, (int, float)) and total_pixels > 0:
        million_pixels = total_pixels / 1_000_000
        st.write(f"- Total ROI pixels: {million_pixels:.2f} million")
    
    # Export option
    st.markdown("---")
    st.markdown("### Export DQE Data")
    
    csv_data = df_dqe.to_csv(index=False)
    st.download_button(
        label="Download DQE Data (CSV)",
        data=csv_data,
        file_name="dqe_data.csv",
        mime="text/csv",
        key="dqe_download_button"
    )
