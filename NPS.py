import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, radial_average
import pandas as pd
import altair as alt
from typing import List, Optional
import io
import time
try:
    import pydicom
except Exception:
    pydicom = None
try:
    from PIL import Image
except Exception:
    Image = None

def _central_roi_mean(image: np.ndarray, roi_size: int) -> float:
    """Compute the mean of a centered square ROI of given size."""
    h, w = image.shape
    roi_h = min(roi_size, h)
    roi_w = min(roi_size, w)
    y0 = (h - roi_h) // 2
    x0 = (w - roi_w) // 2
    roi = image[y0:y0+roi_h, x0:x0+roi_w]
    return float(np.mean(roi)) if roi.size else np.nan

def _autodetect_kerma_from_detector_conversion(image: np.ndarray, default_kerma: float = 2.5) -> float:
    """Auto-detect kerma value using detector conversion cache.
    
    Uses central 100×100 ROI mean pixel value and applies inverse detector 
    conversion function to estimate air kerma.
    
    Args:
        image: Input image array
        default_kerma: Default value to return if auto-detection fails
    
    Returns:
        Estimated kerma in μGy, or default_kerma if detection unavailable
    """
    if 'detector_conversion' not in st.session_state:
        return default_kerma
    
    try:
        # Extract central 100×100 ROI mean
        central_mpv = _central_roi_mean(image, roi_size=100)
        if not np.isfinite(central_mpv):
            return default_kerma
        
        # Get detector conversion parameters
        det_conv = st.session_state['detector_conversion']
        if isinstance(det_conv, dict) and isinstance(det_conv.get('fit'), dict):
            det_conv = det_conv.get('fit', {})
        fit_params = det_conv.get('fit_params', {})
        
        # Try MPV->Kerma conversion
        if 'MPV_vs_Kerma' in fit_params:
            params = fit_params['MPV_vs_Kerma']
            a = params.get('a', 0)
            b = params.get('b', 0)
            func_type = params.get('func_type', 'linear')
            
            if func_type == 'linear' and a != 0:
                # Inverse: k = (m - b) / a
                kerma_estimate = (central_mpv - b) / a
            elif func_type == 'exponential' and a != 0:
                # Inverse: k = (ln(m) - b) / a
                if central_mpv > 0:
                    kerma_estimate = (np.log(central_mpv) - b) / a
                else:
                    return default_kerma
            else:
                return default_kerma
            
            # Sanity check (kerma should be positive and reasonable)
            if 0.1 <= kerma_estimate <= 100:  # typical range for DR systems
                return float(kerma_estimate)
    
    except Exception:
        pass
    
    return default_kerma

def _try_load_dicom(data: bytes) -> Optional[np.ndarray]:
    """Attempt to load image data as DICOM.
    
    Args:
        data: Raw file bytes
    
    Returns:
        2D float array if successful, None otherwise
    """
    if pydicom is None:
        return None
    
    try:
        # Check for DICM marker or valid DICOM header
        is_dicom_like = False
        if len(data) >= 132 and data[128:132] == b'DICM':
            is_dicom_like = True
        else:
            try:
                ds_hdr = pydicom.dcmread(io.BytesIO(data), force=True, stop_before_pixels=True)
                if hasattr(ds_hdr, 'Rows') and hasattr(ds_hdr, 'Columns'):
                    is_dicom_like = True
            except Exception:
                pass
        
        if is_dicom_like:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            if hasattr(ds, 'pixel_array'):
                arr = ds.pixel_array
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    arr = np.mean(arr, axis=-1)
                if isinstance(arr, np.ndarray):
                    return arr.astype(float)
    except Exception:
        pass
    return None

def _try_load_pil_image(data: bytes) -> Optional[np.ndarray]:
    """Attempt to load image data using PIL.
    
    Args:
        data: Raw file bytes
    
    Returns:
        2D float array if successful, None otherwise
    """
    if Image is None:
        return None
    
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=float)
    except Exception:
        return None

def _try_load_raw_std(data: bytes, ext: str, reference_shape: Optional[tuple], 
                      reference_dtype: Optional[np.dtype], fname: str) -> Optional[np.ndarray]:
    """Attempt to load RAW/STD file using reference shape and dtype.
    
    Args:
        data: Raw file bytes
        ext: File extension
        reference_shape: Expected image shape (H, W)
        reference_dtype: Expected data type
        fname: Filename for warning messages
    
    Returns:
        2D float array if successful, None otherwise
    """
    if ext not in ('raw', 'std'):
        return None
    
    if not reference_shape or len(reference_shape) != 2 or not reference_dtype:
        return None
    
    expected_pixels = reference_shape[0] * reference_shape[1]
    itemsize = np.dtype(reference_dtype).itemsize
    total_bytes = len(data)
    expected_bytes = expected_pixels * itemsize
    
    if total_bytes != expected_bytes:
        st.warning(f"RAW/STD '{fname}' size mismatch: bytes={total_bytes}, expected={expected_bytes}; skipped.")
        return None
    
    try:
        return np.frombuffer(data, dtype=reference_dtype).reshape(reference_shape).astype(float)
    except Exception:
        return None


def _bump_nps_refresh():
    """Callback to force a Streamlit rerun marker when any NPS input changes."""
    st.session_state['nps_refresh'] = st.session_state.get('nps_refresh', 0) + 1

def _apply_inverse_detector_conversion(img: np.ndarray, conv: dict) -> tuple[np.ndarray, str]:
    """Apply inverse detector conversion (pixel -> kerma) to image.
    
    Args:
        img: 2D image array in pixel domain
        conv: Detector conversion dictionary from session state
    
    Returns:
        Tuple of (converted_image, domain_used) where domain is 'kerma' or 'pixel'
    """
    if not (isinstance(conv, dict) and conv.get("coeffs") is not None):
        return img, "pixel"
    
    method = conv.get("method")
    coeffs = np.array(conv.get("coeffs"), dtype=float)
    
    try:
        if method == 'linear':
            a, b = coeffs
            if a == 0:
                raise ValueError("Inverse linear conversion undefined (a=0)")
            return (img.astype(float) - b) / a, "kerma"
        elif method == 'log':
            a, b = coeffs
            if a == 0:
                raise ValueError("Inverse log conversion undefined (a=0)")
            with np.errstate(over='ignore', invalid='ignore'):
                result = np.exp((img.astype(float) - b) / a)
            return result, "kerma"
        else:
            return img, "pixel"
    except Exception as e:
        st.warning(f"Inverse detector conversion failed ({e}); proceeding with pixel-domain NPS.")
        return img, "pixel"

def _compute_nnps_profiles(nnps_2d: np.ndarray, freqs: np.ndarray) -> tuple:
    """Extract x and y component profiles from 2D NNPS.
    
    Args:
        nnps_2d: 2D normalized NPS array
        freqs: Frequency axis (1D, shifted)
    
    Returns:
        Tuple of (freqs_positive, nnps_x_positive, nnps_y_positive)
    """
    center_idx = nnps_2d.shape[0] // 2
    nnps_x_profile = nnps_2d[center_idx, :]  # horizontal profile
    nnps_y_profile = nnps_2d[:, center_idx]  # vertical profile
    
    freqs_positive = freqs[center_idx:]
    nnps_x_positive = nnps_x_profile[center_idx:]
    nnps_y_positive = nnps_y_profile[center_idx:]
    
    return freqs_positive, nnps_x_positive, nnps_y_positive

def _create_nnps_dataframe(nnps_1d_data: np.ndarray, nnps_x_data: np.ndarray, 
                           nnps_y_data: np.ndarray, x_label: str) -> pd.DataFrame:
    """Create combined dataframe for NNPS plotting.
    
    Args:
        nnps_1d_data: 1D radial NNPS data [freq, value]
        nnps_x_data: X-component NNPS data [freq, value]
        nnps_y_data: Y-component NNPS data [freq, value]
        x_label: Label for frequency axis
    
    Returns:
        Combined DataFrame with Component column
    """
    df_1d = pd.DataFrame(nnps_1d_data, columns=[x_label, 'NNPS'])
    df_1d['Component'] = '1D Radial'
    
    df_x = pd.DataFrame(nnps_x_data, columns=[x_label, 'NNPS'])
    df_x['Component'] = 'X-component'
    
    df_y = pd.DataFrame(nnps_y_data, columns=[x_label, 'NNPS'])
    df_y['Component'] = 'Y-component'
    
    return pd.concat([df_1d, df_x, df_y], ignore_index=True)


def _load_uploaded_images(files, reference_shape: Optional[tuple]=None, reference_dtype: Optional[np.dtype]=None) -> List[np.ndarray]:
    """Load uploaded files into 2D numpy float arrays.

    Priority order:
      1. DICOM (including pseudo-DICOM .raw/.std with header) via pydicom.
      2. Standard image formats via PIL -> grayscale.
      3. True RAW/STD without header: if reference shape & dtype provided AND byte length matches, reshape.

    Parameters
    ----------
    files : list
        Uploaded files from Streamlit uploader.
    reference_shape : tuple (H, W), optional
        Shape of the primary image to interpret additional raw/std files.
    reference_dtype : np.dtype, optional
        Dtype to assume for additional raw/std files when reshaping.
    """
    arrays: List[np.ndarray] = []

    for f in files or []:
        fname = getattr(f, 'name', 'unknown') or 'unknown'
        ext = fname.lower().rsplit('.', 1)[-1] if '.' in fname else ''

        # Read bytes once
        try:
            data = f.getvalue() if hasattr(f, 'getvalue') else f.read()
        except Exception:
            data = None
        
        if not data:
            st.warning(f"Empty or unreadable file: {fname}")
            continue

        # Try loading in priority order
        arr = _try_load_dicom(data)
        if arr is None:
            arr = _try_load_pil_image(data)
        if arr is None:
            arr = _try_load_raw_std(data, ext, reference_shape, reference_dtype, fname)

        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > 0:
            arrays.append(arr)
        else:
            st.warning(f"Could not interpret '{fname}' as a 2D image; skipped.")
    
    return arrays


def calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col, additional_images: Optional[List[np.ndarray]] = None,
                          big_roi_size: int = 1024, small_roi_size: int = 128):
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
    if image_array is None or (not isinstance(image_array, np.ndarray)) or image_array.ndim != 2 or image_array.size == 0:
        st.error("Valid 2D image_array is required for NPS calculation.")
        return {"NPS_Status": "Error: Invalid image array"}

    if pixel_spacing_row is None or pixel_spacing_col is None or pixel_spacing_row <= 0 or pixel_spacing_col <= 0:
        st.warning("Pixel spacing is not valid or available. NPS spatial frequencies will be in cycles/pixel.")
        x_axis_unit_nps = "cycles/pixel"
        pixel_spacing_avg = 1.0  # treat 1 pixel as unit distance
    else:
        x_axis_unit_nps = "lp/mm"
        pixel_spacing_avg = (pixel_spacing_row + pixel_spacing_col) / 2

    # Collect all images (base + additional)
    all_images: List[np.ndarray] = [image_array]
    if additional_images:
        all_images.extend([img for img in additional_images if isinstance(img, np.ndarray) and img.ndim == 2])

    # Deduplicate identical images (avoid double-counting the current image if it's also in uploads)
    deduped: List[np.ndarray] = []
    for img in all_images:
        is_dup = False
        for u in deduped:
            if img.shape == u.shape and np.array_equal(img, u):
                is_dup = True
                break
        if not is_dup:
            deduped.append(img)
    all_images = deduped

    # Ensure all images are same shape
    shapes = {img.shape for img in all_images}
    if len(shapes) > 1:
        st.error("All images must have identical dimensions for IEC NPS calculation. Please upload images with matching size.")
        return {"NPS_Status": "Error: Image sizes do not match"}

    # Apply inverse detector conversion to all images
    conv = st.session_state.get("detector_conversion")
    if isinstance(conv, dict) and isinstance(conv.get('fit'), dict):
        conv = conv.get('fit', {})
    domain_used = "pixel"
    converted_images = []
    for img in all_images:
        converted_img, domain = _apply_inverse_detector_conversion(img, conv)
        converted_images.append(converted_img)
        if domain == "kerma":
            domain_used = "kerma"
    all_images = converted_images

    try:
        # Compute NPS for each uploaded image independently and average the resulting NNPS across images.
        # Multiple images improve statistics when the image size is small. All images must share identical dimensions.

        # Enforce consistency between ROI sizes
        selected_big = int(big_roi_size)
        selected_small = int(min(small_roi_size, selected_big))
        nps_2d_list = []
        mean_pvs: List[float] = []
        for idx, img in enumerate(all_images):
            imgf = img.astype(float)
            nps_2d_raw, mean_pv = noise_power_spectrum_2d(
                imgf,
                pixel_size=pixel_spacing_avg,
                big_roi_size=selected_big,
                small_roi_size=selected_small
            )
            if not np.isfinite(mean_pv) or mean_pv == 0:
                st.warning(f"Image {idx+1}: non-finite or zero mean PV; skipping from NNPS averaging.")
                continue
            nps_2d_list.append(nps_2d_raw)
            mean_pvs.append(float(mean_pv))

        if not nps_2d_list:
            st.error("No valid images were processed for NNPS. Please verify uploaded images and exposure consistency.")
            return {"NPS_Status": "Error: No valid images"}

        # Average raw NPS across images, then normalize once by a reference mean^2 (average of per-image means)
        nps_2d_avg_raw = np.mean(np.stack(nps_2d_list, axis=0), axis=0)
        mean_ref = float(np.mean(mean_pvs))
        nnps_2d_avg = nps_2d_avg_raw / (mean_ref ** 2)
        
        # Convert from mm² to μm² for better readability (multiply by 10⁶)
        nnps_2d_avg_um2 = nnps_2d_avg * 1e6
        nnps_units = "μm²"

        # Total ROI pixels (simplified): area of the large ROI per image times number of analyzed images
        total_roi_pixels = int(selected_big * selected_big * len(nps_2d_list))

        # --- 1D NNPS Calculation from averaged 2D NNPS ---
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d_avg_um2)

        # To get the corresponding frequency axis, we do the same radial average
        # on a grid of radial frequencies. This is the most accurate method.
        # 1. Create the frequency axes for the 2D grid.
        freqs = np.fft.fftshift(np.fft.fftfreq(selected_small, d=pixel_spacing_avg))

        # 2. Create the 2D frequency grid (meshgrid).
        fx, fy = np.meshgrid(freqs, freqs)
        f_grid = np.sqrt(fx**2 + fy**2)

        # 3. Apply the same radial average to get the 1D frequency axis
        freqs_nnps1d = radial_average(f_grid)

        # 4. Extract x and y component profiles
        freqs_positive, nnps_x_positive, nnps_y_positive = _compute_nnps_profiles(nnps_2d_avg_um2, freqs)

        # 5. Combine into arrays for charting and interpolation
        nnps_data_for_chart = np.array([freqs_nnps1d, nnps_1d_result]).T
        nnps_x_data_for_chart = np.array([freqs_positive, nnps_x_positive]).T
        nnps_y_data_for_chart = np.array([freqs_positive, nnps_y_positive]).T

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
                "units": nnps_units,
            },
            "NNPS_1D_chart_data": nnps_data_for_chart,
            "NNPS_X_chart_data": nnps_x_data_for_chart,
            "NNPS_Y_chart_data": nnps_y_data_for_chart,
            "x_axis_unit_nps": x_axis_unit_nps,
            "domain_used": domain_used,
            "nnps_units": nnps_units,
            "used_images": int(len(nps_2d_list)),
            "total_roi_pixels": total_roi_pixels,
        }
    except Exception as e:
        st.error(f"Error during NPS calculation: {e}")
        return {"NPS_Status": f"Error: {e}"}

def _create_nnps_chart(df_combined: pd.DataFrame, x_label: str, nyquist_freq: float, nnps_units: str) -> alt.LayerChart:
    """Create interactive Altair chart for NNPS visualization.
    
    Args:
        df_combined: DataFrame with frequency, NNPS, and Component columns
        x_label: Label for x-axis (frequency)
        nyquist_freq: Maximum frequency for x-axis
        nnps_units: Units for NNPS (e.g., 'μm²')
    
    Returns:
        Layered Altair chart with hover interactions
    """
    base_chart = alt.Chart(df_combined).mark_line(clip=True).encode(
        x=alt.X(x_label, scale=alt.Scale(domainMax=nyquist_freq)),
        y=alt.Y('NNPS', title=f'NNPS ({nnps_units})'),
        color=alt.Color(
            'Component:N',
            scale=alt.Scale(
                domain=['1D Radial', 'X-component', 'Y-component'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            legend=alt.Legend(title="NNPS Component")
        ),
        strokeDash=alt.StrokeDash(
            'Component:N',
            scale=alt.Scale(
                domain=['1D Radial', 'X-component', 'Y-component'],
                range=[[], [5, 5], [2, 2]]
            ),
            legend=None
        ),
        strokeWidth=alt.StrokeWidth(
            'Component:N',
            scale=alt.Scale(
                domain=['1D Radial', 'X-component', 'Y-component'],
                range=[2, 1.5, 1.5]
            ),
            legend=None
        )
    ).properties(
        title='Normalized Noise Power Spectrum (NNPS)'
    ).interactive()

    # Hover selection and interactive elements
    nearest_selection = alt.selection_point(
        fields=[x_label],
        nearest=True,
        on='mouseover',
        empty='none',
        clear='mouseout'
    )

    selectors = alt.Chart(df_combined).mark_point().encode(
        x=x_label,
        opacity=alt.value(0),
    ).add_params(nearest_selection)

    text = alt.Chart(df_combined).mark_text(
        align='left', dx=7, dy=-7, fontSize=14, fontWeight="normal", 
        stroke='white', strokeWidth=1
    ).transform_calculate(
        hover_text="datum.Component + ': ' + format(datum.NNPS, '.4f')"
    ).encode(
        x=x_label,
        y='NNPS',
        text=alt.when(nearest_selection).then('hover_text:N').otherwise(alt.value('')),
        color=alt.Color(
            'Component:N',
            scale=alt.Scale(
                domain=['1D Radial', 'X-component', 'Y-component'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            legend=None
        )
    )

    points = alt.Chart(df_combined).mark_circle(size=60).encode(
        x=x_label,
        y='NNPS',
        color=alt.Color(
            'Component:N',
            scale=alt.Scale(
                domain=['1D Radial', 'X-component', 'Y-component'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            legend=None
        ),
        opacity=alt.when(nearest_selection).then(alt.value(1)).otherwise(alt.value(0)),
    ).add_params(nearest_selection)

    return alt.layer(base_chart, selectors, points, text)

def display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col, uploaded_files=None):
    st.subheader("Noise Power Spectrum (NPS) Analysis")

    st.write("""
    **IEC 62220-1-1:2015 Compliant NPS Analysis**
    
    Computes the normalized noise power spectrum (NNPS) according to IEC 62220-1-1:2015 standard.
    The analysis averages NNPS across all uploaded images to improve statistical accuracy.
    
    **Requirements:**
    - All images must have identical dimensions
    - Images should be flat-field exposures at consistent exposure levels
    - Aim for ~4 million total ROI pixels for adequate statistics
    
    Upload multiple images in the sidebar to meet the statistical requirements.
    """)

    # Load all uploaded images for NPS analysis
    ref_dtype = image_array.dtype if isinstance(image_array, np.ndarray) else None
    ref_shape = image_array.shape if isinstance(image_array, np.ndarray) and image_array.ndim == 2 else None
    additional_arrays = _load_uploaded_images(uploaded_files, reference_shape=ref_shape, reference_dtype=ref_dtype) if uploaded_files else []

    # Big ROI size selector in mm (IEC recommends 125 mm; user-adjustable)
    if 'nps_big_roi_mm' not in st.session_state:
        st.session_state['nps_big_roi_mm'] = 125.0

    st.select_slider(
        "Big ROI size (mm)",
        options=[float(v) for v in range(25, 251, 5)],
        key='nps_big_roi_mm',
        on_change=_bump_nps_refresh,
    )

    BIG_ROI_SIZE_MM = float(st.session_state['nps_big_roi_mm'])
    
    # Calculate big_roi_size in pixels based on pixel spacing
    if pixel_spacing_row is not None and pixel_spacing_col is not None and pixel_spacing_row > 0 and pixel_spacing_col > 0:
        pixel_spacing_avg = (pixel_spacing_row + pixel_spacing_col) / 2.0
        big_roi_pixels = int(np.round(BIG_ROI_SIZE_MM / pixel_spacing_avg))
    else:
        # Fallback if pixel spacing not available: assume 0.1 mm/pixel
        pixel_spacing_avg = 0.1
        big_roi_pixels = int(np.round(BIG_ROI_SIZE_MM / pixel_spacing_avg))
        st.warning("Pixel spacing unavailable; assuming 0.1 mm/pixel for ROI calculations.")

    # Small ROI size selector (persistent via session_state)
    allowed_small = [8, 16, 32, 64, 128, 256, 512]
    if 'nps_small_roi' not in st.session_state:
        st.session_state['nps_small_roi'] = 128

    st.select_slider(
        "Small ROI size (pixels)",
        options=allowed_small,
        key='nps_small_roi',
        on_change=_bump_nps_refresh,
    )

    # Kerma input with auto-detection
    st.markdown("---")
    st.markdown("### Air Kerma (for DQE Analysis)")
    
    # Auto-detect kerma from detector conversion if available
    detected_kerma = _autodetect_kerma_from_detector_conversion(image_array, default_kerma=2.5)
    
    if detected_kerma != 2.5 and 'detector_conversion' in st.session_state:
        st.success(f"✅ Auto-detected kerma from detector conversion: {detected_kerma:.2f} μGy")
    
    # Initialize session state for kerma
    if 'nps_kerma_value' not in st.session_state:
        st.session_state['nps_kerma_value'] = detected_kerma
    
    kerma_value = st.number_input(
        "Air Kerma (μGy)",
        min_value=0.01,
        max_value=1000.0,
        value=float(st.session_state.get('nps_kerma_value', detected_kerma)),
        step=0.1,
        format="%.2f",
        key='nps_kerma_input',
        on_change=_bump_nps_refresh,
        help="Air kerma value used for DQE computation. Auto-detected from detector conversion if available."
    )
    st.session_state['nps_kerma_value'] = kerma_value

    # Add button to trigger NPS calculation
    st.markdown("---")
    if not st.button("Calculate NPS", key="nps_calculate_button"):
        st.info("Click 'Calculate NPS' button to compute the Noise Power Spectrum.")
        return

    with st.spinner("Updating NPS..."):
        big_roi = big_roi_pixels  # Use calculated value from 125 mm
        small_roi = st.session_state['nps_small_roi']
        nps_results_dict = calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col,
                                                 additional_images=additional_arrays,
                                                 big_roi_size=big_roi, small_roi_size=small_roi)
        # Mark last recompute time
        st.session_state['nps_last_updated'] = time.time()
        
        # Store results in session state cache with kerma value
        if "NPS_Status" not in nps_results_dict or "Error" not in nps_results_dict.get("NPS_Status", ""):
            st.session_state['nps_cache'] = {
                'results': nps_results_dict,
                'kerma_ugy': kerma_value,
                'timestamp': time.time()
            }

    if "NPS_Status" in nps_results_dict and "Error" in nps_results_dict["NPS_Status"]:
        st.error(f"NPS Calculation Failed: {nps_results_dict['NPS_Status']}")
        return
    if not (nps_results_dict and "NNPS_1D_chart_data" in nps_results_dict):
        st.error("NPS calculation did not return expected results.")
        return

    st.success("NPS Analysis Updated!")
    st.subheader("Normalized Noise Power Spectrum")

    nyquist_freq = nps_results_dict["Nyquist"]
    nnps_chart_data = nps_results_dict["NNPS_1D_chart_data"]
    x_axis_unit_nps = nps_results_dict["x_axis_unit_nps"]
    df_nnps = pd.DataFrame(nnps_chart_data, columns=[x_axis_unit_nps, 'NNPS_1D'])

    # Inform user of domain used for NPS calculation
    domain_used = nps_results_dict.get("domain_used", "pixel")
    if domain_used == "kerma":
        st.info("NPS computed on kerma-domain image (after inverse detector conversion).")
    else:
        st.caption("NPS computed on raw pixel values (no inverse conversion applied).")

    # Show how many images were used (if any)
    used_images = nps_results_dict.get("used_images")
    total_roi_pixels = nps_results_dict.get("total_roi_pixels")
    if used_images is not None:
        st.caption(f"Number of images used: {used_images}")
    if total_roi_pixels is not None:
        million_equiv = total_roi_pixels / 1_000_000.0
        if total_roi_pixels < 4_000_000:
            st.warning(f"Total ROI pixels ~ {million_equiv:.2f} million (< 4 million IEC guideline). Consider uploading more images for improved NPS statistics.")
        else:
            st.caption(f"Total ROI pixels ~ {million_equiv:.2f} million (meets/exceeds 4 million guideline).")

    # ---- Create the Altair chart with x and y components ----
    if "NNPS_X_chart_data" in nps_results_dict and "NNPS_Y_chart_data" in nps_results_dict:
        df_combined = _create_nnps_dataframe(
            nps_results_dict["NNPS_1D_chart_data"],
            nps_results_dict["NNPS_X_chart_data"],
            nps_results_dict["NNPS_Y_chart_data"],
            x_axis_unit_nps
        )
    else:
        # Fallback to 1D only
        df_combined = df_nnps.copy()
        df_combined['Component'] = '1D Radial'
        df_combined['NNPS'] = df_combined['NNPS_1D']
    
    # Get NNPS units and create chart
    nnps_units = nps_results_dict.get('nnps_units', 'μm²')
    final_chart = _create_nnps_chart(df_combined, x_axis_unit_nps, nyquist_freq, nnps_units)
    st.altair_chart(final_chart, use_container_width=True)

    # Display the NNPS at target frequency if available
    if "NNPS_at_target_f" in nps_results_dict:
        st.subheader("NNPS at Target Frequencies")
        target_info = nps_results_dict["NNPS_at_target_f"]
        target_units = target_info.get('units', nnps_units)
        nnps_value_1 = f"{target_info['value_1']:.3f}" if not np.isnan(target_info['value_1']) else "N/A"
        nnps_value_2 = f"{target_info['value_2']:.3f}" if not np.isnan(target_info['value_2']) else "N/A"
        st.write(f"**NNPS at {target_info['target_f1']:.2f} {x_axis_unit_nps}**: {nnps_value_1} {target_units}")
        st.write(f"**NNPS at {target_info['target_f2']:.2f} {x_axis_unit_nps}**: {nnps_value_2} {target_units}")
