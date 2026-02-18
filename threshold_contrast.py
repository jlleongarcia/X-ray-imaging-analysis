"""
Threshold Contrast Detail Detectability (TCDD) - Statistical Method
Based on Chao et al. (2000) and Paruccini et al. (2021)

This module implements the statistical method for assessing low-contrast detectability
in X-ray imaging systems using homogeneous phantom images.

References:
- Chao, E.H., et al. (2000). A statistical method of defining low contrast detectability.
- Paruccini, N., et al. (2021). A single phantom, a single statistical method for low-contrast detectability assessment.
- Rose, A. (1948). The sensitivity performance of the human eye on an absolute scale.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import io
from scipy.optimize import curve_fit


# ================================
# Core Statistical Method Functions
# ================================

def extract_central_roi(
    image: np.ndarray,
    roi_size: int = 1024
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extract a central square ROI from the image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    roi_size : int
        Size of the central square ROI (default 1024×1024)
        
    Returns:
    --------
    Tuple[np.ndarray, Tuple[int, int]]
        - Extracted central ROI
        - (center_row, center_col) coordinates
    """
    height, width = image.shape
    
    # Calculate center coordinates
    center_row = height // 2
    center_col = width // 2
    
    # Calculate ROI boundaries
    half_size = roi_size // 2
    start_row = max(0, center_row - half_size)
    end_row = min(height, center_row + half_size)
    start_col = max(0, center_col - half_size)
    end_col = min(width, center_col + half_size)
    
    # Extract central ROI
    central_roi = image[start_row:end_row, start_col:end_col]
    
    return central_roi, (center_row, center_col)


def extract_subrois_fixed_count(
    central_roi: np.ndarray,
    subroi_size: int,
    num_subrois: int,
    random_seed: int = 42
) -> List[np.ndarray]:
    """
    Extract a fixed number of subROIs of specific size from central ROI.
    
    Parameters:
    -----------
    central_roi : np.ndarray
        Central ROI array (e.g., 1024×1024)
    subroi_size : int
        Size of square subROI in pixels (e.g., 2, 4, 8, 16, etc.)
    num_subrois : int
        Number of subROIs to extract (e.g., 25, 36, 49, etc.)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[np.ndarray]
        List of subROI arrays (overlapping is allowed)
        
    Notes:
    ------
    SubROIs are extracted using grid-based sampling with slight randomization.
    Overlapping is permitted to achieve the desired count.
    """
    height, width = central_roi.shape
    subrois = []
    
    # Check if subROI size is valid
    if subroi_size > height or subroi_size > width:
        return subrois
    
    # Calculate grid size from number of subROIs (e.g., 49 -> 7x7)
    grid_size = int(np.sqrt(num_subrois))
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate spacing for grid pattern
    row_spacing = (height - subroi_size) / max(1, grid_size - 1) if grid_size > 1 else 0
    col_spacing = (width - subroi_size) / max(1, grid_size - 1) if grid_size > 1 else 0
    
    # Extract subROIs in approximate grid pattern
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_subrois:
                break
            
            # Calculate position with slight randomization
            if grid_size == 1:
                start_row = (height - subroi_size) // 2
                start_col = (width - subroi_size) // 2
            else:
                base_row = int(i * row_spacing)
                base_col = int(j * col_spacing)
                
                # Add small random offset (±10% of subroi_size)
                offset_range = max(1, subroi_size // 10)
                start_row = base_row + np.random.randint(-offset_range, offset_range + 1)
                start_col = base_col + np.random.randint(-offset_range, offset_range + 1)
            
            # Ensure we don't go out of bounds
            start_row = max(0, min(start_row, height - subroi_size))
            start_col = max(0, min(start_col, width - subroi_size))
            
            # Extract subROI
            subroi = central_roi[start_row:start_row + subroi_size, 
                                start_col:start_col + subroi_size]
            subrois.append(subroi)
            count += 1
        
        if count >= num_subrois:
            break
    
    return subrois


def calculate_ctsm(rois: List[np.ndarray]) -> Tuple[float, float, int]:
    """
    Calculate threshold contrast using the statistical method (SM).
    
    Parameters:
    -----------
    rois : List[np.ndarray]
        List of ROI arrays of identical size
        
    Returns:
    --------
    Tuple[float, float, int]
        - C_T: Threshold contrast at 95% confidence level
        - sigma_chi: Standard deviation of ROI means
        - n_rois: Number of ROIs used
        
    Formula:
    --------
    C_T(d) = 3.29 × σχ(d)
    
    where σχ is the standard deviation of the mean pixel values
    from n ROIs of size d×d pixels.
    
    References:
    -----------
    Chao et al. (2000), Paruccini et al. (2021)
    """
    # Calculate mean pixel value for each ROI
    roi_means = np.array([np.mean(roi) for roi in rois])
    
    # Calculate standard deviation of these means
    sigma_chi = np.std(roi_means, ddof=1)  # Using sample std deviation
    
    # Calculate threshold contrast at 95% confidence level
    # Factor 3.29 corresponds to ~95% confidence in normal distribution
    c_t = 3.29 * sigma_chi
    
    return c_t, sigma_chi, len(rois)


def extract_normalization_roi(
    central_roi: np.ndarray,
    norm_size: int = 120
) -> float:
    """
    Extract a central 120×120 pixel ROI for normalization.
    
    Parameters:
    -----------
    central_roi : np.ndarray
        Central ROI array
    norm_size : int
        Size of normalization ROI (default 120×120)
        
    Returns:
    --------
    float
        Mean pixel value of the normalization ROI
    """
    height, width = central_roi.shape
    center_row = height // 2
    center_col = width // 2
    half_size = norm_size // 2
    
    norm_roi = central_roi[
        center_row - half_size:center_row + half_size,
        center_col - half_size:center_col + half_size
    ]
    
    return np.mean(norm_roi)


def contrast_threshold_model(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Contrast threshold model: C_T(d) = c/d² + b/d + a
    
    Parameters:
    -----------
    d : np.ndarray
        Detail diameter in mm
    a, b, c : float
        Model parameters
        
    Returns:
    --------
    np.ndarray
        Threshold contrast values
    """
    return c / (d**2) + b / d + a


def fit_contrast_threshold_curve(
    diameters: np.ndarray,
    c_t_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Fit the contrast threshold curve: C_T(d) = c/d² + b/d + a
    
    Parameters:
    -----------
    diameters : np.ndarray
        Detail diameters in mm
    c_t_values : np.ndarray
        Threshold contrast values (normalized, in %)
        
    Returns:
    --------
    Tuple containing:
        - fitted_params: [a, b, c] parameters
        - fitted_curve: C_T values for smooth plotting
        - fit_quality: dict with R² and RMSE
    """
    try:
        # Initial guess for parameters
        p0 = [0.01, 0.1, 1.0]
        
        # Fit the curve
        popt, pcov = curve_fit(
            contrast_threshold_model,
            diameters,
            c_t_values,
            p0=p0,
            maxfev=10000
        )
        
        a, b, c = popt
        
        # Calculate fitted curve for plotting
        d_smooth = np.linspace(diameters.min(), diameters.max(), 100)
        fitted_curve = contrast_threshold_model(d_smooth, a, b, c)
        
        # Calculate R² and RMSE
        y_pred = contrast_threshold_model(diameters, a, b, c)
        ss_res = np.sum((c_t_values - y_pred)**2)
        ss_tot = np.sum((c_t_values - np.mean(c_t_values))**2)
        r_squared = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((c_t_values - y_pred)**2))
        
        fit_quality = {
            'r_squared': r_squared,
            'rmse': rmse,
            'd_smooth': d_smooth,
            'fitted_curve': fitted_curve
        }
        
        return popt, fit_quality
        
    except Exception as e:
        st.warning(f"Curve fitting failed: {str(e)}")
        return None, None


def generate_contrast_detail_curve(
    image: np.ndarray,
    subroi_sizes: List[int],
    num_subrois: int,
    pixel_spacing: float = 0.15,
    central_roi_size: int = 1024,
    norm_roi_size: int = 120
) -> Tuple[pd.DataFrame, float]:
    """
    Generate contrast-detail curve by extracting fixed number of subROIs for each size.
    
    Parameters:
    -----------
    image : np.ndarray
        Homogeneous phantom image
    subroi_sizes : List[int]
        List of subROI sizes in pixels (e.g., [2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128])
    num_subrois : int
        Number of subROIs to extract for each size (e.g., 25, 36, 49)
    pixel_spacing : float
        Detector pixel spacing in mm
    central_roi_size : int
        Size of central ROI in pixels (default 1024×1024)
    norm_roi_size : int
        Size of normalization ROI in pixels (default 120×120)
        
    Returns:
    --------
    Tuple[pd.DataFrame, float]
        - DataFrame with columns: subroi_size_pixels, n_subrois, diameter_mm,
          c_t, c_t_normalized, sigma_chi, sigma_chi_normalized
        - normalization_value: mean pixel value of 120×120 central ROI
    """
    # Extract central 1024×1024 ROI
    central_roi, _ = extract_central_roi(image, central_roi_size)
    
    # Extract normalization value from central 120×120 ROI
    normalization_value = extract_normalization_roi(central_roi, norm_roi_size)
    
    results = []
    
    for subroi_size in subroi_sizes:
        # Skip if subROI size is too large for central ROI
        if subroi_size > central_roi_size:
            st.warning(f"SubROI size {subroi_size}×{subroi_size} is larger than central ROI {central_roi_size}×{central_roi_size}. Skipping.")
            continue
        
        # Extract fixed number of subROIs of this size
        subrois = extract_subrois_fixed_count(central_roi, subroi_size, num_subrois)
        
        if len(subrois) < 25:
            st.warning(f"Could not extract {num_subrois} subROIs of size {subroi_size}×{subroi_size}. Only {len(subrois)} extracted. Skipping.")
            continue
        
        # Calculate C_T
        c_t, sigma_chi, n_subrois_used = calculate_ctsm(subrois)
        
        # Normalize by mean pixel value (convert to percentage)
        c_t_normalized = (c_t / normalization_value) * 100
        sigma_chi_normalized = (sigma_chi / normalization_value) * 100
        
        # Calculate diameter in mm
        diameter_mm = subroi_size * pixel_spacing
        
        results.append({
            'subroi_size_pixels': subroi_size,
            'n_subrois': n_subrois_used,
            'diameter_mm': diameter_mm,
            'c_t': c_t,
            'c_t_normalized': c_t_normalized,
            'sigma_chi': sigma_chi,
            'sigma_chi_normalized': sigma_chi_normalized
        })
    
    return pd.DataFrame(results), normalization_value


# ================================
# Visualization Functions
# ================================

def plot_contrast_detail_curve(
    df: pd.DataFrame,
    fit_params: np.ndarray = None,
    fit_quality: Dict = None
) -> plt.Figure:
    """
    Plot the contrast-detail curve with fitted model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe from generate_contrast_detail_curve
    fit_params : np.ndarray
        Fitted parameters [a, b, c] from curve fitting
    fit_quality : Dict
        Dictionary containing fit quality metrics and smooth curve
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot measured data points
    ax.plot(df['diameter_mm'], df['c_t_normalized'], 'bo', markersize=10, 
            label='Measured $C_T$', zorder=3)
    
    # Plot fitted curve if available
    if fit_params is not None and fit_quality is not None:
        a, b, c = fit_params
        ax.plot(fit_quality['d_smooth'], fit_quality['fitted_curve'], 'r-', 
                linewidth=2.5, label=rf'Fitted: $C_T(d) = {c:.4f}/d^2 + {b:.4f}/d + {a:.4f}$',
                zorder=2)
        
        # Add fit quality text
        textstr = f"R² = {fit_quality['r_squared']:.4f}\nRMSE = {fit_quality['rmse']:.4f}%"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    # Formatting
    ax.set_xlabel('Detail Diameter d (mm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Threshold Contrast $C_T$ (%)', fontsize=13, fontweight='bold')
    ax.set_title('Contrast-Detail Detectability Curve\n(Statistical Method - Normalized to 120×120 Central ROI)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    return fig


def visualize_roi_sampling(
    image: np.ndarray,
    subroi_size: int,
    num_subrois: int,
    central_roi_size: int = 1024,
    norm_roi_size: int = 120
) -> plt.Figure:
    """
    Visualize the central ROI and sampled subROI positions.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    subroi_size : int
        Size of subROIs in pixels
    num_subrois : int
        Number of subROIs to sample
    central_roi_size : int
        Size of central ROI in pixels
    norm_roi_size : int
        Size of normalization ROI in pixels
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure showing ROI locations
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display full image
    ax.imshow(image, cmap='gray')
    
    height, width = image.shape
    center_row = height // 2
    center_col = width // 2
    
    # Draw central 1024×1024 ROI
    half_central = central_roi_size // 2
    central_rect = plt.Rectangle(
        (center_col - half_central, center_row - half_central),
        central_roi_size, central_roi_size,
        edgecolor='blue', facecolor='none', linewidth=3,
        label=f'Central ROI ({central_roi_size}×{central_roi_size})'
    )
    ax.add_patch(central_rect)
    
    # Draw 120×120 normalization ROI
    half_norm = norm_roi_size // 2
    norm_rect = plt.Rectangle(
        (center_col - half_norm, center_row - half_norm),
        norm_roi_size, norm_roi_size,
        edgecolor='green', facecolor='none', linewidth=2.5,
        label=f'Normalization ROI ({norm_roi_size}×{norm_roi_size})'
    )
    ax.add_patch(norm_rect)
    
    # Extract central ROI to get subROI positions
    central_roi, _ = extract_central_roi(image, central_roi_size)
    subrois_coords = extract_subrois_fixed_count(central_roi, subroi_size, num_subrois)
    
    # Calculate grid for visualization
    grid_size = int(np.sqrt(num_subrois))
    row_spacing = (central_roi_size - subroi_size) / max(1, grid_size - 1) if grid_size > 1 else 0
    col_spacing = (central_roi_size - subroi_size) / max(1, grid_size - 1) if grid_size > 1 else 0
    
    start_row = center_row - half_central
    start_col = center_col - half_central
    
    np.random.seed(42)  # Match the extraction seed
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_subrois:
                break
            
            if grid_size == 1:
                base_row = (central_roi_size - subroi_size) // 2
                base_col = (central_roi_size - subroi_size) // 2
            else:
                base_row = int(i * row_spacing)
                base_col = int(j * col_spacing)
                
                offset_range = max(1, subroi_size // 10)
                base_row += np.random.randint(-offset_range, offset_range + 1)
                base_col += np.random.randint(-offset_range, offset_range + 1)
            
            base_row = max(0, min(base_row, central_roi_size - subroi_size))
            base_col = max(0, min(base_col, central_roi_size - subroi_size))
            
            subroi_rect = plt.Rectangle(
                (start_col + base_col, start_row + base_row),
                subroi_size, subroi_size,
                edgecolor='red', facecolor='none', linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(subroi_rect)
            count += 1
        
        if count >= num_subrois:
            break
    
    # Add label for subROIs (only once)
    subroi_rect_label = plt.Rectangle(
        (0, 0), 1, 1, edgecolor='red', facecolor='none', linewidth=1.5,
        label=f'{num_subrois} subROIs of size {subroi_size}×{subroi_size}px'
    )
    ax.add_patch(subroi_rect_label)
    
    ax.set_title(f'ROI Sampling Pattern\n{num_subrois} subROIs of {subroi_size}×{subroi_size} pixels from central {central_roi_size}×{central_roi_size} ROI', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ================================
# Streamlit Interface
# ================================

def display_threshold_contrast_section(image_array: np.ndarray, pixel_spacing_row: float, pixel_spacing_col: float):
    """
    Renders the Threshold Contrast Detail Detectability section using Statistical Method.
    
    Parameters:
    -----------
    image_array : np.ndarray
        Input image array (pixel values or kerma if detector conversion available)
    pixel_spacing_row : float
        Pixel spacing in row direction (mm)
    pixel_spacing_col : float
        Pixel spacing in column direction (mm)
    """
    st.header("Threshold Contrast Detail Detectability (TCDD)")
    st.subheader("Statistical Method (Chao et al., 2000)")
    
    # Information about the method
    with st.expander("ℹ️ About the Statistical Method", expanded=False):
        st.markdown("""
        **Method Overview:**
        
        This implementation follows the statistical method for low-contrast detectability assessment:
        
        1. **Extract Central ROI**: 1024×1024 pixel region from image center
        2. **For each subROI size**: Extract fixed number of subROIs (e.g., 49 subROIs of 2×2, then 49 of 3×3, etc.)
        3. **Calculate $$C_{T}$$**: For each size: $$C_{T}(d) = 3.29 × σ(d)$$
        4. **Normalize**: Convert to percentage using mean of central 120×120 ROI
        5. **Fit Curve**: $$C_{T}(d) = \\frac{c}{d^2} + \\frac{b}{d} + a$$
        
        **Key Parameters:**
        - Central ROI: 1024×1024 pixels
        - Normalization ROI: 120×120 pixels (center)
        - Number of subROIs: 25, 36, 49, 64, 81, 100, 121, 144, 169, or 196 (default: 49)
        - SubROI sizes: 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128 pixels
        - Overlapping allowed
        - Confidence level: 95% (factor 3.29)
        
        **References:**
        - Chao, E.H., et al. (2000). Radiology, 217(1), 162-162
        - Paruccini, N., et al. (2021). Physica Medica, 91, 105-112
        - Rose, A. (1948). J. Opt. Soc. Am., 38(2), 196-208
        """)
    
    st.markdown("---")
    
    # Average pixel spacing
    pixel_spacing = (pixel_spacing_row + pixel_spacing_col) / 2
    
    # Check if image is provided
    if image_array is not None:
        # Apply detector conversion if available (convert pixel values to kerma)
        detector_conversion_fn = st.session_state.get('detector_conversion', None)
        if isinstance(detector_conversion_fn, dict) and isinstance(detector_conversion_fn.get('fit'), dict):
            detector_conversion_fn = detector_conversion_fn.get('fit', {})
        
        is_kerma_image = False
        if detector_conversion_fn is not None:
            st.info("✓ Using detector conversion function to convert pixel values to Kerma (μGy)")
            try:
                # Apply the detector conversion function
                predict_mpv = detector_conversion_fn.get('predict_mpv')
                if predict_mpv is not None:
                    image = predict_mpv(image_array.flatten()).reshape(image_array.shape)
                    is_kerma_image = True
                    st.success(f"Image converted to Kerma values")
                else:
                    st.warning("Detector conversion function not properly loaded. Using raw pixel values.")
                    image = image_array.astype(np.float64)
            except Exception as e:
                st.error(f"Error applying detector conversion: {str(e)}")
                image = image_array.astype(np.float64)
        else:
            st.info("No detector conversion function available. Using raw pixel values.")
            image = image_array.astype(np.float64)
        
        # Process image
        try:
            
            st.success(f"✓ Image loaded: {image.shape[0]} × {image.shape[1]} pixels")
            if is_kerma_image:
                st.caption("Working with Kerma image (μGy). All values will be in Kerma units.")
            else:
                st.caption("Working with raw pixel values.")
            
            st.markdown("---")
            
            # Configuration
            st.subheader("1. Configure Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                central_roi_size = st.number_input(
                    "Central ROI size (pixels)",
                    min_value=512,
                    max_value=2048,
                    value=1024,
                    step=128,
                    help="Size of the central square ROI (default 1024×1024)"
                )
                
                norm_roi_size = st.number_input(
                    "Normalization ROI size (pixels)",
                    min_value=60,
                    max_value=240,
                    value=120,
                    step=10,
                    help="Size of central normalization ROI for converting to percentage (default 120×120)"
                )
                
                # Number of subROIs selection
                st.write("**Number of SubROIs:**")
                num_subrois_options = [25, 36, 49, 64, 81, 100, 121, 144, 169, 196]
                num_subrois = st.selectbox(
                    "Select count (N²)",
                    options=num_subrois_options,
                    index=2,  # Default to 49
                    help="Number of subROIs to extract for each size (perfect squares: 5², 6², 7², etc.)"
                )
                grid_equiv = int(np.sqrt(num_subrois))
                st.caption(f"Extracts {grid_equiv}×{grid_equiv} = {num_subrois} subROIs for each size")
            
            with col2:
                # SubROI sizes selection
                st.write("**SubROI Sizes to Evaluate (pixels):**")
                
                # Default sizes: 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128
                default_sizes = [2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128]
                subroi_sizes_str = st.text_input(
                    "Enter sizes (comma-separated)",
                    value=", ".join(map(str, default_sizes)),
                    help="SubROI sizes in pixels (minimum 2×2). {num_subrois} subROIs will be extracted for each size."
                )
                
                try:
                    subroi_sizes = [int(s.strip()) for s in subroi_sizes_str.split(',')]
                    subroi_sizes = [s for s in subroi_sizes if s >= 2]  # Minimum 2x2
                    subroi_sizes.sort()
                    
                    # Show physical diameters
                    physical_sizes = [f"{s*pixel_spacing:.2f}" for s in subroi_sizes]
                    st.caption(f"Diameters (mm): {', '.join(physical_sizes)}")
                    st.caption(f"Total measurements: {len(subroi_sizes)} sizes × {num_subrois} subROIs each")
                    
                except:
                    st.error("Invalid subROI sizes. Use comma-separated integers ≥2 (e.g., 2, 3, 4, 5, 6, 7, 8)")
                    subroi_sizes = default_sizes
            
            # Visualize ROI sampling
            if st.checkbox("Preview ROI sampling pattern", value=False):
                preview_size = st.selectbox("Select subROI size for preview", subroi_sizes, 
                                           index=len(subroi_sizes)//2 if subroi_sizes else 0)
                fig_preview = visualize_roi_sampling(image, preview_size, num_subrois, central_roi_size, norm_roi_size)
                st.pyplot(fig_preview)
                plt.close()
            
            st.markdown("---")
            
            # Calculate button
            st.subheader("2. Calculate Contrast-Detail Curve")
            
            if st.button("🔬 Calculate $$C_{T}(d)$$ & Fit Curve", type="primary"):
                with st.spinner("Extracting central ROI, calculating threshold contrast, and fitting curve..."):
                    
                    # Display image info for debugging
                    st.info(f"📊 Image shape: {image.shape}, Value range: [{image.min():.2f}, {image.max():.2f}]" + 
                            (" μGy" if is_kerma_image else " (pixel values)"))
                    
                    # Generate contrast-detail curve
                    df_results, normalization_value = generate_contrast_detail_curve(
                        image, subroi_sizes, num_subrois, pixel_spacing, central_roi_size, norm_roi_size
                    )
                    
                    if df_results.empty:
                        st.error("Could not generate results. Check grid sizes and image dimensions.")
                    else:
                        # Fit the curve
                        st.info("Fitting contrast threshold model: $$C_{T}(d) = \\frac{c}{d^2} + \\frac{b}{d} + a$$")
                        fit_params, fit_quality = fit_contrast_threshold_curve(
                            df_results['diameter_mm'].values,
                            df_results['c_t_normalized'].values
                        )
                        
                        units = "μGy" if is_kerma_image else "PV"
                        
                        st.success(f"✓ Analysis complete for {len(df_results)} subROI sizes")
                        
                        if fit_params is not None:
                            a, b, c = fit_params
                            st.success(f"✓ Curve fitted successfully: $$C_{{T}}(d) = \\frac{{{c:.4f}}}{{d^2}} + \\frac{{{b:.4f}}}{{d}} + {a:.4f}$$")
                            st.metric("R² (goodness of fit)", f"{fit_quality['r_squared']:.4f}")
                        
                        # Display normalization info
                        st.info(f"**Normalization value** (mean of central {norm_roi_size}×{norm_roi_size} ROI): {normalization_value:.2f} {units}")
                        
                        st.markdown("---")
                        
                        # Display results table
                        st.subheader("Results Table")
                        
                        # Format table for display
                        display_df = df_results.copy()
                        display_df['diameter_mm'] = display_df['diameter_mm'].round(3)
                        display_df['c_t'] = display_df['c_t'].round(2)
                        display_df['c_t_normalized'] = display_df['c_t_normalized'].round(4)
                        display_df['sigma_chi'] = display_df['sigma_chi'].round(2)
                        display_df['sigma_chi_normalized'] = display_df['sigma_chi_normalized'].round(4)
                        
                        # Create markdown table with LaTeX headers
                        markdown_table = f"""
| **SubROI Size (px)** | **# SubROIs** | **Diameter $d$ (mm)** | **$C_{{T}}$ (random units)** | **$C_{{T}}$ (%) | **$\\sigma$ (random units)** | **$\\sigma$ (%)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
                        
                        for _, row in display_df.iterrows():
                            markdown_table += f"| {row['subroi_size_pixels']} | {row['n_subrois']} | {row['diameter_mm']:.3f} | {row['c_t']:.2f} | {row['c_t_normalized']:.4f} | {row['sigma_chi']:.2f} | {row['sigma_chi_normalized']:.4f} |\n"
                        
                        st.markdown(markdown_table)
                        
                        # Add fitted values to download
                        if fit_params is not None:
                            a, b, c = fit_params
                            df_export = df_results.copy()
                            df_export['c_t_fitted'] = contrast_threshold_model(
                                df_export['diameter_mm'].values, a, b, c
                            )
                            df_export['residual'] = df_export['c_t_normalized'] - df_export['c_t_fitted']
                        else:
                            df_export = df_results.copy()
                        
                        # Download CSV
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="tcdd_statistical_method_results.csv",
                            mime="text/csv"
                        )
                        
                        st.markdown("---")
                        
                        # Plot contrast-detail curve
                        st.subheader("Contrast-Detail Curve with Fitted Model")
                        fig_curve = plot_contrast_detail_curve(df_results, fit_params, fit_quality)
                        st.pyplot(fig_curve)
                        
                        # Download plot
                        buf = io.BytesIO()
                        fig_curve.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Plot (PNG)",
                            data=buf,
                            file_name="contrast_detail_curve.png",
                            mime="image/png"
                        )
                        plt.close()
                        
                        st.markdown("---")
                        
                        # Display fitted parameters
                        if fit_params is not None:
                            st.subheader("Fitted Model Parameters")
                            col_p1, col_p2, col_p3 = st.columns(3)
                            
                            with col_p1:
                                st.metric("Parameter a", f"{a:.6f}")
                            with col_p2:
                                st.metric("Parameter b", f"{b:.6f}")
                            with col_p3:
                                st.metric("Parameter c", f"{c:.6f}")
                            
                            st.latex(r"C_{th}(d) = \frac{" + f"{c:.4f}" + r"}{d^2} + \frac{" + f"{b:.4f}" + r"}{d} + " + f"{a:.4f}")
                            
                            st.markdown("---")
                            
                            # Calculate and display contrast at specific detail sizes
                            st.subheader("Threshold Contrast at Standard Detail Sizes")
                            st.caption("Calculated from fitted curve model")
                            
                            # Calculate C_T at 0.5 mm and 2 mm using fitted model
                            c_t_05mm = contrast_threshold_model(np.array([0.5]), a, b, c)[0]
                            c_t_2mm = contrast_threshold_model(np.array([2.0]), a, b, c)[0]
                            
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.metric(
                                    "$$C_T$$ at 0.5 mm", 
                                    f"{c_t_05mm:.4f} %",
                                    help="Threshold contrast for 0.5 mm detail diameter (from fitted curve)"
                                )
                            
                            with col_d2:
                                st.metric(
                                    "$$C_T$$ at 2.0 mm", 
                                    f"{c_t_2mm:.4f} %",
                                    help="Threshold contrast for 2 mm detail diameter (from fitted curve)"
                                )
                        
                        st.markdown("---")
                
                # Statistical summary
                st.subheader("Statistical Summary")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Smallest Detail", f"{df_results['diameter_mm'].min():.3f} mm")
                with col_b:
                    st.metric("Largest Detail", f"{df_results['diameter_mm'].max():.3f} mm")
                with col_c:
                    st.metric("Min $C_T$", f"{df_results['c_t_normalized'].min():.4f} %")
                with col_d:
                    st.metric("Max $C_T$", f"{df_results['c_t_normalized'].max():.4f} %")
                
                # Interpretation
                st.info("""
                **Interpretation:**
                
                - **$C_T$ (%)**: Threshold contrast normalized to mean pixel value (lower = better detectability)
                - **Fitted curve**: $C_{T}(d) = \\frac{c}{d^2} + \\frac{b}{d} + a$ describes contrast-detail relationship
                - **Larger detail sizes** typically have lower threshold contrast (easier to detect)
                - **R²**: Goodness of fit (closer to 1 = better fit)
                - Values represent minimum contrast difference at 95% confidence level
                - Compare with baseline values to assess system performance changes
                """)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("👆 Please upload a homogeneous phantom image using the sidebar uploader")
        
        # Show example of expected workflow
        with st.expander("📖 Usage Guide", expanded=False):
            st.markdown("""
            **Step-by-step workflow:**
            
            1. **Acquire Image**: Capture a uniform phantom image (no test details)
                - Use RAW/STD/'For Processing' format
                - Same acquisition as uniformity test
                - Ensure good statistics (proper exposure)
                - Image should be at least 1024×1024 pixels or larger
            
            2. **Upload**: Load the image using the sidebar uploader
            
            3. **Configure**:
                - Set central ROI size (default 1024×1024 pixels)
                - Set normalization ROI size (default 120×120 pixels)
                - Select number of subROIs (25, 36, 49, 64, etc.) - default 49
                - Enter subROI sizes to test (2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128 pixels)
            
            4. **Preview** (optional): Check ROI sampling pattern for a specific size
            
            5. **Analyze**: Click "Calculate C_T & Fit Curve"
            
            6. **Review Results**:
                - Examine fitted curve: $$C_{T}(d) = c/d² + b/d + a$$
                - Check R² value for fit quality
                - Review normalized C_T values (in %)
                - Compare with baseline or reference values
                - Download results (CSV) and plot (PNG)
            
            **Key Concepts:**
            - **Central 1024×1024 ROI**: Main analysis region from image center
            - **120×120 Normalization ROI**: Used to convert C_T to percentage
            - **Fixed subROI count**: Same number extracted for each size (e.g., 49)
            - **Multiple sizes tested**: 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128 pixels
            - **Overlapping allowed**: SubROIs may overlap to achieve desired count
            - **Detail diameter d**: Physical size of subROI in mm
            - **C_{T}(d)**: Threshold contrast as function of detail size
            
            **Tips:**
            - Use uniform phantom (same as uniformity analysis)
            - Ensure central region is artifact-free
            - Default 49 subROIs (7×7 pattern) is recommended
            - Minimum 25 subROIs required for statistical reliability
            - Smaller subROI sizes → smaller detail diameters → higher C_{T} values
            - Good fit (R² > 0.95) validates the model
            """)


