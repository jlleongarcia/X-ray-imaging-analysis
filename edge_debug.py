"""
Edge Detection Debugging Tool for MTF Analysis

This module helps diagnose edge detection orientation issues by providing
detailed visualization and analysis of edge detection algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import streamlit as st
import pandas as pd


def debug_edge_detection(edge_roi_array, pixel_spacing, show_plots=True):
    """
    Debug edge detection to understand orientation issues.
    
    Parameters
    ----------
    edge_roi_array : np.ndarray
        2D array containing the edge ROI
    pixel_spacing : float
        Pixel spacing in mm
    show_plots : bool
        Whether to display debug plots
        
    Returns
    -------
    dict
        Debug information about edge detection
    """
    
    debug_info = {}
    
    # Normalize edge data
    edge_normalized = (edge_roi_array - edge_roi_array.min()) / (edge_roi_array.max() - edge_roi_array.min())
    
    # Calculate gradients using Sobel
    sobel_h = sobel(edge_normalized, axis=0)  # horizontal gradients (d/dx)
    sobel_v = sobel(edge_normalized, axis=1)  # vertical gradients (d/dy)
    
    # Gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    gradient_angle = np.arctan2(sobel_v, sobel_h)  # angle of gradient vector
    
    # Find edge points (high gradient magnitude)
    edge_threshold = np.percentile(gradient_magnitude, 90)
    edge_points = gradient_magnitude > edge_threshold
    
    # Get coordinates of edge points
    y_coords, x_coords = np.where(edge_points)
    
    debug_info['num_edge_points'] = len(y_coords)
    debug_info['gradient_threshold'] = edge_threshold
    debug_info['max_gradient'] = np.max(gradient_magnitude)
    
    if len(y_coords) < 10:
        debug_info['error'] = "Insufficient edge points detected"
        return debug_info
    
    # Method 1: Simple line fit (for comparison, but often poor quality)
    coeffs = np.polyfit(x_coords, y_coords, 1)
    slope_linefit = coeffs[0]
    intercept = coeffs[1]
    angle_linefit_rad = np.arctan(slope_linefit)
    angle_linefit_deg = np.degrees(angle_linefit_rad)
    is_vertical_linefit = abs(angle_linefit_rad) > np.pi / 4
    
    # Calculate line fit quality
    y_pred = slope_linefit * x_coords + intercept
    ss_res = np.sum((y_coords - y_pred) ** 2)
    ss_tot = np.sum((y_coords - np.mean(y_coords)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Method 2: Principal Component Analysis (PCA) - most robust approach
    # Center the edge coordinates (critical step!)
    edge_coords_centered = np.column_stack([
        x_coords - np.mean(x_coords), 
        y_coords - np.mean(y_coords)
    ])
    
    # Compute covariance matrix of the centered coordinates
    cov_matrix = np.cov(edge_coords_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Principal direction is the eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvals)
    principal_direction = eigenvecs[:, principal_idx]
    
    # Calculate angle of principal direction
    angle_pca_rad = np.arctan2(principal_direction[1], principal_direction[0])
    angle_pca_deg = np.degrees(angle_pca_rad)
    is_vertical_pca = abs(angle_pca_rad) > np.pi / 4
    
    # PCA confidence: ratio of largest to smallest eigenvalue
    pca_confidence = eigenvals[principal_idx] / (eigenvals[1 - principal_idx] + 1e-9)
    
    # Method 3: Local gradient direction (only at edge points, not whole image)
    # Weight gradients by magnitude for better accuracy, but ONLY at detected edge points
    edge_grad_h = sobel_h[edge_points]
    edge_grad_v = sobel_v[edge_points]
    edge_grad_mag = gradient_magnitude[edge_points]
    
    # Weighted average of gradients only at edge locations
    weighted_grad_h = np.sum(edge_grad_h * edge_grad_mag) / np.sum(edge_grad_mag)
    weighted_grad_v = np.sum(edge_grad_v * edge_grad_mag) / np.sum(edge_grad_mag)
    grad_angle_weighted = np.arctan2(weighted_grad_v, weighted_grad_h)
    
    # Edge is perpendicular to gradient
    edge_angle_grad_rad = grad_angle_weighted - np.pi/2
    # Normalize to [-π/2, π/2]
    if edge_angle_grad_rad > np.pi/2:
        edge_angle_grad_rad -= np.pi
    elif edge_angle_grad_rad < -np.pi/2:
        edge_angle_grad_rad += np.pi
    edge_angle_grad_deg = np.degrees(edge_angle_grad_rad)
    is_vertical_grad = abs(edge_angle_grad_rad) > np.pi/4
    
    # Method 4: Coordinate span analysis (geometric approach)
    x_span = np.max(x_coords) - np.min(x_coords)
    y_span = np.max(y_coords) - np.min(y_coords)
    is_vertical_span = y_span > x_span  # vertical edge spans more in y direction
    span_ratio = y_span / (x_span + 1e-9)
    
    # Method 5: Coordinate variance analysis (statistical approach)
    x_var = np.var(x_coords)
    y_var = np.var(y_coords)
    is_vertical_var = x_var < y_var  # vertical edge has less variance in x
    variance_ratio = y_var / (x_var + 1e-9)
    
    # Method 7: Robust angle estimation using median of local gradients
    # Sample local gradients at multiple points along the edge
    local_angles = []  # Initialize outside the conditional block
    
    if len(y_coords) > 20:
        # Sample evenly spaced points
        n_samples = min(20, len(y_coords) // 2)
        indices = np.linspace(0, len(y_coords) - 1, n_samples, dtype=int)
        
        for idx in indices:
            y_center, x_center = y_coords[idx], x_coords[idx]
            # Get local window around this point
            window_size = 3
            y_min = max(0, y_center - window_size)
            y_max = min(gradient_angle.shape[0], y_center + window_size + 1)
            x_min = max(0, x_center - window_size)
            x_max = min(gradient_angle.shape[1], x_center + window_size + 1)
            
            local_grad_angle = gradient_angle[y_min:y_max, x_min:x_max]
            local_grad_mag = gradient_magnitude[y_min:y_max, x_min:x_max]
            
            # Weighted average of gradient angles in local window
            if np.sum(local_grad_mag) > 0:
                avg_grad_angle = np.average(local_grad_angle, weights=local_grad_mag)
                # Edge perpendicular to gradient
                edge_angle = avg_grad_angle - np.pi/2
                # Normalize
                if edge_angle > np.pi/2:
                    edge_angle -= np.pi
                elif edge_angle < -np.pi/2:
                    edge_angle += np.pi
                local_angles.append(edge_angle)
    
    # Calculate robust angle regardless of whether we have local samples
    if local_angles:
        # Use median for robustness
        angle_robust_rad = np.median(local_angles)
        angle_robust_deg = np.degrees(angle_robust_rad)
        is_vertical_robust = abs(angle_robust_rad) > np.pi/4
    else:
        # Fall back to gradient method for small edge regions
        angle_robust_rad = edge_angle_grad_rad
        angle_robust_deg = edge_angle_grad_deg
        is_vertical_robust = is_vertical_grad
    
    debug_info.update({
        # Line fitting results (often poor quality)
        'slope_linefit': slope_linefit,
        'intercept': intercept,
        'angle_linefit_rad': angle_linefit_rad,
        'angle_linefit_deg': angle_linefit_deg,
        'is_vertical_linefit': is_vertical_linefit,
        'line_fit_r_squared': r_squared,
        
        # PCA results (most robust method)
        'angle_pca_rad': angle_pca_rad,
        'angle_pca_deg': angle_pca_deg,
        'is_vertical_pca': is_vertical_pca,
        'pca_confidence': pca_confidence,
        
        # Gradient-based results (edge points only)
        'angle_gradient_rad': edge_angle_grad_rad,
        'angle_gradient_deg': edge_angle_grad_deg,
        'is_vertical_gradient': is_vertical_grad,
        
        # Geometric analysis (simple but effective)
        'is_vertical_span': is_vertical_span,
        'is_vertical_variance': is_vertical_var,
        'x_span': x_span,
        'y_span': y_span,
        'x_variance': x_var,
        'y_variance': y_var,
        'span_ratio': span_ratio,
        'variance_ratio': variance_ratio,
    })
    
    # Smart consensus: PCA is most reliable, use it as primary method
    if pca_confidence > 2.0:  # High confidence in PCA
        consensus_orientation = is_vertical_pca
        consensus_angle_deg = angle_pca_deg
        consensus_angle_rad = angle_pca_rad
        best_method = f"PCA (confidence={pca_confidence:.1f})"
    else:
        # If PCA has low confidence, use voting from geometric methods
        geometric_votes = [is_vertical_span, is_vertical_var, is_vertical_grad]
        consensus_orientation = sum(geometric_votes) >= 2  # Majority vote
        
        # For angle, prefer gradient method over line fitting
        if len(x_coords) > 20:
            consensus_angle_deg = edge_angle_grad_deg
            consensus_angle_rad = edge_angle_grad_rad
            best_method = "Gradient (edge points)"
        else:
            consensus_angle_deg = angle_pca_deg
            consensus_angle_rad = angle_pca_rad
            best_method = f"PCA (low confidence={pca_confidence:.1f})"
    
    debug_info.update({
        'is_vertical_consensus': consensus_orientation,
        'consensus_angle_deg': consensus_angle_deg,
        'consensus_angle_rad': consensus_angle_rad,
        'best_angle_method': best_method,
    })
    
    if show_plots:
        # Create debug plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(edge_normalized, cmap='gray')
        axes[0, 0].set_title(f'Original Edge ROI\n{edge_roi_array.shape[0]}×{edge_roi_array.shape[1]}')
        axes[0, 0].axis('off')
        
        # Gradient magnitude
        axes[0, 1].imshow(gradient_magnitude, cmap='viridis')
        axes[0, 1].set_title(f'Gradient Magnitude\nMax: {np.max(gradient_magnitude):.3f}')
        axes[0, 1].axis('off')
        
        # Edge points with fitted line (keep it simple and working)
        axes[0, 2].imshow(edge_normalized, cmap='gray', alpha=0.7)
        axes[0, 2].scatter(x_coords, y_coords, c='red', s=1, alpha=0.5, label='Edge points')
        
        # Show line fit only if it exists and has reasonable quality
        if r_squared > 0.3:
            x_line = np.array([np.min(x_coords), np.max(x_coords)])
            y_line_fit = slope_linefit * x_line + intercept
            axes[0, 2].plot(x_line, y_line_fit, 'yellow', linewidth=2,
                           label=f'Line fit (R²={r_squared:.2f})')
        
        axes[0, 2].set_title(f'Edge Detection\nFound {len(x_coords)} edge points')
        axes[0, 2].legend()
        
        # Gradient direction (subsampled for clarity)
        subsample = max(1, min(edge_normalized.shape) // 20)  # Adaptive subsampling
        y_sample, x_sample = np.mgrid[0:gradient_angle.shape[0]:subsample, 0:gradient_angle.shape[1]:subsample]
        u = np.cos(gradient_angle[::subsample, ::subsample])
        v = np.sin(gradient_angle[::subsample, ::subsample])
        axes[1, 0].imshow(edge_normalized, cmap='gray', alpha=0.5)
        axes[1, 0].quiver(x_sample, y_sample, u, v, gradient_magnitude[::subsample, ::subsample], 
                         cmap='viridis', scale=20, width=0.003)
        axes[1, 0].set_title('Gradient Direction\n(arrows point in gradient direction)')
        
        # Horizontal and vertical gradients
        axes[1, 1].imshow(sobel_h, cmap='RdBu_r')
        axes[1, 1].set_title('Horizontal Gradients (d/dx)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(sobel_v, cmap='RdBu_r')
        axes[1, 2].set_title('Vertical Gradients (d/dy)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Simplified summary table with focus on working methods
        summary_data = {
            'Method': [
                '🎯 PCA Analysis',
                'Gradient (edge only)',
                'Coordinate Span',
                'Coordinate Variance',
                '✅ CONSENSUS'
            ],
            'Is Vertical': [
                is_vertical_pca,
                is_vertical_grad,
                is_vertical_span,
                is_vertical_var,
                consensus_orientation
            ],
            'Angle/Details': [
                f'{angle_pca_deg:.1f}° (conf={pca_confidence:.1f})',
                f'{edge_angle_grad_deg:.1f}°',
                f'Y/X ratio = {span_ratio:.2f}',
                f'Var ratio = {variance_ratio:.2f}',
                f'{consensus_angle_deg:.1f}° ({best_method})'
            ]
        }
        
        st.write("**Edge Orientation Analysis:**")
        st.dataframe(pd.DataFrame(summary_data))
        
        # Show PCA details (the star of the show!)
        st.write("**🎯 PCA Analysis Details:**")
        st.info(f"""
        **How PCA Works for Edge Detection:**
        
        1. **Edge Points**: Found {len(x_coords)} points along the edge
        2. **Centering**: Subtract mean position to center the data cloud
        3. **Covariance**: Calculate how x and y coordinates vary together
        4. **Principal Direction**: Find direction of maximum variance (eigenvalue = {eigenvals[principal_idx]:.1f})
        5. **Confidence**: Ratio of largest/smallest eigenvalue = {pca_confidence:.1f}
        
        **Why PCA is Superior:**
        - ✅ Robust to noise and outliers
        - ✅ No coordinate system bias
        - ✅ Mathematically optimal direction
        - ✅ Provides confidence measure
        
        **Result**: Edge angle = {angle_pca_deg:.1f}°, Orientation = {'Vertical' if is_vertical_pca else 'Horizontal'}
        """)
        
        # Show key results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Edge Points Found", len(x_coords))
            st.metric("PCA Confidence", f"{pca_confidence:.1f}")
        with col2:
            st.metric("PCA Angle", f"{angle_pca_deg:.1f}°")
            st.metric("PCA Orientation", "Vertical" if is_vertical_pca else "Horizontal")
        with col3:
            st.metric("Final Angle", f"{consensus_angle_deg:.1f}°")
            st.metric("Final Method", best_method)
        
        # Quality assessment
        if len(x_coords) < 30:
            st.warning(f"⚠️ Few edge points detected ({len(x_coords)}). Try adjusting ROI selection.")
        elif len(x_coords) > 1000:
            st.warning(f"⚠️ Many edge points detected ({len(x_coords)}). Edge may be too broad or noisy.")
        else:
            st.success(f"✅ Good number of edge points detected ({len(x_coords)}).")
        
        if abs(consensus_angle_deg) < 2 or abs(consensus_angle_deg) > 8:
            st.warning(f"⚠️ Edge angle ({abs(consensus_angle_deg):.1f}°) outside optimal range (3-5°).")
        else:
            st.success(f"✅ Edge angle ({abs(consensus_angle_deg):.1f}°) is within recommended range.")
    
    
    
    return debug_info


def analyze_edge_simple(edge_roi_array, pixel_spacing):
    """
    Simple edge analysis without requiring EdgeMTF class.
    
    Parameters
    ----------
    edge_roi_array : np.ndarray
        Edge ROI array
    pixel_spacing : float
        Pixel spacing in mm
        
    Returns
    -------
    dict
        Simple analysis results using robust methods
    """
    try:
        # Perform robust edge detection analysis
        debug_info = debug_edge_detection(edge_roi_array, pixel_spacing, show_plots=False)
        
        results = {
            # Use robust/consensus results instead of line fitting
            'edge_angle_rad': debug_info.get('consensus_angle_rad', 0),
            'edge_angle_deg': debug_info.get('consensus_angle_deg', 0),
            'is_vertical': debug_info.get('is_vertical_consensus', False),
            'line_fit_quality': debug_info.get('line_fit_r_squared', 0),
            'num_edge_points': debug_info.get('num_edge_points', 0),
            'best_method': debug_info.get('best_angle_method', 'Unknown'),
            'robust_angle_deg': debug_info.get('angle_robust_deg', 0),
            'pca_angle_deg': debug_info.get('angle_pca_deg', 0),
            'gradient_angle_deg': debug_info.get('angle_gradient_deg', 0),
            'success': True
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# For backwards compatibility, alias the analyze_edge_with_pylinac function
analyze_edge_with_pylinac = analyze_edge_simple