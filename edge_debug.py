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


def debug_edge_detection(edge_roi_array, show_plots=True):
    """
    Debug edge detection to understand orientation issues.
    
    Parameters
    ----------
    edge_roi_array : np.ndarray
        2D array containing the edge ROI
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
    
    # Principal Component Analysis (PCA) - the optimal method
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
    
    # Work with absolute values - we don't care about sign
    angle_pca_deg_abs = abs(angle_pca_deg)
    angle_pca_rad_abs = abs(angle_pca_rad)
    
    # Determine orientation based on absolute angle
    is_vertical_pca = angle_pca_deg_abs > 45
    
    # PCA confidence: ratio of largest to smallest eigenvalue
    pca_confidence = eigenvals[principal_idx] / (eigenvals[1 - principal_idx] + 1e-9)
    
    # Store PCA results (using absolute values)
    debug_info.update({
        'angle_pca_rad': angle_pca_rad_abs,
        'angle_pca_deg': angle_pca_deg_abs,
        'is_vertical_pca': is_vertical_pca,
        'pca_confidence': pca_confidence,
        'edge_angle_rad': angle_pca_rad_abs,  # Use PCA as final result
        'edge_angle_deg': angle_pca_deg_abs,
        'is_vertical': is_vertical_pca,
    })
    
    if show_plots:
        # Create debug plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(edge_normalized, cmap='gray')
        axes[0].set_title(f'Original Edge ROI\n{edge_roi_array.shape[0]}×{edge_roi_array.shape[1]}')
        axes[0].axis('off')
        
        # Gradient magnitude
        axes[1].imshow(gradient_magnitude, cmap='viridis')
        axes[1].set_title(f'Gradient Magnitude\nMax: {np.max(gradient_magnitude):.3f}')
        axes[1].axis('off')
        
        # Edge points with PCA principal direction
        axes[2].imshow(edge_normalized, cmap='gray', alpha=0.7)
        axes[2].scatter(x_coords, y_coords, c='red', s=1, alpha=0.5, label='Edge points')
        
        # Draw PCA principal direction through centroid
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        # Draw line along principal direction
        x_span = np.max(x_coords) - np.min(x_coords)
        y_span = np.max(y_coords) - np.min(y_coords)
        length = max(x_span, y_span) * 0.6  # 60% of max span
        dx = principal_direction[0] * length
        dy = principal_direction[1] * length
        axes[2].plot([centroid_x - dx, centroid_x + dx], 
                       [centroid_y - dy, centroid_y + dy], 
                       'yellow', linewidth=2, label=f'PCA direction ({angle_pca_deg_abs:.1f}°)')
        axes[2].plot(centroid_x, centroid_y, 'go', markersize=8, label='Centroid')
        
        axes[2].set_title(f'Edge Detection (PCA)\nFound {len(x_coords)} edge points')
        axes[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Simple PCA results display
        st.write("**🎯 PCA Edge Detection Results:**")
        
        # Show key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Edge Angle", f"{angle_pca_deg_abs:.1f}°")
        with col2:
            st.metric("Orientation", "Vertical" if is_vertical_pca else "Horizontal")
        with col3:
            st.metric("PCA Confidence", f"{pca_confidence:.2f}")
        
        # Warnings based on angle and orientation
        if is_vertical_pca:
            # Vertical edge: optimal range is 85-87°
            if angle_pca_deg_abs < 85 or angle_pca_deg_abs > 87:
                st.warning(f"⚠️ Vertical edge angle ({angle_pca_deg_abs:.1f}°) outside optimal range (85-87°).")
            else:
                st.success(f"✅ Vertical edge angle ({angle_pca_deg_abs:.1f}°) is within optimal range (85-87°).")
        else:
            # Horizontal edge: optimal range is 3-5°
            if angle_pca_deg_abs < 3 or angle_pca_deg_abs > 5:
                st.warning(f"⚠️ Horizontal edge angle ({angle_pca_deg_abs:.1f}°) outside optimal range (3-5°).")
            else:
                st.success(f"✅ Horizontal edge angle ({angle_pca_deg_abs:.1f}°) is within optimal range (3-5°).")
    
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
        debug_info = debug_edge_detection(edge_roi_array, show_plots=False)
        
        results = {
            # Use PCA results
            'edge_angle_rad': debug_info.get('edge_angle_rad', 0),
            'edge_angle_deg': debug_info.get('edge_angle_deg', 0),
            'is_vertical': debug_info.get('is_vertical', False),
            'pca_confidence': debug_info.get('pca_confidence', 0),
            'num_edge_points': debug_info.get('num_edge_points', 0),
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