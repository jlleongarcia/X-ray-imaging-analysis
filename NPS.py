import streamlit as st
import numpy as np
from pylinac.core.nps import noise_power_spectrum_2d, noise_power_spectrum_1d, radial_average
import pandas as pd
import altair as alt
from typing import List, Optional
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


def _load_uploaded_images(files) -> List[np.ndarray]:
    """Best-effort loader for uploaded files into grayscale numpy arrays."""
    arrays: List[np.ndarray] = []
    for f in files or []:
        arr = None
        # Try DICOM first if pydicom is available
        if pydicom is not None:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, 'pixel_array'):
                    arr = ds.pixel_array.astype(float)
            except Exception:
                arr = None
        # Fallback to PIL
        if arr is None and Image is not None:
            try:
                img = Image.open(f)
                if img.mode != 'L':
                    img = img.convert('L')
                arr = np.array(img, dtype=float)
            except Exception:
                arr = None
        if arr is not None and arr.ndim == 2:
            arrays.append(arr)
        else:
            st.warning(f"Could not load an image from uploaded file: {getattr(f, 'name', 'unknown')}")
    return arrays


def calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col, additional_images: Optional[List[np.ndarray]] = None):
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

    # Collect all images (base + additional) and apply optional inverse detector conversion to each
    all_images: List[np.ndarray] = [image_array]
    if additional_images:
        all_images.extend([img for img in additional_images if isinstance(img, np.ndarray) and img.ndim == 2])

    # Ensure all images are same shape
    shapes = {img.shape for img in all_images}
    if len(shapes) > 1:
        st.error("All images must have identical dimensions for IEC NPS calculation. Please upload images with matching size.")
        return {"NPS_Status": "Error: Image sizes do not match"}

    # Attempt inverse detector conversion to kerma domain if a linear or log fit is cached.
    domain_used = "pixel"
    conv = st.session_state.get("detector_conversion")
    def _apply_inverse(img: np.ndarray) -> np.ndarray:
        nonlocal domain_used
        if not (isinstance(conv, dict) and conv.get("coeffs") is not None):
            return img
        method = conv.get("method")
        coeffs = np.array(conv.get("coeffs"), dtype=float)
        try:
            if method == 'linear':
                a, b = coeffs
                if a == 0:
                    raise ValueError("Inverse linear conversion undefined (a=0)")
                domain_used = "kerma"
                return (img.astype(float) - b) / a
            elif method == 'log':
                a, b = coeffs
                if a == 0:
                    raise ValueError("Inverse log conversion undefined (a=0)")
                with np.errstate(over='ignore', invalid='ignore'):
                    out = np.exp((img.astype(float) - b) / a)
                domain_used = "kerma"
                return out
            else:
                return img
        except Exception as inv_e:
            st.warning(f"Inverse detector conversion failed ({inv_e}); proceeding with pixel-domain NPS.")
            domain_used = "pixel"
            return img

    all_images = [_apply_inverse(img) for img in all_images]

    try:
        # Compute NPS for each uploaded image independently and average the resulting NNPS across images.
        # Multiple images improve statistics when the image size is small. All images must share identical dimensions.

        # Define the list of allowed power-of-two values for the large central ROI and for the small ROIs within
        allowed_big = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        allowed_small = [8, 16, 32, 64, 128, 256, 512]
        selected_big = st.select_slider(
            label="Slide to select the pixel size for the large central ROI:",
            options=allowed_big,
            value=1024
        )
        selected_small = st.select_slider(
            label="Slide to select the pixel size for the small ROIs within:",
            options=allowed_small,
            value=128
        )
        nnps_2d_list = []
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
            nnps_2d = nps_2d_raw / (mean_pv ** 2)
            nnps_2d_list.append(nnps_2d)

        if not nnps_2d_list:
            st.error("No valid images were processed for NNPS. Please verify uploaded images and exposure consistency.")
            return {"NPS_Status": "Error: No valid images"}

        # Average NNPS across images
        nnps_2d_avg = np.mean(np.stack(nnps_2d_list, axis=0), axis=0)

        # Compute an estimate of total ROI pixels used for statistics guidance (~4 million recommended)
        n_small_side = max(1, selected_big // selected_small)
        rois_per_image = n_small_side * n_small_side
        pixels_per_roi = selected_small * selected_small
        total_roi_pixels = int(rois_per_image * pixels_per_roi * len(nnps_2d_list))

        # --- 1D NNPS Calculation from averaged 2D NNPS ---
        nnps_1d_result = noise_power_spectrum_1d(spectrum_2d=nnps_2d_avg)

        # To get the corresponding frequency axis, we do the same radial average
        # on a grid of radial frequencies. This is the most accurate method.
        # 1. Create the frequency axes for the 2D grid.
        freqs = np.fft.fftshift(np.fft.fftfreq(selected_small, d=pixel_spacing_avg))

        # 2. Create the 2D frequency grid (meshgrid).
        fx, fy = np.meshgrid(freqs, freqs)
        f_grid = np.sqrt(fx**2 + fy**2)

        # 3. Apply the same radial average to get the 1D frequency axis
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
            "domain_used": domain_used,
            "used_images": int(len(nnps_2d_list)),
            "total_roi_pixels": total_roi_pixels,
        }
    except Exception as e:
        st.error(f"Error during NPS calculation: {e}")
        return {"NPS_Status": f"Error: {e}"}

def display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col, uploaded_files=None):
    st.subheader("Noise Power Spectrum (NPS) Analysis")

    st.write("""
    Computes NPS per image and averages NNPS across all images you've uploaded in the sidebar.
    If your images are small, upload multiple flat-field images (same size & exposure) in the sidebar to improve statistics.
    Images must have identical dimensions; aim for ~4 million total ROI pixels.
    """)

    # Use the images already uploaded in the sidebar
    additional_arrays = _load_uploaded_images(uploaded_files) if uploaded_files else []

    if st.button("Run NPS Analysis"):
        with st.spinner("Calculating NPS..."):
            nps_results_dict = calculate_nps_metrics(image_array, pixel_spacing_row, pixel_spacing_col, additional_images=additional_arrays)
            
            if "NPS_Status" in nps_results_dict and "Error" in nps_results_dict["NPS_Status"]:
                st.error(f"NPS Calculation Failed: {nps_results_dict['NPS_Status']}")
            elif nps_results_dict and "NNPS_1D_chart_data" in nps_results_dict:
                st.success("NPS Analysis Complete!")
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
