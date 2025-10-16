import streamlit as st
import pydicom
import numpy as np
import os
from PIL import Image

# Import functions from your analysis modules
from uniformity import display_uniformity_analysis_section
from NPS import display_nps_analysis_section
from MTF import display_mtf_analysis_section
from threshold_contrast import display_threshold_contrast_section
from comparison_tool import display_comparison_tool_section
from dicomizer import generate_dicom_from_raw
from detector_conversion import display_detector_conversion_section

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="X-ray Image Analysis Toolkit", layout="wide")

def main_app_ui():
    # --- Initialize session state for data sharing ---
    if 'mtf_data' not in st.session_state:
        st.session_state['mtf_data'] = None
    if 'nnps_data' not in st.session_state:
        st.session_state['nnps_data'] = None

    # Always display session state status in the sidebar for debugging
    st.sidebar.markdown("---")
    st.sidebar.subheader("Saved Analysis Data Status")
    st.sidebar.write(f"MTF Data: {'Loaded ✅' if st.session_state['mtf_data'] is not None else 'Missing ⚠️'}")
    st.sidebar.write(f"NNPS Data: {'Loaded ✅' if st.session_state['nnps_data'] is not None else 'Missing ⚠️'}")
    st.sidebar.markdown("---")

    # --- File Upload and Initial Image Display ---
    # Use a key for the file uploader to manage its state explicitly
    uploaded_files = st.sidebar.file_uploader(
        "Choose DICOM (.dcm) or RAW (.raw) file(s)",
        type=["dcm", "dicom", "raw"],
        accept_multiple_files=True
    )

    image_array = None
    pixel_spacing_row = None
    pixel_spacing_col = None
    dicom_filename = None
    dicom_dataset = None  # Store the full dataset, will be None for RAW
    is_difference_image = False  # Flag to track if we are analyzing a difference image

    if uploaded_files:
        # Determine file types
        file_extensions = {os.path.splitext(f.name)[1].lower() for f in uploaded_files}
        is_raw_upload = '.raw' in file_extensions
        is_dicom_upload = '.dcm' in file_extensions or '.dicom' in file_extensions

        # The comparison tool handles mixed files. For all other analyses, we only allow one type.
        is_comparison_candidate = is_raw_upload and is_dicom_upload

        if is_raw_upload and not is_dicom_upload:
            # Allow uploading multiple RAW files. Provide a selector to choose which RAW to analyze.
            raw_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() == '.raw']
            if not raw_files:
                st.error("No RAW files found in upload.")
                return

            # Default to the first RAW file if multiple are uploaded
            selected_raw = raw_files[0]

            # --- RAW File Processing ---
            raw_file = selected_raw
            dicom_filename = raw_file.name

            st.sidebar.subheader("RAW Image Parameters")
            st.sidebar.info("Please provide the details for your RAW image file.")

            dtype_map = {
                '16-bit Unsigned Integer': np.uint16,
                '8-bit Unsigned Integer': np.uint8,
                '32-bit Float': np.float32
            }
            dtype_str = st.sidebar.selectbox(
                "Pixel Data Type",
                options=list(dtype_map.keys()),
                index=0,
                key="raw_dtype",
                help="Select the data type first to auto-suggest dimensions."
            )
            np_dtype = dtype_map[dtype_str]

            try:
                raw_data = raw_file.getvalue()
                itemsize = np.dtype(np_dtype).itemsize
                file_size = len(raw_data)

                if file_size % itemsize != 0:
                    st.sidebar.error(f"File size ({file_size} bytes) is not a multiple of the pixel size ({itemsize} bytes). Please check the selected data type.")
                    return

                total_pixels = file_size // itemsize

                def get_factors(n):
                    factors = set()
                    for i in range(1, int(np.sqrt(n)) + 1):
                        if n % i == 0:
                            factors.add((i, n // i))
                            factors.add((n // i, i))
                    return sorted(list(factors))

                possible_dims = get_factors(total_pixels)

                if not possible_dims:
                    st.sidebar.warning("Could not determine any possible dimensions. Please enter manually.")
                    width = st.sidebar.number_input("Image Width (pixels)", min_value=1, value=1024)
                    height = st.sidebar.number_input("Image Height (pixels)", min_value=1, value=1024)
                else:
                    # Find the most "square" dimension and assign it.
                    default_dim_index = len(possible_dims) // 2
                    height, width = possible_dims[default_dim_index]
                    st.sidebar.info(f"Auto-detected dimensions: **{width} x {height}**")

                pixel_spacing_row = st.sidebar.number_input("Pixel Spacing Row (mm/px)", min_value=0.001, value=0.1, step=0.01, format="%.3f", key="raw_ps_row")
                pixel_spacing_col = st.sidebar.number_input("Pixel Spacing Col (mm/px)", min_value=0.001, value=0.1, step=0.01, format="%.3f", key="raw_ps_col")

                image_array = np.frombuffer(raw_data, dtype=np_dtype).reshape((height, width))
                st.sidebar.success("RAW file loaded successfully.")

                # --- "Dicomize" Feature (moved to sidebar for better UX) ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("Convert to DICOM")
                if st.sidebar.button("Generate DICOM file", key="dicomize_button"):
                    try:
                        # Call the refactored dicomizer function
                        dicom_bytes, new_filename = generate_dicom_from_raw(
                            image_array=image_array,
                            pixel_spacing_row=pixel_spacing_row,
                            pixel_spacing_col=pixel_spacing_col,
                            original_filename=dicom_filename
                        )
                        
                        # Create a download button in the sidebar
                        st.sidebar.download_button(
                            label="📥 Download .dcm file",
                            data=dicom_bytes,
                            file_name=new_filename,
                            mime="application/dicom",
                            key="dicom_download_button"
                        )
                        st.sidebar.success(f"Ready to download {new_filename}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to create DICOM: {e}")
            except Exception as e:
                st.error(f"Error processing RAW file: {e}")
                return

        elif is_dicom_upload:
            # Any DICOM upload routes to the comparison tool for RAW vs DICOM comparison.
            st.header("Developer: Compare RAW vs DICOM")
            display_comparison_tool_section(uploaded_files)
            return

    # --- Main Area ---
    if image_array is not None: # Standard analysis path
        st.header(f"Analysis for: {dicom_filename}")
        if pixel_spacing_row and pixel_spacing_col:
            st.write(f"Pixel Spacing: {pixel_spacing_row:.3f} mm/px (row) x {pixel_spacing_col:.3f} mm/px (col)")
        else:
            st.write("Pixel Spacing: Not available. Some analyses may be disabled or use pixel-based units.")

        # --- DICOM Header Information and Raw Data Option ---
        if dicom_dataset is not None:
            st.subheader("DICOM Image Properties")

            # Only display whether the image is DERIVED or not; other tag displays were removed.
            image_type = getattr(dicom_dataset, 'ImageType', ['UNKNOWN'])
            is_derived = 'DERIVED' in image_type
            is_primary = 'PRIMARY' in image_type or 'ORIGINAL' in image_type
            if is_derived:
                st.warning("This image is 'DERIVED' — it has undergone processing from original data.")
            elif is_primary:
                st.checkbox("This image appears to be original or suffered minimal processing.")
            else:
                st.info("Image processing cannot be checked.")

            if is_difference_image:
                st.success("This is a difference image created from two DICOMs using their stored pixel values.")
        else:  # This is a RAW file
            st.info("This is a RAW image file. Analysis will be performed directly on the pixel data using the parameters you provided in the sidebar.")

        # Display original image
        display_array = image_array.copy()  # Work on a copy to avoid modifying the array passed to analysis

        # To avoid overflow/underflow when normalizing integer dtypes, convert to float first
        display_array = display_array.astype(np.float64)

        # Handle potential NaN or inf values from previous calculations
        display_array[np.isnan(display_array)] = 0
        display_array[np.isinf(display_array)] = 0

        # Recompute min/max after sanitization
        min_val = np.min(display_array)
        max_val = np.max(display_array)

        if max_val > min_val:
            # Normalize to 0-255 range using float arithmetic
            display_array = 255.0 * (display_array - min_val) / (max_val - min_val)
        else:
            # Handle constant image
            display_array = np.zeros_like(display_array, dtype=np.float64)

        # Finally cast to uint8 for display
        display_array = np.clip(display_array, 0, 255).astype(np.uint8)

        if len(display_array.shape) == 2:
            img_pil = Image.fromarray(display_array)
            st.image(img_pil, caption="Loaded Image (Normalized for Display)", use_container_width=True)
        else:
            st.warning(f"Image has unexpected shape {image_array.shape}. Cannot display directly.")
        
        # --- New Tab-Based UI for Analysis Modules ---
        st.markdown("---")
        tab_detector, tab_uniformity, tab_nps, tab_mtf, tab_contrast = st.tabs([
            "Detector Conversion", "Uniformity Analysis", "NPS Analysis", "MTF Analysis", "Contrast Analysis"
        ])

        with tab_detector:
            # Prefer the sidebar-uploaded RAW files if available; pass them to the detector UI so re-upload is not needed
            raw_sidebar_files = None
            if uploaded_files:
                # Filter only .raw files from the sidebar upload list
                raw_sidebar_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() == '.raw'] or None

            detector_results = display_detector_conversion_section(uploaded_files=raw_sidebar_files)
            # If the detector module returned structured output, persist it to session state
            if detector_results is not None:
                st.session_state['detector_conv'] = detector_results

        with tab_uniformity:
            display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

        with tab_nps:
            display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

        with tab_mtf:
            display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

        with tab_contrast:
            display_threshold_contrast_section(pixel_spacing_row, pixel_spacing_col)

    elif uploaded_files: # This handles the comparison case where image_array is not pre-loaded
        # If files are uploaded but no single image array was created, it's the comparison tool case.
        st.header("Developer: Compare RAW vs DICOM")
        display_comparison_tool_section(uploaded_files)

    elif not uploaded_files:
        st.info("Please upload RAW files in the sidebar to analyze detector response curve.")

    # --- Add a clear session state button for debugging ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Saved Analysis Data"):
        st.session_state['mtf_data'] = None
        st.session_state['nnps_data'] = None
        st.success("All saved analysis data cleared!")
        st.rerun() # Rerun to reflect the cleared state


if __name__ == "__main__":
    
    # Run the main Streamlit UI
    main_app_ui()
