import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import os
import tempfile

# Import functions from your analysis modules
from uniformity import display_uniformity_analysis_section
from NPS import display_nps_analysis_section
from MTF import display_mtf_analysis_section
from threshold_contrast import display_threshold_contrast_section
from comparison_tool import display_comparison_tool_section

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
            if len(uploaded_files) > 1:
                st.error("Analysis of multiple RAW files is not yet supported. Please upload a single .raw file.")
                return
            
            # --- RAW File Processing ---
            raw_file = uploaded_files[0]
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

            except Exception as e:
                st.error(f"Error processing RAW file: {e}")
                return

        elif is_dicom_upload and not is_raw_upload:
            st.sidebar.subheader("DICOM Processing Options")
            # Note: DICOM-to-RAW conversion feature removed. We will use the stored pixel data
            # available in the DICOM (`pixel_array`) for analysis. Any footer trimming logic
            # has been removed.

            if len(uploaded_files) == 1:
                uploaded_file_widget = uploaded_files[0]
                dicom_filename = uploaded_file_widget.name
                try:
                    dicom_dataset = pydicom.dcmread(uploaded_file_widget)
                    if 'PixelData' in dicom_dataset:
                        # Use pydicom's pixel_array (the stored/stored values) for analysis.
                        # Note: we intentionally do NOT attempt to recreate a RAW file or
                        # infer processing steps beyond the final stored pixel values.
                        try:
                            # Ensure decompression if needed
                            if dicom_dataset.file_meta.TransferSyntaxUID.is_compressed:
                                dicom_dataset.decompress()
                        except Exception:
                            # Some files may not have file_meta; ignore decompression errors
                            pass

                        image_array = dicom_dataset.pixel_array
                    else:
                        st.error("DICOM file does not contain pixel data.")
                        return
                    
                    if 'PixelSpacing' in dicom_dataset:
                        pixel_spacing = dicom_dataset.PixelSpacing
                        if len(pixel_spacing) == 2:
                            pixel_spacing_row = float(pixel_spacing[0])
                            pixel_spacing_col = float(pixel_spacing[1])
                        else:
                            st.warning(f"Pixel Spacing tag (0028,0030) has unexpected format: {pixel_spacing}.")
                    else:
                        st.warning("Pixel Spacing tag (0028,0030) not found in DICOM header.")

                except Exception as e:
                    st.error(f"Error reading DICOM file: {e}")
                    return

            elif len(uploaded_files) == 2:
                is_difference_image = True
                st.info("Two DICOM files uploaded. Each image will be reverted to its raw 'stored values' before subtraction to create the difference image.")
                img_arrays_stored_values = []
                pixel_spacings = []
                filenames = []
                dicom_dataset = None

                for i, uploaded_file_widget in enumerate(uploaded_files):
                    filenames.append(uploaded_file_widget.name)
                    try:
                        ds_temp = pydicom.dcmread(uploaded_file_widget, force=True)
                        if i == 0:
                            dicom_dataset = ds_temp  # Store the first dataset object
                        if 'PixelData' not in ds_temp:
                            st.error(f"DICOM file {uploaded_file_widget.name} does not contain pixel data.")
                            return

                        # Directly get the stored pixel data for each image
                        try:
                            if ds_temp.file_meta.TransferSyntaxUID.is_compressed:
                                ds_temp.decompress()
                        except Exception:
                            pass
                        stored_pixel_array = ds_temp.pixel_array
                        img_arrays_stored_values.append(stored_pixel_array)

                        # Get pixel spacing
                        if 'PixelSpacing' in ds_temp and len(ds_temp.PixelSpacing) == 2:
                            ps = ds_temp.PixelSpacing
                            pixel_spacings.append((float(ps[0]), float(ps[1])))
                        else:
                            st.warning(f"Pixel Spacing not found or invalid in {uploaded_file_widget.name}.")
                            pixel_spacings.append((None, None))

                    except Exception as e:
                        st.error(f"Error reading DICOM file {uploaded_file_widget.name}: {e}")
                        return

                if len(img_arrays_stored_values) == 2:
                    if img_arrays_stored_values[0].shape != img_arrays_stored_values[1].shape:
                        st.error(f"Image dimensions mismatch: {filenames[0]} ({img_arrays_stored_values[0].shape}) vs {filenames[1]} ({img_arrays_stored_values[1].shape}). Cannot calculate difference.")
                        return
                    if pixel_spacings[0] != pixel_spacings[1] and pixel_spacings[0] is not None and pixel_spacings[1] is not None:
                        st.warning(f"Pixel spacings mismatch: {filenames[0]} ({pixel_spacings[0]}) vs {filenames[1]} ({pixel_spacings[1]}). Using spacing from the first image.")

                    # Calculate difference image from stored values
                    image_array = img_arrays_stored_values[0] - img_arrays_stored_values[1]
                    dicom_filename = f"Difference of {filenames[0]} and {filenames[1]}"
                    
                    # Use pixel spacing from the first image
                    if pixel_spacings[0][0] is not None:
                        pixel_spacing_row, pixel_spacing_col = pixel_spacings[0]
                    else:
                        st.warning("Pixel spacing not available for difference image. NPS will use cycles/pixel.")
                else:
                    st.warning("Could not process the two DICOM files.")
                    return
            else:
                st.warning("Please upload either one or two DICOM files for analysis.")
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
        
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ("Select an analysis...", "Uniformity Analysis", "NPS Analysis", "MTF Analysis", "Contrast Analysis", "Developer: Compare RAW vs DICOM")
        )

        if analysis_type == "Uniformity Analysis":
            display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "NPS Analysis":
            display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "MTF Analysis":
            display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "Contrast Analysis":
            display_threshold_contrast_section(pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "Developer: Compare RAW vs DICOM":
            # This case should not be hit if image_array is loaded, but as a fallback:
            st.error("Comparison tool cannot be run on a single pre-loaded image. Please upload one DICOM and one RAW file together.")

    elif uploaded_files: # This handles the comparison case where image_array is not pre-loaded
        analysis_type = st.sidebar.selectbox("Choose Analysis Type:", ("Developer: Compare RAW vs DICOM",))
        if analysis_type == "Developer: Compare RAW vs DICOM":
            display_comparison_tool_section(uploaded_files)

    elif not uploaded_files:
        st.info("Please upload one or two DICOM files, or a single RAW file, using the sidebar to begin analysis.")

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
