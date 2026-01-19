import streamlit as st
import pydicom
import numpy as np
import os
import io
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

def detect_file_type(file_bytes, filename):
    """
    Detect file type by examining DICOM tag (0008,0068) Presentation Intent Type.
    Returns: 'dicom', 'raw', or 'unknown'
    
    Logic:
    - If DICOM tag (0008,0068) exists:
      - "FOR PRESENTATION" → true DICOM file
      - "FOR PROCESSING" → RAW file (even if has .dcm extension)
    - If no DICOM tag → RAW file
    """
    try:
        # Try to parse as DICOM to check for tag (0008,0068)
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True, stop_before_pixels=True)
            
            # Check for Presentation Intent Type tag (0008,0068)
            if hasattr(ds, 'PresentationIntentType') and ds.PresentationIntentType:
                presentation_intent = str(ds.PresentationIntentType).upper().strip()
                
                # Check the presentation intent type
                if "FOR PRESENTATION" in presentation_intent:
                    return 'dicom'
                elif "FOR PROCESSING" in presentation_intent:
                    return 'raw'
                else:
                    # Has PresentationIntentType but neither FOR PRESENTATION nor FOR PROCESSING
                    # Default to DICOM if it's a valid DICOM structure
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        return 'dicom'
            else:
                # Valid DICOM structure but no PresentationIntentType tag - assume DICOM
                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    return 'dicom'
                    
        except Exception:
            # Not a valid DICOM file - treat as RAW
            return 'raw'
        
        # Fallback: if we reach here, treat as RAW
        return 'raw'
        
    except Exception:
        return 'unknown'

def main_app_ui():
    # --- Initialize session state for data sharing ---
    # Ensure detector conversion cache key exists (stores predict_mpv and fit metadata)
    if 'detector_conversion' not in st.session_state:
        st.session_state['detector_conversion'] = None

    # Always display session state status in the sidebar for debugging
    st.sidebar.markdown("---")
    st.sidebar.subheader("Saved Analysis Data Status")
    _dc = st.session_state['detector_conversion']
    _dc_loaded = isinstance(_dc, dict) and (_dc.get('predict_mpv') is not None)
    st.sidebar.write(f"Detector Conversion Fn: {'Loaded ✅' if _dc_loaded else 'Missing ⚠️'}")
    st.sidebar.markdown("---")

    # --- File Upload and Initial Image Display ---
    # Use a key for the file uploader to manage its state explicitly
    uploaded_files = st.sidebar.file_uploader(
        "Choose DICOM (.dcm), RAW/STD (.raw/.std) files, or extensionless files",
        type=None,  # Accept all file types including extensionless
        accept_multiple_files=True,
        help="Maximum file size: 500 MB per file"
    )

    image_array = None
    pixel_spacing_row = None
    pixel_spacing_col = None
    dicom_filename = None
    dicom_dataset = None  # Store the full dataset, will be None for RAW

    if uploaded_files:
        # Determine file types by content and extension
        file_types = []
        for f in uploaded_files:
            file_bytes = f.getvalue()
            detected_type = detect_file_type(file_bytes, f.name)
            file_types.append((f, detected_type))
        
        # Categorize files
        dicom_files = [f for f, ftype in file_types if ftype == 'dicom']
        raw_files = [f for f, ftype in file_types if ftype == 'raw']
        unknown_files = [f for f, ftype in file_types if ftype == 'unknown']
        
        # Show info about detected file types
        if uploaded_files:
            st.sidebar.markdown("**Detected file types:**")
            for f, ftype in file_types:
                if ftype == 'dicom':
                    st.sidebar.write(f"🏥 {f.name} → DICOM (FOR PRESENTATION)")
                elif ftype == 'raw':
                    ext = os.path.splitext(f.name)[1].lower()
                    if ext in ['.dcm', '.dicom']:
                        st.sidebar.write(f"📷 {f.name} → RAW (FOR PROCESSING)")
                    elif ext:
                        st.sidebar.write(f"📷 {f.name} → RAW/STD")
                    else:
                        st.sidebar.write(f"📷 {f.name} → RAW (extensionless)")
                else:
                    st.sidebar.write(f"❓ {f.name} → Unknown")
        
        # Debug information for DICOM tag checking
        if uploaded_files and st.sidebar.checkbox("Show DICOM Debug Info", value=False):
            st.sidebar.markdown("**Debug Info:**")
            for f, ftype in file_types:
                try:
                    ds = pydicom.dcmread(io.BytesIO(f.getvalue()), force=True, stop_before_pixels=True)
                    if hasattr(ds, 'PresentationIntentType'):
                        intent = ds.PresentationIntentType
                        st.sidebar.write(f"🔍 {f.name}: PresentationIntentType = '{intent}'")
                    else:
                        st.sidebar.write(f"🔍 {f.name}: No PresentationIntentType tag")
                except Exception as e:
                    st.sidebar.write(f"🔍 {f.name}: Not DICOM or parse error")
        
        # Show warning for unknown files
        if unknown_files:
            st.sidebar.warning(f"Unknown file types detected: {[f.name for f in unknown_files]}")
        
        is_dicom_upload = len(dicom_files) > 0
        is_raw_upload = len(raw_files) > 0
        
        # Check for mixed file types and show error (except for comparison tool)
        if is_dicom_upload and is_raw_upload:
            st.sidebar.error("⚠️ Mixed file types detected! Please upload only one type of files (either all RAW or all DICOM). Note: You can still use the comparison tool to compare RAW vs DICOM files.")
            if st.sidebar.button("Use Comparison Tool Instead"):
                st.header("Developer: Compare RAW vs DICOM")
                display_comparison_tool_section(uploaded_files)
                return
            else:
                return  # Stop processing until user fixes the upload
        
        # The comparison tool handles mixed files. For all other analyses, we only allow one type.
        is_comparison_candidate = is_raw_upload and is_dicom_upload

        if is_raw_upload and not is_dicom_upload:
            # Allow uploading multiple RAW/STD files. Use detected raw files.
            if not raw_files:
                st.error("No RAW/STD files found in upload.")
                return

            # Default to the first RAW file if multiple are uploaded
            selected_raw = raw_files[0]

            # --- RAW or DICOM (forced) Processing ---
            raw_file = selected_raw
            dicom_filename = raw_file.name

            st.sidebar.subheader("RAW/STD Image Parameters")
            st.sidebar.info("Please provide the details for your RAW or STD image file.")

            # Read bytes once for detection and later processing
            raw_data = raw_file.getvalue()

            # Heuristic: detect DICOM-like content even for .raw/.std
            dicom_suspected = False
            try:
                if len(raw_data) >= 132 and raw_data[128:132] == b'DICM':
                    dicom_suspected = True
                else:
                    # Try parsing header only (no pixel decode) with force
                    ds_hdr = pydicom.dcmread(io.BytesIO(raw_data), force=True, stop_before_pixels=True)
                    if hasattr(ds_hdr, 'Rows') and hasattr(ds_hdr, 'Columns'):
                        dicom_suspected = True
            except Exception:
                pass

            # Let user choose interpretation: RAW/STD (by extension) or DICOM
            interpret_options = ["RAW/STD", "DICOM"]
            default_index = 1 if dicom_suspected else 0
            interpret_choice = st.sidebar.radio(
                "Interpret uploaded file as:", options=interpret_options, index=default_index,
                help="If your file is actually a DICOM saved with .raw/.std extension, choose DICOM to parse headers.", key="interpret_choice_raw"
            )
            if dicom_suspected:
                st.sidebar.caption("DICOM-like header detected; defaulted to DICOM.")

            if interpret_choice == "DICOM":
                # Try to parse as DICOM even without preamble
                try:
                    ds = pydicom.dcmread(io.BytesIO(raw_data), force=True)
                    dicom_dataset = ds
                    dicom_filename = raw_file.name
                    # Extract pixel spacing if available
                    ps = getattr(ds, 'PixelSpacing', None)
                    if ps and len(ps) >= 2:
                        pixel_spacing_row = float(ps[0])
                        pixel_spacing_col = float(ps[1])
                    # Pixel data
                    image_array = ds.pixel_array
                    st.sidebar.success("Parsed as DICOM (forced). Using Rows/Columns from header.")
                except Exception as e:
                    st.sidebar.error(f"Failed to parse as DICOM: {e}")
                    return
            else:
                # RAW/STD interpretation path
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
                    st.sidebar.error(f"Could not determine valid dimensions for {total_pixels} pixels.")
                    return

                # Filter to keep only reasonable aspect ratios (width/height between 1/3 and 3)
                reasonable_dims = []
                for h, w in possible_dims:
                    aspect_ratio = w / h
                    if 1/3 <= aspect_ratio <= 3:
                        reasonable_dims.append((h, w))
                
                if not reasonable_dims:
                    # Fallback to all dims if filtering removed everything
                    reasonable_dims = possible_dims
                    st.sidebar.warning("No square-like dimensions found. Showing all possibilities.")
                
                # Find the most "square" dimension as default
                default_dim_index = len(reasonable_dims) // 2
                
                # Format dimensions for dropdown display
                dim_options = [f"{w} x {h}" for h, w in reasonable_dims]
                
                # Let user select from dropdown
                selected_dim = st.sidebar.selectbox(
                    "Image Dimensions (Width x Height)",
                    options=dim_options,
                    index=default_dim_index,
                    key="raw_dimensions",
                    help="Auto-detected dimensions. Showing only square-like options (aspect ratio between 1:3 and 3:1)."
                )
                
                # Parse selected dimensions
                width, height = map(int, selected_dim.split(" x "))
                st.sidebar.caption(f"Selected: **{width} x {height}** pixels ({width * height:,} total pixels)")

                pixel_spacing_row = st.sidebar.number_input("Pixel Spacing Row (mm/px)", min_value=0.001, value=0.1, step=0.01, format="%.3f", key="raw_ps_row")
                pixel_spacing_col = st.sidebar.number_input("Pixel Spacing Col (mm/px)", min_value=0.001, value=0.1, step=0.01, format="%.3f", key="raw_ps_col")

                image_array = np.frombuffer(raw_data, dtype=np_dtype).reshape((height, width))
                st.sidebar.success("RAW file loaded successfully.")

            # --- "Dicomize" Feature ---
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
                st.success("This image appears to be original or suffered minimal processing.")
            else:
                st.info("Image processing cannot be checked.")

        else:  # This is a RAW/STD file
            st.info("This is a RAW/STD image file. Analysis will be performed directly on the pixel data using the parameters you provided in the sidebar.")

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
            with st.expander("📷 Loaded Image", expanded=False):
                st.image(img_pil, caption="Loaded Image (Normalized for Display)", use_container_width=True)
        else:
            st.warning(f"Image has unexpected shape {image_array.shape}. Cannot display directly.")
        
        # --- Analysis Type Selection ---
        st.markdown("---")
        
        # Main category selection
        analysis_category = st.selectbox(
            "Select Analysis Category",
            ["Convert to DICOM", "Flat Panel Analysis"],
            help="Choose the analysis type or utility tool"
        )

        if analysis_category == "Convert to DICOM":
            st.subheader("🏥 Convert Image to DICOM Format")
            st.markdown("""
            Convert your raw or standard image file to DICOM format with customizable metadata.
            Upload an image file in the sidebar first.
            """)
            
            if st.button("Generate DICOM file", key="dicomize_button", type="primary", use_container_width=True):
                try:
                    # Call the dicomizer function
                    dicom_bytes, new_filename = generate_dicom_from_raw(
                        image_array,
                        original_filename=dicom_filename if dicom_filename else "converted_image.dcm",
                        pixel_spacing_row=pixel_spacing_row,
                        pixel_spacing_col=pixel_spacing_col
                    )
                    
                    st.success(f"✅ DICOM file generated: **{new_filename}**")
                    st.download_button(
                        label="⬇️ Download DICOM",
                        data=dicom_bytes,
                        file_name=new_filename,
                        mime="application/dicom",
                        key="download_dicom",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ DICOM generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif analysis_category == "Flat Panel Analysis":
            # Sub-category tabs for flat panel analyses
            tab_detector, tab_uniformity, tab_nps, tab_mtf, tab_contrast = st.tabs([
                "Detector Conversion", "Uniformity", "NPS", "MTF", "Contrast"
            ])

            with tab_detector:
                # Prefer the sidebar-uploaded RAW/STD files if available; pass them to the detector UI so re-upload is not needed
                raw_sidebar_files = None
                if uploaded_files:
                    # Use the detected raw files from content analysis
                    raw_sidebar_files = raw_files or None

                detector_results = display_detector_conversion_section(uploaded_files=raw_sidebar_files)
                # If the detector module returned structured output, persist it to session state
                if detector_results is not None:
                    st.session_state['detector_conv'] = detector_results

            with tab_uniformity:
                display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

            with tab_nps:
                # Pass all files uploaded in the sidebar into NPS so it can use them all
                display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col, uploaded_files=uploaded_files)

            with tab_mtf:
                display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

            with tab_contrast:
                display_threshold_contrast_section(image_array, pixel_spacing_row, pixel_spacing_col)

    elif uploaded_files: # This handles the comparison case where image_array is not pre-loaded
        # If files are uploaded but no single image array was created, it's the comparison tool case.
        st.header("Developer: Compare RAW vs DICOM")
        display_comparison_tool_section(uploaded_files)

    elif not uploaded_files:
        st.info("Please upload RAW files in the sidebar to analyze detector response curve.")

    # --- Add a clear session state button for debugging ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Saved Analysis Data"):
        st.session_state['detector_conversion'] = None
        # Also clear structured detector results if present
        st.session_state['detector_conv'] = None
        st.success("All saved analysis data cleared!")
        st.rerun() # Rerun to reflect the cleared state


if __name__ == "__main__":
    
    # Run the main Streamlit UI
    main_app_ui()
