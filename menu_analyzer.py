import streamlit as st
import pydicom
import numpy as np
import os
import io
from raw_endian import frombuffer_with_endian
from analysis_payload import ImagePayload, file_name_and_bytes
from metadata_summary import render_metadata_summary

# Import functions from your analysis modules
from uniformity import display_uniformity_analysis_section
from NPS import display_nps_analysis_section
from MTF import display_mtf_analysis_section
from threshold_contrast import display_threshold_contrast_section
from comparison_tool import display_comparison_tool_section
from dicomizer import generate_dicom_from_raw
from detector_conversion import display_detector_conversion_section
from DQE import display_dqe_analysis_section

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="X-ray Image Analysis Toolkit", layout="wide")


def _build_preloaded_payloads(file_types) -> list[ImagePayload]:
    """Build preloaded payloads so bytes are read once and shared across APIs."""
    payloads = []
    for f, ftype, detected_type in file_types:
        fname, file_bytes = file_name_and_bytes(f)
        payloads.append({
            'name': fname,
            'bytes': file_bytes,
            'file_type': ftype,
            'detected_type': detected_type,
        })
    return payloads


def _ensure_payload_loaded(payload: dict, show_status: bool = False) -> dict:
    """Ensure a preloaded payload has decoded image data from centralized loader."""
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get('image_array'), np.ndarray):
        return payload

    file_type = payload.get('file_type')
    if file_type not in ('raw', 'dicom'):
        return payload

    img, ps_row, ps_col, _ = load_single_image(
        payload,
        file_type,
        show_status=show_status,
        shared_raw_params=st.session_state.get('shared_raw_params_current_test'),
    )
    payload['image_array'] = img
    payload['pixel_spacing_row'] = ps_row
    payload['pixel_spacing_col'] = ps_col
    return payload


def _build_shared_raw_params(raw_payloads, context_key=""):
    """Render one shared RAW parameter section and return parameters for all RAW files."""
    if not raw_payloads:
        return None

    ref_name, ref_bytes = file_name_and_bytes(raw_payloads[0])

    dicom_rows = None
    dicom_cols = None
    dicom_ps_row = None
    dicom_ps_col = None

    try:
        ds_meta = pydicom.dcmread(io.BytesIO(ref_bytes), force=True, stop_before_pixels=True)
        dicom_rows = getattr(ds_meta, 'Rows', None)
        dicom_cols = getattr(ds_meta, 'Columns', None)
        ps = getattr(ds_meta, 'ImagerPixelSpacing', None)
        if ps is None:
            ps = getattr(ds_meta, 'PixelSpacing', None)
        if ps and len(ps) >= 2:
            dicom_ps_row = float(ps[0])
            dicom_ps_col = float(ps[1])
    except Exception:
        pass

    with st.sidebar:
        st.markdown("### 🔧 RAW File Parameters")
        st.caption("Applied to all uploaded RAW/STD files in this analysis.")
        st.caption(f"Reference file: {ref_name}")

        if dicom_rows and dicom_cols:
            st.info(f"📋 DICOM metadata found: {dicom_cols}×{dicom_rows} pixels")

        dtype_map = {
            '16-bit Unsigned Integer': np.uint16,
            '8-bit Unsigned Integer': np.uint8,
            '32-bit Float': np.float32
        }
        dtype_str = st.selectbox(
            "Pixel Data Type",
            options=list(dtype_map.keys()),
            index=0,
            key=f"{context_key}dtype_shared"
        )
        np_dtype = dtype_map[dtype_str]

        itemsize = np.dtype(np_dtype).itemsize
        file_size = len(ref_bytes)

        if file_size % itemsize != 0:
            st.error(f"File size ({file_size} bytes) is not a multiple of pixel size ({itemsize} bytes)")
            return None

        total_pixels = file_size // itemsize

        if dicom_rows and dicom_cols:
            width = int(dicom_cols)
            height = int(dicom_rows)
            st.write("**Dimensions (from DICOM tags):**")
            st.write(f"{width} × {height} pixels")
            st.caption("Using Rows (0028,0010) and Columns (0028,0011) tags")
        else:
            def get_factors(n):
                factors = set()
                for i in range(1, int(np.sqrt(n)) + 1):
                    if n % i == 0:
                        factors.add((i, n // i))
                        factors.add((n // i, i))
                return sorted(list(factors))

            possible_dims = get_factors(total_pixels)
            if not possible_dims:
                st.error(f"Could not determine valid dimensions for {total_pixels} pixels")
                return None

            reasonable_dims = []
            for h, w in possible_dims:
                aspect_ratio = w / h
                if 1 / 3 <= aspect_ratio <= 3:
                    reasonable_dims.append((h, w))

            if not reasonable_dims:
                reasonable_dims = possible_dims
                st.warning("Showing all dimensions (no square-like options found)")

            default_dim_index = len(reasonable_dims) // 2
            dim_options = [f"{w} x {h}" for h, w in reasonable_dims]
            selected_dim = st.selectbox(
                "Image Dimensions (Width x Height)",
                options=dim_options,
                index=default_dim_index,
                key=f"{context_key}dims_shared"
            )

            width, height = map(int, selected_dim.split(" x "))
            st.caption(f"Selected: **{width} x {height}** pixels ({width * height:,} total)")

        default_ps_row = dicom_ps_row if dicom_ps_row else 0.1
        default_ps_col = dicom_ps_col if dicom_ps_col else 0.1

        pixel_spacing_row = st.number_input(
            "Pixel Spacing Row (mm/px)",
            min_value=0.001,
            value=default_ps_row,
            step=0.01,
            format="%.3f",
            key=f"{context_key}ps_row_shared",
            help="From ImagerPixelSpacing (0018,1164)" if dicom_ps_row else None
        )
        pixel_spacing_col = st.number_input(
            "Pixel Spacing Col (mm/px)",
            min_value=0.001,
            value=default_ps_col,
            step=0.01,
            format="%.3f",
            key=f"{context_key}ps_col_shared",
            help="From ImagerPixelSpacing (0018,1164)" if dicom_ps_col else None
        )

    return {
        'dtype': np_dtype,
        'width': int(width),
        'height': int(height),
        'pixel_spacing_row': float(pixel_spacing_row),
        'pixel_spacing_col': float(pixel_spacing_col),
    }

def detect_file_type(file_bytes, filename):
    """
    Detect file type by examining DICOM tag (0008,0068) Presentation Intent Type.
    Returns: 'dicom', 'raw', or 'unknown'
    
    Logic:
    - If DICOM tag (0008,0068) exists:
      - "FOR PRESENTATION" → true DICOM file
      - "FOR PROCESSING" → RAW file (even if has .dcm extension)
        - If no PresentationIntentType, inspect ImageType (0008,0008):
            - ORIGINAL/PRIMARY → RAW/STD file
            - DERIVED/SECONDARY → DICOM file
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
                # No PresentationIntentType: fallback to ImageType (0008,0008)
                image_type = getattr(ds, 'ImageType', None)
                if image_type:
                    if isinstance(image_type, str):
                        image_type_values = [x.strip().upper() for x in image_type.replace('\\', '/').split('/') if x.strip()]
                    else:
                        image_type_values = [str(x).strip().upper() for x in image_type if str(x).strip()]

                    if 'ORIGINAL' in image_type_values or 'PRIMARY' in image_type_values:
                        return 'raw'
                    if 'DERIVED' in image_type_values or 'SECONDARY' in image_type_values:
                        return 'dicom'

                # If ImageType is absent or not matching expected values,
                # keep previous behavior for valid DICOM structures
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
    # Ensure detector conversion cache key exists (single nested object for fit + results)
    if 'detector_conversion' not in st.session_state:
        st.session_state['detector_conversion'] = {
            'fit': {},
            'results': None,
            'ei_fit': {},
            'sd2_fit': {},
        }
    
    # Initialize selected category and test in session state
    if 'selected_category' not in st.session_state:
        st.session_state['selected_category'] = None
    if 'selected_test' not in st.session_state:
        st.session_state['selected_test'] = None

    if 'raw_little_endian_default' not in st.session_state:
        st.session_state['raw_little_endian_default'] = True

    # Always display session state status in the sidebar for debugging
    st.sidebar.markdown("---")
    st.sidebar.subheader("Saved Analysis Data Status")
    
    # Detector conversion status
    _dc = st.session_state['detector_conversion']
    _fit = _dc.get('fit', {}) if isinstance(_dc, dict) else {}
    _dc_loaded = isinstance(_fit, dict) and (_fit.get('predict_mpv') is not None)
    st.sidebar.write(f"Detector Conversion: {'✅ Cached' if _dc_loaded else '⚠️ Missing'}")
    _dc_struct = _dc.get('results') if isinstance(_dc, dict) else None
    if isinstance(_dc_struct, dict) and isinstance(_dc_struct.get('files'), list):
        st.sidebar.write(f"Detector files cached: {len(_dc_struct['files'])}")
    
    # MTF cache status
    _mtf_cache = st.session_state.get('mtf_cache')
    if _mtf_cache and isinstance(_mtf_cache, dict):
        _mtf_results = _mtf_cache.get('results', [])
        _geom_mean = _mtf_cache.get('mtf_geometric_mean')
        if _geom_mean and _geom_mean.get('available'):
            st.sidebar.write(f"MTF: ✅ {len(_mtf_results)} measurements + Geometric mean")
        else:
            st.sidebar.write(f"MTF: ⚠️ Not enough measurements for geometric mean")
    else:
        st.sidebar.write("MTF: ⚠️ Not computed")
    
    # NPS cache status
    _nps_cache = st.session_state.get('nps_cache')
    if _nps_cache and isinstance(_nps_cache, dict):
        _kerma = _nps_cache.get('kerma_ugy', 'N/A')
        st.sidebar.write(f"NPS: ✅ Cached (K = {_kerma:.2f} μGy)")
    else:
        st.sidebar.write("NPS: ⚠️ Not computed")
    
    # DQE status (computed on-demand, not cached)
    if _mtf_cache and _nps_cache:
        _mtf_valid = _mtf_cache.get('mtf_geometric_mean', {}).get('available', False)
        if _mtf_valid:
            st.sidebar.write("DQE: ✅ Ready to compute")
        else:
            st.sidebar.write("DQE: ⚠️ Need orthogonal MTF")
    else:
        st.sidebar.write("DQE: ⚠️ Need MTF + NPS")

    with st.sidebar:
        st.markdown("---")
        st.subheader("RAW Endian Settings")
        st.checkbox(
            "Little-endian (global default)",
            key='raw_little_endian_default',
            help="Applied to RAW/STD parsing app-wide. DICOM Transfer Syntax UID (0002,0010) overrides when available."
        )
    
    st.sidebar.markdown("---")

    # --- MAIN PAGE: CATEGORY AND TEST SELECTION ---
    st.title("🔬 X-ray Image Analysis Toolkit")
    st.markdown("### Welcome! Select an analysis to get started")
    
    # Define analysis categories and their tests
    analysis_catalog = {
        "Flat Panel QA": {
            "icon": "📊",
            "description": "Quality assessment tests for flat panel detectors",
            "tests": {
                "Detector Response Curve": {
                    "description": "Calibrate detector pixel values to radiation dose",
                    "files_needed": "3+ RAW images at different exposures",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Multiple RAW files", "Different exposure levels", "Same detector settings"],
                    "computation": "Uses least-squares fitting to establish: $MPV = f(K_{air})$ and $EI = f(K_{air})$. Enables conversion from detector units to air kerma for subsequent analyses.\n\n Provides detailed noise components analysis by Weighted Robust Linear Models to separate quantum noise, electronic noise, and structural noise. Forces parameters to be non-negative.",
                    "icon": "📈"
                },
                "Uniformity Analysis": {
                    "description": "Measure spatial uniformity across detector area",
                    "files_needed": "1 flat-field image",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "No phantom objects", "Covers full detector area"],
                    "computation": "Calculates uniformity metrics within a central 80% area ROI using a sliding 30mm × 30mm window.\n\n**MPV Global Uniformity:**\n$$U_{global} = max\left(\\frac{|MPV_{ij}-MVP|}{MPV}\\right)$$\n\n**MPV Local Uniformity:**\n$$U_{local} = max\left(\\frac{|MPV_{ij}-MVP_{8n}|}{MPV_{8n}}\\right)$$\n\n**SNR Global Uniformity:**\n$$SNR_{global} = max\left(\\frac{|SNR_{ij}-SNR|}{SNR}\\right)$$\n\n**SNR Local Uniformity:**\n$$SNR_{local} = max\left(\\frac{|SNR_{ij}-SNR_{8n}|}{SNR_{8n}}\\right)$$",
                    "icon": "🔲"
                },
                "MTF (Sharpness)": {
                    "description": "Measure spatial resolution and sharpness",
                    "files_needed": "1-2 images with edge/slit phantom",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Edge or slit phantom", "Sharp edge visible", "2 orthogonal edges for DQE analysis"],
                    "computation": "IEC 62220-1-1:2015 slanted edge method. Uses Hough transform for edge detection, computes Edge Spread Function (ESF), differentiates to get Line Spread Function: $LSF = \\frac{dESF}{dx}$, then Fourier transform: $MTF(f) = |\\mathcal{F}\\{LSF\\}|$. Reports $MTF_{10\\%}$ and $MTF_{50\\%}$ as key metrics.",
                    "icon": "📐"
                },
                "NPS (Noise)": {
                    "description": "Characterize noise power spectrum",
                    "files_needed": "1+ uniform images",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "Multiple images recommended", "Known air kerma value"],
                    "computation": "IEC 62220-1-1:2015 standard. Extracts $N \\times N$ ROIs, applies 2D FFT to each ROI, averages power spectra. Normalizes: $$NPS(u,v) = \\frac{\\Delta x \\Delta y}{N_x N_y} |FFT|^2$$ Computes radial profile and integrates for total noise.",
                    "icon": "📡"
                },
                "DQE (Detective Quantum Efficiency)": {
                    "description": "Calculate overall detector quality metric",
                    "files_needed": "Requires MTF + NPS results",
                    "file_types": "Uses cached results",
                    "requirements": ["MTF from orthogonal edges", "NPS computed", "Known air kerma value"],
                    "computation": "IEC 62220-1-1:2015 formula: $$DQE(f) = \\frac{MTF^2(f)}{NPS(f) \\cdot K_{air}}$$ \n\n Where: \n\n- $\\text{MTF}(f)$ = Modulation Transfer Function (geometric mean of orthogonal edges) \n\n - $\\text{NNPS}(f)$ = Normalized Noise Power Spectrum (radial average) \n\n - $K_{air}$ = Air kerma in μGy \n\n - $f$ = Spatial frequency in lp/mm",
                    "icon": "🎯"
                },
                "Threshold Contrast": {
                    "description": "Measure low-contrast detectability",
                    "files_needed": "1 image with contrast phantom",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "No phantom objects", "Covers full detector area"],
                    "computation": "Computes Threshold-Contrast Detail Detectability by statistical analysis, based on Paruccini et al. (2021, https://doi.org/10.1016/j.ejmp.2021.10.007) report.",
                    "icon": "🎭"
                }
            }
        },
        "Developer Tools": {
            "icon": "🛠️",
            "description": "File conversion and comparison utilities",
            "tests": {
                "Convert to DICOM": {
                    "description": "Convert RAW/image files to DICOM format",
                    "files_needed": "1 image file",
                    "file_types": "RAW, STD, or any image format",
                    "requirements": ["Image file to convert", "Pixel spacing (optional)", "Custom metadata (optional)"],
                    "computation": "Creates DICOM-compliant file using pydicom library. Embeds pixel data, image dimensions, and metadata. Sets PresentationIntentType='FOR PROCESSING' tag. Generates SOP Instance UID and dataset identifiers per DICOM standard.",
                    "icon": "🏥"
                },
                "RAW vs DICOM Comparison": {
                    "description": "Compare RAW and DICOM versions of same image",
                    "files_needed": "2 files (1 RAW + 1 DICOM)",
                    "file_types": "1 RAW + 1 DICOM file",
                    "requirements": ["Same image in both formats", "RAW parameters known", "DICOM has metadata"],
                    "computation": "Pixel-by-pixel comparison between RAW and DICOM arrays. Computes difference map, calculates statistics (mean, max, RMSE). Visualizes discrepancies via difference histogram and spatial difference map. Validates data integrity post-conversion.",
                    "icon": "⚖️"
                }
            }
        }
    }
    
    # Step 1: Category Selection
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            f"{analysis_catalog['Flat Panel QA']['icon']} **Flat Panel QA**\n\n{analysis_catalog['Flat Panel QA']['description']}", 
            use_container_width=True,
            type="primary" if st.session_state['selected_category'] == "Flat Panel QA" else "secondary"
        ):
            st.session_state['selected_category'] = "Flat Panel QA"
            st.session_state['selected_test'] = None  # Reset test selection
            st.rerun()
    
    with col2:
        if st.button(
            f"{analysis_catalog['Developer Tools']['icon']} **Developer Tools**\n\n{analysis_catalog['Developer Tools']['description']}", 
            use_container_width=True,
            type="primary" if st.session_state['selected_category'] == "Developer Tools" else "secondary"
        ):
            st.session_state['selected_category'] = "Developer Tools"
            st.session_state['selected_test'] = None  # Reset test selection
            st.rerun()
    
    # Step 2: Test Selection (only shown after category is selected)
    if st.session_state['selected_category']:
        st.markdown("---")
        st.subheader(f"{analysis_catalog[st.session_state['selected_category']]['icon']} {st.session_state['selected_category']}")
        st.markdown("**Select a test:**")
        
        tests = analysis_catalog[st.session_state['selected_category']]['tests']
        
        # Create test selection buttons in a grid
        num_tests = len(tests)
        cols_per_row = 3 if num_tests > 3 else num_tests
        
        test_names = list(tests.keys())
        for i in range(0, len(test_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(test_names):
                    test_name = test_names[i + j]
                    test_info = tests[test_name]
                    with col:
                        if st.button(
                            f"{test_info['icon']} **{test_name}**", 
                            use_container_width=True,
                            type="primary" if st.session_state['selected_test'] == test_name else "secondary"
                        ):
                            st.session_state['selected_test'] = test_name
                            st.rerun()
        
        # Step 3: Show Requirements and Upload Section (only after test is selected)
        if st.session_state['selected_test']:
            selected_test_info = tests[st.session_state['selected_test']]
            
            st.markdown("---")
            st.subheader(f"{selected_test_info['icon']} {st.session_state['selected_test']}")
            st.markdown(f"*{selected_test_info['description']}*")
            
            # Requirements and Computation Card
            requirements_list = '\n'.join([f"- {req}" for req in selected_test_info['requirements']])
            computation_text = f"""\n\n**Computation:**\n\n{selected_test_info['computation']}""" if 'computation' in selected_test_info else ""
            
            st.info(f"""**📋 Requirements:**
- **Files needed:** {selected_test_info['files_needed']}
- **File types:** {selected_test_info['file_types']}

**Details:**

{requirements_list}{computation_text}
""")
            
            # Now show the upload section
            st.markdown("---")
            st.markdown("### 📤 Upload Files")
            
            # Dynamic file uploader based on selected test
            uploaded_files = st.file_uploader(
                f"Upload files for {st.session_state['selected_test']}",
                type=None,
                accept_multiple_files=True,
                help=f"Upload: {selected_test_info['file_types']}",
                key=f"uploader_{st.session_state['selected_test']}"
            )
            
            # Process files if uploaded
            if uploaded_files:
                process_analysis_workflow(
                    uploaded_files, 
                    st.session_state['selected_category'],
                    st.session_state['selected_test'],
                    analysis_catalog
                )
            else:
                st.warning("⬆️ Please upload the required files to continue")
    
    else:
        # No category selected yet - show welcome message
        st.info("👆 Select a category above to begin")

    # --- Add a clear session state button for debugging ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Saved Analysis Data"):
        st.session_state['detector_conversion'] = {
            'fit': {},
            'results': None,
            'ei_fit': {},
            'sd2_fit': {},
        }
        # Clear MTF and NPS caches
        if 'mtf_cache' in st.session_state:
            del st.session_state['mtf_cache']
        if 'nps_cache' in st.session_state:
            del st.session_state['nps_cache']
        st.success("All saved analysis data cleared!")
        st.rerun() # Rerun to reflect the cleared state


def process_analysis_workflow(uploaded_files, category, test_name, analysis_catalog):
    """
    Process the uploaded files and route to appropriate analysis function.
    This replaces the old monolithic file processing logic.
    """
    
    # Determine file types by content
    detected_file_types = []
    for f in uploaded_files:
        file_bytes = f.getvalue()
        detected_type = detect_file_type(file_bytes, f.name)
        detected_file_types.append((f, detected_type))

    # Allow users to override file type classification before validation
    with st.sidebar:
        st.markdown("### 🧭 File Type Overrides")
        st.caption("Adjust how each uploaded file is interpreted before test requirements are checked.")

    override_options = ["RAW/STD", "DICOM", "Unknown"]
    detected_to_ui = {'raw': "RAW/STD", 'dicom': "DICOM", 'unknown': "Unknown"}
    ui_to_internal = {"RAW/STD": 'raw', "DICOM": 'dicom', "Unknown": 'unknown'}

    file_types = []
    for idx, (f, detected_type) in enumerate(detected_file_types):
        default_ui = detected_to_ui.get(detected_type, "Unknown")
        with st.sidebar:
            selected_ui = st.selectbox(
                f"{f.name}",
                options=override_options,
                index=override_options.index(default_ui),
                key=f"file_type_override_{test_name}_{idx}_{f.name}"
            )
        file_types.append((f, ui_to_internal[selected_ui], detected_type))

    preloaded_payloads = _build_preloaded_payloads(file_types)
    
    # Categorize files
    dicom_files = [f for f, ftype, _ in file_types if ftype == 'dicom']
    raw_files = [f for f, ftype, _ in file_types if ftype == 'raw']
    unknown_files = [f for f, ftype, _ in file_types if ftype == 'unknown']
    raw_payloads = [p for p in preloaded_payloads if p['file_type'] == 'raw']

    shared_raw_params = _build_shared_raw_params(raw_payloads, context_key=f"{test_name}_") if raw_payloads else None
    st.session_state['shared_raw_params_current_test'] = shared_raw_params
    
    # Show file detection info in expander
    with st.expander("📁 Uploaded Files", expanded=False):
        for f, ftype, detected_type in file_types:
            is_overridden = ftype != detected_type
            source_note = "manual override" if is_overridden else "auto-detected"

            if ftype == 'dicom':
                st.success(f"🏥 {f.name} → DICOM ({source_note})")
            elif ftype == 'raw':
                ext = os.path.splitext(f.name)[1].lower()
                if ext in ['.dcm', '.dicom']:
                    st.info(f"📷 {f.name} → RAW ({source_note})")
                elif ext:
                    st.info(f"📷 {f.name} → RAW/STD ({source_note})")
                else:
                    st.info(f"📷 {f.name} → RAW (extensionless, {source_note})")
            else:
                st.warning(f"❓ {f.name} → Unknown format ({source_note})")
    
    # Show warning for unknown files
    if unknown_files:
        st.error(f"⚠️ Unknown file types detected: {[f.name for f in unknown_files]}")
        return
    
    # Route to appropriate analysis based on test type
    if test_name == "RAW vs DICOM Comparison":
        # Special case: comparison tool handles mixed files
        st.markdown("---")
        for idx, p in enumerate(preloaded_payloads):
            _ensure_payload_loaded(p, show_status=False)
        display_comparison_tool_section(preloaded_payloads)
        return
    
    elif test_name == "Convert to DICOM":
        # DICOM conversion: needs 1 file (RAW or image)
        if len(uploaded_files) != 1:
            st.warning(f"⚠️ Please upload exactly 1 file for DICOM conversion (currently {len(uploaded_files)} uploaded)")
            return
        
        # Load the image first
        image_array, pixel_spacing_row, pixel_spacing_col, dicom_filename = load_single_image(
            preloaded_payloads[0], file_types[0][1]
        )
        
        if image_array is not None:
            st.markdown("---")
            render_metadata_summary(
                image_array,
                pixel_spacing_row,
                pixel_spacing_col,
                domain='pixel',
                filename=dicom_filename,
                title='🖼️ Image Metadata Summary',
            )
            
            # Show image preview
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Normalize for display
                img_display = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                st.image(img_display, caption="Image Preview", use_container_width=True)
            
            st.markdown("---")
            st.subheader("🏥 Generate DICOM File")
            
            if st.button("Generate DICOM", key="dicomize_button", type="primary", use_container_width=True):
                try:
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
        return
    
    elif test_name == "Detector Response Curve":
        # Detector conversion: needs multiple RAW files
        if len(raw_files) < 3:
            st.warning(f"⚠️ Detector Response Curve requires at least 3 RAW files (currently {len(raw_files)} uploaded)")
            return

        # Show the same RAW/DICOM interpretation metadata controls used by other analyses
        # (based on first uploaded file) so users can see file type, dimensions, and pixel spacing.
        ref_file = raw_payloads[0]
        ref_image, ref_ps_row, ref_ps_col, ref_name = load_single_image(
            ref_file,
            'raw',
            show_status=False,
            shared_raw_params=shared_raw_params,
        )
        if ref_image is None:
            st.error("❌ Failed to load reference image for preview. Please check RAW parameters.")
            return

        # Same image preview tool used by other APIs
        with st.expander("🖼️ Image Preview", expanded=False):
            render_metadata_summary(
                ref_image,
                ref_ps_row,
                ref_ps_col,
                domain='pixel',
                filename=ref_name,
                title='🖼️ Image Metadata Summary',
            )

            # Normalize for display
            img_display = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
            st.image(img_display, caption=f"Preview: {ref_name}", use_container_width=True)
        
        st.markdown("---")
        detector_results = display_detector_conversion_section(uploaded_files=raw_payloads)
        if detector_results is not None:
            _dc_state = st.session_state.get('detector_conversion')
            if not isinstance(_dc_state, dict):
                _dc_state = {'fit': {}, 'results': None, 'ei_fit': {}, 'sd2_fit': {}}
            _dc_state['results'] = detector_results
            st.session_state['detector_conversion'] = _dc_state
        return
    
    # For all other tests: validate file type consistency
    is_dicom_upload = len(dicom_files) > 0
    is_raw_upload = len(raw_files) > 0
    
    if is_dicom_upload and is_raw_upload:
        st.error("⚠️ Mixed file types detected! Please upload only one type (either all RAW or all DICOM) for this analysis.")
        return
    
    # Load the primary image for single-image analyses
    if not uploaded_files:
        return
    
    # Load first image
    primary_file = preloaded_payloads[0]
    primary_file_type = file_types[0][1]
    
    image_array, pixel_spacing_row, pixel_spacing_col, dicom_filename = load_single_image(
        primary_file, primary_file_type, shared_raw_params=shared_raw_params
    )
    
    if image_array is None:
        st.error("❌ Failed to load image. Please check the file format and parameters.")
        return
    
    # Image preview in expander
    with st.expander("🖼️ Image Preview", expanded=False):
        render_metadata_summary(
            image_array,
            pixel_spacing_row,
            pixel_spacing_col,
            domain='pixel',
            filename=dicom_filename,
            title='🖼️ Image Metadata Summary',
        )
        
        # Normalize for display
        img_display = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        st.image(img_display, caption=f"Preview: {dicom_filename}", use_container_width=True)
     
    # Route to specific analysis
    st.markdown("---")
    
    if test_name == "Uniformity Analysis":
        display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
    
    elif test_name == "MTF (Sharpness)":
        # Pass uploaded files to enable MTF comparison mode when 2 images are uploaded
        display_mtf_analysis_section(
            image_array,
            pixel_spacing_row,
            pixel_spacing_col,
            preloaded_files=preloaded_payloads,
        )
    
    elif test_name == "NPS (Noise)":
        for idx, p in enumerate(preloaded_payloads):
            _ensure_payload_loaded(p, show_status=False)
        display_nps_analysis_section(
            image_array,
            pixel_spacing_row,
            pixel_spacing_col,
            preloaded_files=preloaded_payloads,
        )
    
    elif test_name == "Threshold Contrast":
        display_threshold_contrast_section(image_array, pixel_spacing_row, pixel_spacing_col)
    
    elif test_name == "DQE (Detective Quantum Efficiency)":
        display_dqe_analysis_section()


def load_single_image(uploaded_file, file_type, show_status=True, shared_raw_params=None):
    """
    Load a single image file (DICOM or RAW) and return the image array and metadata.
    Returns: (image_array, pixel_spacing_row, pixel_spacing_col, filename)
    """
    
    image_array = None
    pixel_spacing_row = None
    pixel_spacing_col = None
    filename, file_bytes = file_name_and_bytes(uploaded_file)

    if file_type == 'dicom':
        # Load as DICOM
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            
            # Extract dimensions from DICOM tags (0028,0010) Rows and (0028,0011) Columns
            rows = getattr(ds, 'Rows', None)
            cols = getattr(ds, 'Columns', None)
            
            # Extract pixel spacing - prefer ImagerPixelSpacing (0018,1164) for detector data
            # Fallback to PixelSpacing (0028,0030) if ImagerPixelSpacing not available
            ps = getattr(ds, 'ImagerPixelSpacing', None)
            if ps is None:
                ps = getattr(ds, 'PixelSpacing', None)
            
            if ps and len(ps) >= 2:
                pixel_spacing_row = float(ps[0])
                pixel_spacing_col = float(ps[1])
            
            # Load pixel data - pydicom uses Rows/Columns internally
            image_array = ds.pixel_array
            
            # Verify dimensions match
            if rows and cols:
                if image_array.shape != (rows, cols):
                    st.warning(f"⚠️ Dimension mismatch: Tags say {cols}×{rows}, array is {image_array.shape[1]}×{image_array.shape[0]}")
            
            if show_status:
                st.success(f"✅ Loaded DICOM file: {filename}")
        except Exception as e:
            st.error(f"❌ Failed to load DICOM: {e}")
            return None, None, None, None
    
    elif file_type == 'raw':
        if not isinstance(shared_raw_params, dict):
            st.error("Strict ingestion mode: shared RAW parameters are required before decoding RAW files.")
            return None, None, None, None

        np_dtype = np.dtype(shared_raw_params['dtype'])
        width = int(shared_raw_params['width'])
        height = int(shared_raw_params['height'])
        pixel_spacing_row = float(shared_raw_params['pixel_spacing_row'])
        pixel_spacing_col = float(shared_raw_params['pixel_spacing_col'])

        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            if hasattr(ds, 'pixel_array'):
                image_array = ds.pixel_array
                if show_status:
                    st.success("✅ RAW file loaded (pixel data extracted from DICOM structure)")
            else:
                arr, endian_used, endian_source = frombuffer_with_endian(
                    file_bytes,
                    np_dtype,
                    default_little_endian=bool(st.session_state.get('raw_little_endian_default', True)),
                    auto_endian_from_dicom=True,
                )
                image_array = arr.reshape((height, width))
                if show_status:
                    st.success("✅ RAW file loaded successfully")
                    st.caption(f"Endian used: {'little' if endian_used else 'big'} ({endian_source})")
        except Exception:
            try:
                arr, endian_used, endian_source = frombuffer_with_endian(
                    file_bytes,
                    np_dtype,
                    default_little_endian=bool(st.session_state.get('raw_little_endian_default', True)),
                    auto_endian_from_dicom=True,
                )
                image_array = arr.reshape((height, width))
                if show_status:
                    st.success("✅ RAW file loaded successfully")
                    st.caption(f"Endian used: {'little' if endian_used else 'big'} ({endian_source})")
            except Exception as e:
                st.error(f"❌ Failed to load RAW: {e}")
                return None, None, None, None

        return image_array, pixel_spacing_row, pixel_spacing_col, filename
    
    return image_array, pixel_spacing_row, pixel_spacing_col, filename


if __name__ == "__main__":
    
    # Run the main Streamlit UI
    main_app_ui()
