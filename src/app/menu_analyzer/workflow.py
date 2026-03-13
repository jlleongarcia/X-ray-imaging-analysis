import os

import streamlit as st

from src.qa.flat_panel_qa.detector_conversion import display_detector_conversion_section
from src.qa.flat_panel_qa.dqe import display_dqe_analysis_section
from src.qa.flat_panel_qa.mtf import display_mtf_analysis_section
from src.qa.flat_panel_qa.nps import display_nps_analysis_section
from src.qa.flat_panel_qa.threshold_contrast import display_threshold_contrast_section
from src.qa.flat_panel_qa.uniformity import display_uniformity_analysis_section
from src.qa.dicom_analysis.postprocessing_constancy import display_dicom_postprocessing_analysis_section
from src.tools.developer_tools.comparison_tool import display_comparison_tool_section
from src.tools.developer_tools.dicomizer import generate_dicom_from_raw

from .file_detection import detect_file_type
from .image_loader import build_preloaded_payloads, ensure_payload_loaded, load_single_image
from .raw_params import build_shared_raw_params


def process_analysis_workflow(uploaded_files, category, test_name, analysis_catalog):
    """
    Process the uploaded files and route to appropriate analysis function.
    This replaces the old monolithic file processing logic.
    """

    if test_name == "Detective Quantum Efficiency (DQE)":
        st.markdown("---")
        display_dqe_analysis_section()
        return

    detected_file_types = []
    for f in uploaded_files:
        file_bytes = f.getvalue()
        detected_type = detect_file_type(file_bytes, f.name)
        detected_file_types.append((f, detected_type))

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

    preloaded_payloads = build_preloaded_payloads(file_types)

    dicom_files = [f for f, ftype, _ in file_types if ftype == 'dicom']
    raw_files = [f for f, ftype, _ in file_types if ftype == 'raw']
    unknown_files = [f for f, ftype, _ in file_types if ftype == 'unknown']
    raw_payloads = [p for p in preloaded_payloads if p['file_type'] == 'raw']

    shared_raw_params = build_shared_raw_params(raw_payloads, context_key=f"{test_name}_") if raw_payloads else None
    st.session_state['shared_raw_params_current_test'] = shared_raw_params

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

    if unknown_files:
        st.error(f"⚠️ Unknown file types detected: {[f.name for f in unknown_files]}")
        return

    if test_name == "DICOM Post-processing Analysis":
        if len(dicom_files) == 0:
            st.warning("⚠️ Please upload at least one DICOM file for this analysis")
            return

        if len(raw_files) > 0:
            st.error("⚠️ This analysis only accepts DICOM files")
            return

        for p in preloaded_payloads:
            ensure_payload_loaded(p, show_status=False)
        display_dicom_postprocessing_analysis_section(preloaded_payloads)
        return

    if test_name == "RAW vs DICOM Comparison":
        st.markdown("---")
        for p in preloaded_payloads:
            ensure_payload_loaded(p, show_status=False)
        display_comparison_tool_section(preloaded_payloads)
        return

    elif test_name == "Convert to DICOM":
        if len(uploaded_files) != 1:
            st.warning(f"⚠️ Please upload exactly 1 file for DICOM conversion (currently {len(uploaded_files)} uploaded)")
            return

        image_array, pixel_spacing_row, pixel_spacing_col, dicom_filename = load_single_image(
            preloaded_payloads[0], file_types[0][1]
        )

        if image_array is not None:
            st.markdown("---")
            with st.expander("🖼️ Image Preview", expanded=False):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
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
        if len(raw_files) < 3:
            st.warning(f"⚠️ Detector Response Curve requires at least 3 RAW files (currently {len(raw_files)} uploaded)")
            return

        # Use shared RAW decoding path before preview/analysis.
        for payload in raw_payloads:
            ensure_payload_loaded(payload, show_status=False)

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

        with st.expander("🖼️ Image Preview", expanded=False):
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

    is_dicom_upload = len(dicom_files) > 0
    is_raw_upload = len(raw_files) > 0

    if is_dicom_upload and is_raw_upload:
        st.error("⚠️ Mixed file types detected! Please upload only one type (either all RAW or all DICOM) for this analysis.")
        return

    if not uploaded_files:
        return

    primary_file = preloaded_payloads[0]
    primary_file_type = file_types[0][1]

    image_array, pixel_spacing_row, pixel_spacing_col, dicom_filename = load_single_image(
        primary_file, primary_file_type, shared_raw_params=shared_raw_params
    )

    if image_array is None:
        st.error("❌ Failed to load image. Please check the file format and parameters.")
        return

    with st.expander("🖼️ Image Preview", expanded=False):
        img_display = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        st.image(img_display, caption=f"Preview: {dicom_filename}", use_container_width=True)

    st.markdown("---")

    if test_name == "Uniformity":
        display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)

    elif test_name == "Modulation Transfer Function (MTF)":
        for p in preloaded_payloads:
            ensure_payload_loaded(p, show_status=False)
        display_mtf_analysis_section(
            image_array,
            pixel_spacing_row,
            pixel_spacing_col,
            preloaded_files=preloaded_payloads,
        )

    elif test_name == "Noise Power Spectrum (NPS)":
        for p in preloaded_payloads:
            ensure_payload_loaded(p, show_status=False)
        display_nps_analysis_section(
            image_array,
            pixel_spacing_row,
            pixel_spacing_col,
            preloaded_files=preloaded_payloads,
        )

    elif test_name == "Threshold Contrast Detail Detectability (TCDD)":
        display_threshold_contrast_section(image_array, pixel_spacing_row, pixel_spacing_col)
