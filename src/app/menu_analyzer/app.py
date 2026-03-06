import streamlit as st

st.set_page_config(page_title="X-ray Image Analysis Toolkit", layout="wide")

from src.qa.flat_panel_qa.dqe import display_dqe_analysis_section
from src.app.menu_analyzer.catalog import get_analysis_catalog
from src.app.menu_analyzer.workflow import process_analysis_workflow


def main_app_ui():
    if 'detector_conversion' not in st.session_state:
        st.session_state['detector_conversion'] = {
            'fit': {},
            'results': None,
            'ei_fit': {},
            'sd2_fit': {},
        }

    if 'selected_category' not in st.session_state:
        st.session_state['selected_category'] = None
    if 'selected_test' not in st.session_state:
        st.session_state['selected_test'] = None

    if 'raw_little_endian_default' not in st.session_state:
        st.session_state['raw_little_endian_default'] = True

    st.sidebar.markdown("---")
    st.sidebar.subheader("Saved Analysis Data Status")

    _dc = st.session_state['detector_conversion']
    _fit = _dc.get('fit', {}) if isinstance(_dc, dict) else {}
    _dc_loaded = isinstance(_fit, dict) and (_fit.get('predict_mpv') is not None)
    st.sidebar.write(f"Detector Conversion: {'✅ Cached' if _dc_loaded else '⚠️ Missing'}")
    _dc_struct = _dc.get('results') if isinstance(_dc, dict) else None
    if isinstance(_dc_struct, dict) and isinstance(_dc_struct.get('files'), list):
        st.sidebar.write(f"Detector files cached: {len(_dc_struct['files'])}")

    _mtf_cache = st.session_state.get('mtf_cache')
    if _mtf_cache and isinstance(_mtf_cache, dict):
        _mtf_results = _mtf_cache.get('results', [])
        _geom_mean = _mtf_cache.get('mtf_geometric_mean')
        if _geom_mean and _geom_mean.get('available'):
            st.sidebar.write(f"MTF: ✅ {len(_mtf_results)} measurements + Geometric mean")
        else:
            st.sidebar.write("MTF: ⚠️ Not enough measurements for geometric mean")
    else:
        st.sidebar.write("MTF: ⚠️ Not computed")

    _nps_cache = st.session_state.get('nps_cache')
    if _nps_cache and isinstance(_nps_cache, dict):
        _kerma = _nps_cache.get('kerma_ugy', 'N/A')
        st.sidebar.write(f"NPS: ✅ Cached (K = {_kerma:.2f} μGy)")
    else:
        st.sidebar.write("NPS: ⚠️ Not computed")

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

    st.title("🔬 X-ray Image Analysis Toolkit")
    st.markdown("### Welcome! Select an analysis to get started")

    analysis_catalog = get_analysis_catalog()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            f"{analysis_catalog['Flat Panel QA']['icon']} **Flat Panel QA**\n\n{analysis_catalog['Flat Panel QA']['description']}",
            use_container_width=True,
            type="primary" if st.session_state['selected_category'] == "Flat Panel QA" else "secondary"
        ):
            st.session_state['selected_category'] = "Flat Panel QA"
            st.session_state['selected_test'] = None
            st.rerun()

    with col2:
        if st.button(
            f"{analysis_catalog['Developer Tools']['icon']} **Developer Tools**\n\n{analysis_catalog['Developer Tools']['description']}",
            use_container_width=True,
            type="primary" if st.session_state['selected_category'] == "Developer Tools" else "secondary"
        ):
            st.session_state['selected_category'] = "Developer Tools"
            st.session_state['selected_test'] = None
            st.rerun()

    if st.session_state['selected_category']:
        st.markdown("---")
        st.subheader(f"{analysis_catalog[st.session_state['selected_category']]['icon']} {st.session_state['selected_category']}")
        st.markdown("**Select a test:**")

        tests = analysis_catalog[st.session_state['selected_category']]['tests']
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

        if st.session_state['selected_test']:
            selected_test_info = tests[st.session_state['selected_test']]

            st.markdown("---")
            st.subheader(f"{selected_test_info['icon']} {st.session_state['selected_test']}")
            st.markdown(f"*{selected_test_info['description']}*")

            requirements_list = '\n'.join([f"- {req}" for req in selected_test_info['requirements']])
            computation_text = f"""\n\n**Computation:**\n\n{selected_test_info['computation']}""" if 'computation' in selected_test_info else ""

            st.info(f"""**📋 Requirements:**
- **Files needed:** {selected_test_info['files_needed']}
- **File types:** {selected_test_info['file_types']}

**Details:**

{requirements_list}{computation_text}
""")

            if st.session_state['selected_test'] == "DQE (Detective Quantum Efficiency)":
                st.markdown("---")
                display_dqe_analysis_section()
            else:
                st.markdown("---")
                st.markdown("### 📤 Upload Files")

                uploaded_files = st.file_uploader(
                    f"Upload files for {st.session_state['selected_test']}",
                    type=None,
                    accept_multiple_files=True,
                    help=f"Upload: {selected_test_info['file_types']}",
                    key=f"uploader_{st.session_state['selected_test']}"
                )

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
        st.info("👆 Select a category above to begin")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear All Saved Analysis Data"):
        st.session_state['detector_conversion'] = {
            'fit': {},
            'results': None,
            'ei_fit': {},
            'sd2_fit': {},
        }
        if 'mtf_cache' in st.session_state:
            del st.session_state['mtf_cache']
        if 'nps_cache' in st.session_state:
            del st.session_state['nps_cache']
        st.success("All saved analysis data cleared!")
        st.rerun()


if __name__ == "__main__":
    main_app_ui()
