import streamlit as st
import numpy as np
import os
from analysis_payload import ImagePayload, file_name_and_bytes
from metadata_summary import render_metadata_summary

# Note: dicom_utils removed. Use pydicom.pixel_array (stored values) directly.

def display_array_comparison(arr1, arr2, name1="DICOM-derived", name2="Original RAW"):
    """
    Compares two numpy arrays and displays the results in Streamlit.
    """
    st.subheader("Comparison Results")

    # 1. Shape
    shape_match = arr1.shape == arr2.shape
    if shape_match:
        st.success(f"✅ **Shape:** Both images have the same dimensions: `{arr1.shape}`")
    else:
        st.error(f"❌ **Shape:** Mismatch! `{name1}` is `{arr1.shape}`, but `{name2}` is `{arr2.shape}`.")
        st.info("Comparison cannot continue if shapes are different.")
        return

    # 2. Data Type
    dtype_match = arr1.dtype == arr2.dtype
    if dtype_match:
        st.success(f"✅ **Data Type:** Both images have the same data type: `{arr1.dtype}`")
    else:
        st.warning(f"⚠️ **Data Type:** Mismatch! `{name1}` is `{arr1.dtype}`, but `{name2}` is `{arr2.dtype}`. This can cause comparison failure even if values look similar.")

    # 3. Element-wise equality
    are_equal = np.array_equal(arr1, arr2)
    if are_equal:
        st.balloons()
        st.success("🎉 **Perfect Match!** The pixel data in both files is identical.")
    else:
        st.error("❌ **No Match:** The pixel data is NOT identical.")
        
        diff = arr1.astype(np.float64) - arr2.astype(np.float64)
        num_diff_pixels = np.count_nonzero(diff)
        
        st.subheader("Difference Diagnostics")
        st.write(f"**Number of differing pixels:** {num_diff_pixels} (out of {arr1.size})")
        st.write(f"**Maximum absolute difference:** {np.max(np.abs(diff)):.2f}")
        
        col1, col2 = st.columns(2)
        col1.metric(f"Mean Value ({name1})", f"{np.mean(arr1):.2f}")
        col2.metric(f"Mean Value ({name2})", f"{np.mean(arr2):.2f}")


def _file_name(file_obj):
    return file_name_and_bytes(file_obj)[0]

def display_comparison_tool_section(preloaded_files: list[ImagePayload]):
    """ Renders the comparison tool UI. """
    st.header("Developer Tool: Image Data Comparison")
    files = preloaded_files
    
    file_extensions = {os.path.splitext(_file_name(f))[1].lower() for f in files}
    is_raw_upload = '.raw' in file_extensions
    is_dicom_upload = '.dcm' in file_extensions or '.dicom' in file_extensions

    if not (len(files) == 2 and is_raw_upload and is_dicom_upload):
        st.warning("For comparison, please upload exactly one DICOM file and its corresponding RAW file.")
        return

    st.info("Pixel data from the uploaded DICOM and RAW files will be compared.")

    dicom_file = next((f for f in files if os.path.splitext(_file_name(f))[1].lower() in ['.dcm', '.dicom']))
    raw_file = next((f for f in files if os.path.splitext(_file_name(f))[1].lower() == '.raw'))

    if not isinstance(dicom_file, dict) or not isinstance(dicom_file.get('image_array'), np.ndarray):
        st.error(f"Strict ingestion mode: DICOM payload '{_file_name(dicom_file)}' is missing decoded image_array.")
        return
    if not isinstance(raw_file, dict) or not isinstance(raw_file.get('image_array'), np.ndarray):
        st.error(f"Strict ingestion mode: RAW payload '{_file_name(raw_file)}' is missing decoded image_array.")
        return

    image_array_dicom = dicom_file.get('image_array')
    image_array_raw = raw_file.get('image_array')

    with st.expander("Image Metadata Summary", expanded=False):
        st.markdown("**DICOM input**")
        render_metadata_summary(
            image_array_dicom,
            dicom_file.get('pixel_spacing_row'),
            dicom_file.get('pixel_spacing_col'),
            domain='pixel',
            filename=_file_name(dicom_file),
            title='Metadata',
        )
        st.markdown("---")
        st.markdown("**RAW input**")
        render_metadata_summary(
            image_array_raw,
            raw_file.get('pixel_spacing_row'),
            raw_file.get('pixel_spacing_col'),
            domain='pixel',
            filename=_file_name(raw_file),
            title='Metadata',
        )

    display_array_comparison(image_array_dicom, image_array_raw)