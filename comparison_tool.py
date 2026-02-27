import streamlit as st
import numpy as np
import pydicom
import os
import io

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


def _file_name_and_bytes(file_obj):
    """Return (filename, bytes) for Streamlit upload objects or preloaded payload dicts."""
    if isinstance(file_obj, dict):
        return file_obj.get('name', 'unknown'), file_obj.get('bytes', b'')
    return getattr(file_obj, 'name', 'unknown'), file_obj.getvalue()


def _file_name(file_obj):
    return _file_name_and_bytes(file_obj)[0]

def display_comparison_tool_section(preloaded_files):
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

    try:
        if isinstance(dicom_file, dict) and isinstance(dicom_file.get('image_array'), np.ndarray):
            image_array_dicom = dicom_file.get('image_array')
        else:
            _, dicom_bytes = _file_name_and_bytes(dicom_file)
            dicom_dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))
            try:
                if dicom_dataset.file_meta.TransferSyntaxUID.is_compressed:
                    dicom_dataset.decompress()
            except Exception:
                pass

            image_array_dicom = dicom_dataset.pixel_array
        st.write(f"**DICOM File:** `{_file_name(dicom_file)}`")
        st.write(f"&nbsp;&nbsp;&nbsp;- Dimensions: `{image_array_dicom.shape[1]}x{image_array_dicom.shape[0]}` | Data Type: `{image_array_dicom.dtype}`")
    except Exception as e:
        st.error(f"Error reading DICOM file `{_file_name(dicom_file)}`: {e}")
        return

    st.sidebar.subheader("RAW Image Parameters for Comparison")
    dtype_map = {'16-bit Unsigned Integer': np.uint16, '8-bit Unsigned Integer': np.uint8, '32-bit Float': np.float32}
    dtype_str = st.sidebar.selectbox("Pixel Data Type", options=list(dtype_map.keys()), index=0, key="raw_dtype_compare")
    np_dtype = dtype_map[dtype_str]
    
    raw_name, raw_data = _file_name_and_bytes(raw_file)
    itemsize = np.dtype(np_dtype).itemsize
    total_pixels = len(raw_data) // itemsize

    def get_factors(n):
        factors = set(); [factors.add((i, n//i)) or factors.add((n//i, i)) for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0]; return sorted(list(factors))
    
    possible_dims = get_factors(total_pixels)
    height, width = possible_dims[len(possible_dims) // 2]
    
    if isinstance(raw_file, dict) and isinstance(raw_file.get('image_array'), np.ndarray):
        image_array_raw = raw_file.get('image_array')
    else:
        raw_arr = np.frombuffer(raw_data, dtype=np_dtype)
        image_array_raw = raw_arr.reshape((height, width))
    st.write(f"**RAW File:** `{raw_name}`")
    st.write(f"&nbsp;&nbsp;&nbsp;- Assumed Dimensions: `{width}x{height}` | Data Type: `{np_dtype}`")

    display_array_comparison(image_array_dicom, image_array_raw)