import io

import numpy as np
import pydicom
import streamlit as st

from src.core.io.analysis_payload import ImagePayload, file_name_and_bytes
from src.core.io.raw_endian import frombuffer_with_endian


def build_preloaded_payloads(file_types) -> list[ImagePayload]:
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


def ensure_payload_loaded(payload: dict, show_status: bool = False) -> dict:
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
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

            rows = getattr(ds, 'Rows', None)
            cols = getattr(ds, 'Columns', None)

            ps = getattr(ds, 'ImagerPixelSpacing', None)
            if ps is None:
                ps = getattr(ds, 'PixelSpacing', None)

            if ps and len(ps) >= 2:
                pixel_spacing_row = float(ps[0])
                pixel_spacing_col = float(ps[1])

            image_array = ds.pixel_array

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
        skip_extra_bytes = int(shared_raw_params.get('skip_extra_bytes', 0))
        extra_bytes_location = str(shared_raw_params.get('extra_bytes_location', 'start'))
        pixel_spacing_row = float(shared_raw_params['pixel_spacing_row'])
        pixel_spacing_col = float(shared_raw_params['pixel_spacing_col'])

        expected_bytes = width * height * np_dtype.itemsize
        file_size = len(file_bytes)
        available_extra_bytes = file_size - expected_bytes
        if available_extra_bytes < 0:
            st.error(f"❌ File too small: requires {expected_bytes:,} bytes, found {file_size:,}")
            return None, None, None, None

        if skip_extra_bytes < 0 or skip_extra_bytes > available_extra_bytes:
            st.error(
                f"❌ Invalid bytes-to-skip ({skip_extra_bytes}); allowed range is 0..{available_extra_bytes}"
            )
            return None, None, None, None

        if skip_extra_bytes % np_dtype.itemsize != 0:
            st.error(
                f"❌ Bytes to skip ({skip_extra_bytes}) must be multiple of pixel size ({np_dtype.itemsize})"
            )
            return None, None, None, None

        if extra_bytes_location == 'end':
            parse_bytes = file_bytes[:file_size - skip_extra_bytes]
        else:
            parse_bytes = file_bytes[skip_extra_bytes:]

        try_dicom_decode = (skip_extra_bytes == 0 and extra_bytes_location == 'start')

        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True) if try_dicom_decode else None
            if ds is not None and hasattr(ds, 'pixel_array'):
                image_array = ds.pixel_array
                if show_status:
                    st.success("✅ RAW file loaded (pixel data extracted from DICOM structure)")
            else:
                arr, endian_used, endian_source = frombuffer_with_endian(
                    parse_bytes,
                    np_dtype,
                    default_little_endian=bool(st.session_state.get('raw_little_endian_default', True)),
                    auto_endian_from_dicom=True,
                )
                needed_pixels = width * height
                if arr.size < needed_pixels:
                    raise ValueError(f"Decoded pixels ({arr.size}) are fewer than required ({needed_pixels})")
                image_array = arr[:needed_pixels].reshape((height, width))
                if show_status:
                    st.success("✅ RAW file loaded successfully")
        except Exception:
            try:
                arr, endian_used, endian_source = frombuffer_with_endian(
                    parse_bytes,
                    np_dtype,
                    default_little_endian=bool(st.session_state.get('raw_little_endian_default', True)),
                    auto_endian_from_dicom=True,
                )
                needed_pixels = width * height
                if arr.size < needed_pixels:
                    raise ValueError(f"Decoded pixels ({arr.size}) are fewer than required ({needed_pixels})")
                image_array = arr[:needed_pixels].reshape((height, width))
                if show_status:
                    st.success("✅ RAW file loaded successfully")
            except Exception as e:
                st.error(f"❌ Failed to load RAW: {e}")
                return None, None, None, None

        return image_array, pixel_spacing_row, pixel_spacing_col, filename

    return image_array, pixel_spacing_row, pixel_spacing_col, filename
