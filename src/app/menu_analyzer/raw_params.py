import io

import numpy as np
import pydicom
import streamlit as st

from src.core.io.analysis_payload import file_name_and_bytes


def _extract_dicom_pixel_dtype_hint(file_bytes: bytes):
    """Infer RAW pixel dtype from embedded DICOM Pixel Data metadata when available.

    Returns:
        tuple[str | None, str | None]: (dtype_hint, source)
        dtype_hint in {'uint8', 'uint16', 'float32'}
    """
    try:
        ds = pydicom.dcmread(
            io.BytesIO(file_bytes),
            force=True,
            stop_before_pixels=False,
            defer_size="1 KB",
        )
    except Exception:
        return None, None

    pixel_data_elem = ds.get((0x7FE0, 0x0010))
    if pixel_data_elem is None:
        return None, None

    has_float_pixel_data = ds.get((0x7FE0, 0x0008)) is not None
    has_double_float_pixel_data = ds.get((0x7FE0, 0x0009)) is not None

    bits_allocated = getattr(ds, 'BitsAllocated', None)
    if bits_allocated is not None:
        try:
            bits_allocated = int(bits_allocated)
        except (TypeError, ValueError):
            bits_allocated = None

    if bits_allocated == 8:
        return 'uint8', 'Pixel Data + BitsAllocated=8'
    if bits_allocated == 16:
        return 'uint16', 'Pixel Data + BitsAllocated=16'
    if bits_allocated == 32 and (has_float_pixel_data or has_double_float_pixel_data):
        return 'float32', 'Float Pixel Data + BitsAllocated=32'

    vr = str(getattr(pixel_data_elem, 'VR', '')).upper()
    if vr == 'OB':
        return 'uint8', 'Pixel Data VR=OB'
    if vr == 'OW':
        return 'uint16', 'Pixel Data VR=OW'
    if vr == 'OF':
        return 'float32', 'Pixel Data VR=OF'

    return None, None


def build_shared_raw_params(raw_payloads, context_key=""):
    """Render one shared RAW parameter section and return parameters for all RAW files."""
    if not raw_payloads:
        return None

    ref_name, ref_bytes = file_name_and_bytes(raw_payloads[0])

    dicom_rows = None
    dicom_cols = None
    dicom_ps_row = None
    dicom_ps_col = None
    dtype_hints = []
    dtype_hint_sources = set()

    for raw_file in raw_payloads:
        _, raw_bytes = file_name_and_bytes(raw_file)
        hint, source = _extract_dicom_pixel_dtype_hint(raw_bytes)
        if hint in {'uint8', 'uint16', 'float32'}:
            dtype_hints.append(hint)
        if source:
            dtype_hint_sources.add(source)

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
            st.info("📋 DICOM metadata found")
        dtype_map = {
            '16-bit Unsigned Integer': np.uint16,
            '8-bit Unsigned Integer': np.uint8,
            '32-bit Float': np.float32
        }
        hint_to_label = {
            'uint8': '8-bit Unsigned Integer',
            'uint16': '16-bit Unsigned Integer',
            'float32': '32-bit Float',
        }

        default_dtype_label = '16-bit Unsigned Integer'
        if dtype_hints:
            unique_dtype_hints = sorted(set(dtype_hints))
            if len(unique_dtype_hints) == 1:
                default_dtype_label = hint_to_label.get(unique_dtype_hints[0], default_dtype_label)
            else:
                st.warning(
                    "Conflicting Pixel Data dtype hints detected across uploaded RAW files. "
                    "Using fallback default: 16-bit Unsigned Integer."
                )

        if dtype_hint_sources:
            st.caption(
                "Pixel dtype auto-preset from DICOM Pixel Data (7FE0,0010): "
                + ", ".join(sorted(dtype_hint_sources))
            )

        dtype_str = st.selectbox(
            "Pixel Data Type",
            options=list(dtype_map.keys()),
            index=list(dtype_map.keys()).index(default_dtype_label),
            key=f"{context_key}dtype_shared"
        )
        np_dtype = dtype_map[dtype_str]

        itemsize = np.dtype(np_dtype).itemsize
        file_size = len(ref_bytes)

        if file_size < itemsize:
            st.error(f"File size ({file_size} bytes) is too small for pixel size ({itemsize} bytes)")
            return None

        full_pixels = file_size // itemsize
        remainder_bytes = file_size % itemsize
        if remainder_bytes:
            st.warning(
                f"File has {remainder_bytes} trailing byte(s) not aligned with pixel size; "
                f"they are excluded from auto-dimension candidates."
            )

        total_pixels = full_pixels

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

        if dicom_rows and dicom_cols:
            auto_width = int(dicom_cols)
            auto_height = int(dicom_rows)
            st.write("**Dimensions (from DICOM tags):**")
            st.write(f"{auto_width} × {auto_height} pixels")
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

            auto_width, auto_height = map(int, selected_dim.split(" x "))
            st.caption(f"Auto-detected: **{auto_width} x {auto_height}** pixels ({auto_width * auto_height:,} total)")

        manual_dims = st.checkbox(
            "Manually override dimensions",
            value=False,
            key=f"{context_key}manual_dims_shared",
            help="Enable to set width/height manually instead of using auto-detected values."
        )

        if manual_dims:
            width = int(st.number_input(
                "Manual width (px)",
                min_value=1,
                value=int(auto_width),
                step=1,
                key=f"{context_key}manual_width_shared",
            ))
            height = int(st.number_input(
                "Manual height (px)",
                min_value=1,
                value=int(auto_height),
                step=1,
                key=f"{context_key}manual_height_shared",
            ))
        else:
            width, height = int(auto_width), int(auto_height)

        expected_bytes = int(width) * int(height) * itemsize
        if expected_bytes > file_size:
            st.error(f"Selected dimensions require {expected_bytes:,} bytes, but file has {file_size:,} bytes")
            return None

        extra_bytes = file_size - expected_bytes
        if extra_bytes > 0:
            st.warning(f"Detected {extra_bytes:,} extra byte(s) beyond selected image size")
        else:
            st.success("Selected dimensions exactly match file size")

        if manual_dims:
            if extra_bytes > 0:
                extra_bytes_location_ui = st.selectbox(
                    "Extra bytes location",
                    options=["Header (skip first bytes)", "Trailer (skip last bytes)"],
                    index=0,
                    key=f"{context_key}extra_bytes_location_shared",
                    help="Choose whether extra bytes are a header (prefix) or trailer (suffix)."
                )
                extra_bytes_location = 'start' if extra_bytes_location_ui.startswith("Header") else 'end'

                skip_extra_bytes = int(st.number_input(
                    "Bytes to skip",
                    min_value=0,
                    max_value=int(extra_bytes),
                    value=int(extra_bytes),
                    step=int(itemsize),
                    key=f"{context_key}skip_extra_bytes_shared",
                    help="Usually set to all extra bytes; keep default unless format details say otherwise."
                ))
            else:
                extra_bytes_location = 'start'
                skip_extra_bytes = 0
        else:
            extra_bytes_location = 'start'
            skip_extra_bytes = 0

        if skip_extra_bytes % itemsize != 0:
            st.error(f"Bytes to skip ({skip_extra_bytes}) must be a multiple of pixel size ({itemsize})")
            return None

    return {
        'dtype': np_dtype,
        'width': int(width),
        'height': int(height),
        'skip_extra_bytes': int(skip_extra_bytes),
        'extra_bytes_location': extra_bytes_location,
        'pixel_spacing_row': float(pixel_spacing_row),
        'pixel_spacing_col': float(pixel_spacing_col),
    }
