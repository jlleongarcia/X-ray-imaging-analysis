from typing import Literal, TypedDict

import numpy as np


class ImagePayload(TypedDict, total=False):
    """Standard preloaded payload schema used across analysis modules.

    Fields:
    - name: original filename
    - bytes: file bytes read once at upload time
    - file_type: user-selected/overridden interpretation ('raw' | 'dicom' | 'unknown')
    - detected_type: auto-detected interpretation ('raw' | 'dicom' | 'unknown')
    - image_array: centralized decoded 2D image array (required by strict ingestion mode)
    - pixel_spacing_row: row spacing in mm/pixel (optional)
    - pixel_spacing_col: column spacing in mm/pixel (optional)
    """

    name: str
    bytes: bytes
    file_type: Literal['raw', 'dicom', 'unknown']
    detected_type: Literal['raw', 'dicom', 'unknown']
    image_array: np.ndarray
    pixel_spacing_row: float
    pixel_spacing_col: float


def file_name_and_bytes(file_obj) -> tuple[str, bytes]:
    """Return (filename, bytes) for preloaded payload dict or Streamlit upload object."""
    if isinstance(file_obj, dict):
        return file_obj.get('name', 'unknown'), file_obj.get('bytes', b'')
    name = getattr(file_obj, 'name', 'unknown')
    data = file_obj.getvalue() if hasattr(file_obj, 'getvalue') else b''
    return name, data
