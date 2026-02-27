import io
from typing import Optional, Tuple

import numpy as np

try:
    import pydicom
except Exception:
    pydicom = None


BIG_ENDIAN_TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.2"


def infer_endian_from_dicom_header(raw_bytes: bytes) -> Tuple[Optional[bool], str]:
    """Infer little/big-endian from DICOM Transfer Syntax UID (0002,0010).

    Returns:
        (little_endian | None, source_message)
    """
    if pydicom is None:
        return None, "pydicom unavailable"

    try:
        ds_meta = pydicom.dcmread(io.BytesIO(raw_bytes), force=True, stop_before_pixels=True)
        transfer_syntax_uid = None

        if getattr(ds_meta, "file_meta", None) is not None:
            transfer_syntax_uid = getattr(ds_meta.file_meta, "TransferSyntaxUID", None)

        if transfer_syntax_uid is None:
            elem = ds_meta.get((0x0002, 0x0010))
            transfer_syntax_uid = elem.value if elem is not None else None

        if not transfer_syntax_uid:
            return None, "no Transfer Syntax UID found"

        transfer_syntax_uid = str(transfer_syntax_uid).strip()
        little_endian = transfer_syntax_uid != BIG_ENDIAN_TRANSFER_SYNTAX_UID
        return little_endian, f"DICOM Transfer Syntax UID {transfer_syntax_uid}"
    except Exception:
        return None, "no readable DICOM header"


def dtype_with_endian(dtype, little_endian: bool = True) -> np.dtype:
    """Return dtype with explicit byteorder for multibyte element sizes."""
    dtype_obj = np.dtype(dtype)
    if dtype_obj.itemsize <= 1:
        return dtype_obj
    return dtype_obj.newbyteorder('<' if little_endian else '>')


def resolve_effective_endian(
    raw_bytes: bytes,
    default_little_endian: bool = True,
    auto_endian_from_dicom: bool = True,
) -> Tuple[bool, str]:
    """Resolve effective endian using default setting and optional DICOM override."""
    endian_used = bool(default_little_endian)
    source = "global default"

    if auto_endian_from_dicom:
        detected_endian, detected_source = infer_endian_from_dicom_header(raw_bytes)
        if detected_endian is not None:
            endian_used = bool(detected_endian)
            source = detected_source

    return endian_used, source


def frombuffer_with_endian(
    raw_bytes: bytes,
    dtype,
    default_little_endian: bool = True,
    auto_endian_from_dicom: bool = True,
):
    """Read bytes with effective endian and return (array, endian_used, source)."""
    endian_used, source = resolve_effective_endian(
        raw_bytes,
        default_little_endian=default_little_endian,
        auto_endian_from_dicom=auto_endian_from_dicom,
    )
    effective_dtype = dtype_with_endian(dtype, endian_used)
    return np.frombuffer(raw_bytes, dtype=effective_dtype), endian_used, source
