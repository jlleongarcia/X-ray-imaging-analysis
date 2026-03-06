import io

import numpy as np
import pandas as pd
import pydicom
import streamlit as st


def _extract_center_roi(image_array: np.ndarray, roi_height: int = 100, roi_width: int = 100):
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        return None

    height, width = image_array.shape
    if height < roi_height or width < roi_width:
        return None

    row_start = (height - roi_height) // 2
    col_start = (width - roi_width) // 2
    row_end = row_start + roi_height
    col_end = col_start + roi_width

    return image_array[row_start:row_end, col_start:col_end]


def _compute_snr_from_roi(roi_array: np.ndarray) -> float:
    if roi_array is None or roi_array.size == 0:
        return np.nan

    mean_signal = float(np.mean(roi_array))
    noise_sd = float(np.std(roi_array))

    if np.isclose(noise_sd, 0.0):
        return np.nan

    return mean_signal / noise_sd


def _get_dicom_tags(file_bytes: bytes) -> tuple[str, str]:
    relative_xray_exposure = ""
    body_part_examined = ""

    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        relative_xray_exposure = str(getattr(ds, 'RelativeXRayExposure', ""))
        body_part_examined = str(getattr(ds, 'BodyPartExamined', ""))
    except Exception:
        pass

    return relative_xray_exposure, body_part_examined


def display_dicom_postprocessing_analysis_section(preloaded_files: list[dict]):
    st.subheader("DICOM Post-processing Analysis")
    st.caption("Computes central ROI SNR ($100 \\times 100$ pixels) for each uploaded DICOM image.")

    if not preloaded_files:
        st.warning("No files available for analysis.")
        return

    if st.button("Run DICOM Analysis", type="primary", use_container_width=True):
        rows = []

        with st.spinner("Computing SNR for uploaded DICOM files..."):
            for payload in preloaded_files:
                file_name = payload.get('name', 'unknown')
                image_array = payload.get('image_array')
                file_bytes = payload.get('bytes', b'')

                relative_xray_exposure, body_part_examined = _get_dicom_tags(file_bytes)

                if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
                    rows.append({
                        "File": file_name,
                        "Body Part Examined": body_part_examined,
                        "Relative X-ray Exposure": relative_xray_exposure,
                        "SNR (100x100 center ROI)": np.nan,
                        "Status": "Failed to decode image",
                    })
                    continue

                roi_array = _extract_center_roi(image_array, roi_height=100, roi_width=100)
                if roi_array is None:
                    rows.append({
                        "File": file_name,
                        "Body Part Examined": body_part_examined,
                        "Relative X-ray Exposure": relative_xray_exposure,
                        "SNR (100x100 center ROI)": np.nan,
                        "Status": "Image smaller than 100x100",
                    })
                    continue

                snr_value = _compute_snr_from_roi(roi_array)
                status_msg = "OK" if np.isfinite(snr_value) else "Invalid ROI noise (std=0 or non-finite)"

                rows.append({
                    "File": file_name,
                    "Body Part Examined": body_part_examined,
                    "Relative X-ray Exposure": relative_xray_exposure,
                    "SNR (100x100 center ROI)": snr_value,
                    "Status": status_msg,
                })

        if not rows:
            st.warning("No valid files to analyze.")
            return

        results_df = pd.DataFrame(rows)
        if "SNR (100x100 center ROI)" in results_df.columns:
            results_df["SNR (100x100 center ROI)"] = pd.to_numeric(
                results_df["SNR (100x100 center ROI)"],
                errors='coerce'
            ).round(4)

        st.success("DICOM analysis complete.")
        styled_results_df = results_df.style.set_properties(
            subset=["SNR (100x100 center ROI)"],
            **{"text-align": "left"}
        )
        st.dataframe(styled_results_df, use_container_width=True)
