import io

import numpy as np
import pandas as pd
import pydicom
import streamlit as st


SNR_COLUMN = "Signal-to-Noise Ratio"
PIXEL_SPACING_COLUMN = "Pixel Spacing"


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


def _compute_snr_components(roi_array: np.ndarray) -> tuple[float, float, float]:
    if roi_array is None or roi_array.size == 0:
        return np.nan, np.nan, np.nan

    mean_signal = float(np.mean(roi_array))
    noise_sd = float(np.std(roi_array))

    if np.isclose(noise_sd, 0.0):
        return mean_signal, noise_sd, np.nan

    return mean_signal, noise_sd, (mean_signal / noise_sd)


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


def _format_pixel_spacing(pixel_spacing_row, pixel_spacing_col) -> str:
    if pixel_spacing_row is None or pixel_spacing_col is None:
        return ""

    try:
        return f"{float(pixel_spacing_row):.4f}, {float(pixel_spacing_col):.4f}"
    except (TypeError, ValueError):
        return ""


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
                pixel_spacing_row = payload.get('pixel_spacing_row')
                pixel_spacing_col = payload.get('pixel_spacing_col')
                pixel_spacing = _format_pixel_spacing(pixel_spacing_row, pixel_spacing_col)

                relative_xray_exposure, body_part_examined = _get_dicom_tags(file_bytes)

                if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
                    rows.append({
                        "File": file_name,
                        "Body Part Examined": body_part_examined,
                        "Relative X-ray Exposure": relative_xray_exposure,
                        PIXEL_SPACING_COLUMN: pixel_spacing,
                        "Mean Pixel Value": np.nan,
                        "Standard Deviation": np.nan,
                        SNR_COLUMN: np.nan,
                        "Status": "Failed to decode image",
                    })
                    continue

                roi_array = _extract_center_roi(image_array, roi_height=100, roi_width=100)
                if roi_array is None:
                    rows.append({
                        "File": file_name,
                        "Body Part Examined": body_part_examined,
                        "Relative X-ray Exposure": relative_xray_exposure,
                        PIXEL_SPACING_COLUMN: pixel_spacing,
                        "Mean Pixel Value": np.nan,
                        "Standard Deviation": np.nan,
                        SNR_COLUMN: np.nan,
                        "Status": "Image smaller than 100x100",
                    })
                    continue

                mean_signal, noise_sd, snr_value = _compute_snr_components(roi_array)
                status_msg = "OK" if np.isfinite(snr_value) else "Invalid ROI noise (std=0 or non-finite)"

                rows.append({
                    "File": file_name,
                    "Body Part Examined": body_part_examined,
                    "Relative X-ray Exposure": relative_xray_exposure,
                    PIXEL_SPACING_COLUMN: pixel_spacing,
                    "Mean Pixel Value": mean_signal,
                    "Standard Deviation": noise_sd,
                    SNR_COLUMN: snr_value,
                    "Status": status_msg,
                })

        if not rows:
            st.warning("No valid files to analyze.")
            return

        results_df = pd.DataFrame(rows)
        if SNR_COLUMN in results_df.columns:
            results_df[SNR_COLUMN] = pd.to_numeric(
                results_df[SNR_COLUMN],
                errors='coerce'
            ).round(4)
        if "Mean Pixel Value" in results_df.columns:
            results_df["Mean Pixel Value"] = pd.to_numeric(
                results_df["Mean Pixel Value"],
                errors='coerce'
            ).round(4)
        if "Standard Deviation" in results_df.columns:
            results_df["Standard Deviation"] = pd.to_numeric(
                results_df["Standard Deviation"],
                errors='coerce'
            ).round(4)

        st.success("DICOM analysis complete.")
        # Streamlit right-aligns numeric columns by default; cast SNR to text for left alignment.
        display_df = results_df.copy()
        if "Mean Pixel Value" in display_df.columns:
            display_df["Mean Pixel Value"] = display_df["Mean Pixel Value"].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
        if "Standard Deviation" in display_df.columns:
            display_df["Standard Deviation"] = display_df["Standard Deviation"].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
        if SNR_COLUMN in display_df.columns:
            display_df[SNR_COLUMN] = display_df[SNR_COLUMN].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )

        st.dataframe(display_df, use_container_width=True)
