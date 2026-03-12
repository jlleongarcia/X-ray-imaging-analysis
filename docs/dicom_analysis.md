# DICOM Analysis — `src/qa/dicom_analysis/`

## Purpose

The `dicom_analysis` package implements **post-processing constancy testing** for clinical DICOM images. Unlike the flat panel QA modules (which operate on raw, unprocessed detector data), this module evaluates images that have already passed through the manufacturer's image processing pipeline — the images clinicians actually see and diagnose from.

The key question it answers: *Is the image processing chain producing consistent output quality over time?*

---

## Architecture Overview

```
src/qa/dicom_analysis/
└── postprocessing_constancy.py   ← SNR-based constancy analysis for processed DICOM images
```

---

## Physics Background

### Signal-to-Noise Ratio (SNR)

The fundamental metric computed by this module is:

$$\text{SNR} = \frac{\mu}{\sigma}$$

where $\mu$ is the mean pixel value and $\sigma$ is the standard deviation within a region of interest (ROI). For a flat-field exposure (uniform phantom or air shot), the SNR in the processed image reflects the combined effect of:

- **Detector noise** (quantum, electronic, structural).
- **Post-processing algorithms** (gain correction, noise reduction, edge enhancement, look-up table mapping).
- **Acquisition parameters** (kV, mAs, filtration).

By tracking SNR over time with consistent acquisition parameters, deviations indicate changes in either the detector hardware or the processing chain. This is the principle behind **constancy testing** in medical physics quality assurance.

### Central ROI Approach

The analysis uses a fixed **100 × 100 pixel** ROI extracted from the image center. This choice:

- Avoids the heel effect and geometric fall-off present at image edges.
- Provides a statistically meaningful sample (10,000 pixels).
- Is reproducible across measurements without requiring phantom landmarks.

---

## Module Reference

### `postprocessing_constancy.py`

#### `_extract_center_roi(image_array, roi_height=100, roi_width=100) -> np.ndarray | None`

Extracts a central rectangular ROI from the input image. Returns `None` if the image is smaller than the requested ROI dimensions.

#### `_compute_snr_components(roi_array) -> tuple[float, float, float]`

Computes the three magnitudes used in constancy reporting from the center ROI:

- **Mean Pixel Value** ($\mu$)
- **Standard Deviation** ($\sigma$)
- **SNR** ($\mu/\sigma$)

Handles the edge case of zero standard deviation by returning `np.nan` for SNR while still returning the measured mean and standard deviation.

#### `_get_dicom_tags(file_bytes) -> tuple[str, str]`

Extracts two DICOM metadata tags that provide context for each measurement:

- **`(0018,1405)` Relative X-Ray Exposure**: A manufacturer-reported exposure index that correlates with detector dose. Used to verify that acquisition parameters were consistent.
- **`(0018,0015)` Body Part Examined**: Identifies the anatomical region, useful when tracking constancy across different exam protocols.

#### `display_dicom_postprocessing_analysis_section(preloaded_files) -> None`

The Streamlit UI entry point. For each uploaded DICOM file:

1. Decodes the pixel array via the centralized image loader.
2. Extracts the 100 × 100 center ROI.
3. Computes Mean Pixel Value, Standard Deviation, and SNR.
4. Extracts metadata tags.
5. Assembles results into a pandas DataFrame with columns: **File**, **Body Part Examined**, **Relative X-ray Exposure**, **Mean Pixel Value**, **Standard Deviation**, **SNR**, **Status**.

The results table enables at-a-glance comparison across multiple images — for example, a series of monthly constancy images acquired with the same protocol.

---

## Typical Workflow

```
Upload N DICOM images (monthly constancy acquisitions)
       │
       ▼
 For each image:
   ┌─────────────────────────────────────┐
   │ 1. Decode pixel array (pydicom)     │
   │ 2. Extract 100×100 center ROI       │
      │ 3. Compute μ, σ, and SNR = μ / σ    │
      │ 4. Read Relative X-ray Exposure tag │
      │ 5. Read Body Part Examined tag      │
   └─────────────────────────────────────┘
       │
       ▼
 Results table (one row per image)
       │
       ▼
 Trend analysis: is SNR stable over time?
```

---

## Clinical Context

Post-processing constancy testing is part of routine **quality assurance programs** for digital radiography systems as recommended by:

- **AAPM TG-150** (American Association of Physicists in Medicine): Performance evaluation of digital radiography systems.
- **IEC 62494-1**: Exposure index of digital X-ray imaging systems.
- National and international regulatory frameworks for medical imaging equipment.

The module supports the medical physicist's workflow of periodically acquiring standardized images and verifying that the system's processing chain has not drifted. A sudden drop in SNR might indicate a failing detector element, a software update that altered processing parameters, or a calibration error.

---

## Design Decisions

- **Fixed ROI size (100 × 100)**: Standardized across all measurements for reproducibility. Large enough for statistical validity, small enough to avoid heel effect contamination.
- **Metadata extraction**: Including DICOM tags alongside Mean Pixel Value, Standard Deviation, and SNR values allows the physicist to identify and filter outliers caused by non-standard acquisition parameters rather than true system degradation.
- **Multi-file support**: Accepting multiple DICOM files simultaneously enables batch processing of constancy series, reducing manual effort in routine QA programs.
