# Developer Tools — `src/tools/developer_tools/`

## Purpose

The `developer_tools` package provides **utility functions** for working with raw detector data files and DICOM containers. These tools are not part of the QA analysis pipeline — they serve as **diagnostic and conversion aids** for medical physicists and engineers who need to inspect, compare, and transform image data during development, troubleshooting, or data preparation workflows.

---

## Architecture Overview

```
src/tools/developer_tools/
├── comparison_tool.py   ← Pixel-level comparison between RAW and DICOM representations
└── dicomizer.py         ← Converts RAW pixel arrays into valid DICOM files
```

---

## Module Reference

### `comparison_tool.py` — RAW vs DICOM Pixel Comparison

Performs a **pixel-by-pixel comparison** between a RAW file and its DICOM counterpart to verify data integrity across format conversions. This is essential when validating that:

- A RAW-to-DICOM conversion preserved pixel values exactly.
- A DICOM file tagged as "FOR PROCESSING" contains the same pixel data as the original RAW file.
- No byte-order, bit-depth, or dimension errors were introduced during decoding.

#### Input

- **Exactly 2 files**: 1 RAW and 1 DICOM (detected or overridden types).

#### `display_comparison_tool_section(preloaded_files) -> None`

The Streamlit UI entry point. Validates that exactly one RAW and one DICOM file are present, extracts decoded image arrays from both preloaded payloads, and then performs a series of checks:

| Check | Pass | Fail |
|-------|------|------|
| **Shape match** | Both arrays have identical dimensions (rows × columns) | Dimension mismatch — cannot proceed with pixel comparison |
| **Data type match** | Same NumPy dtype | Warning: dtype mismatch may indicate bit-depth or signed/unsigned differences |
| **Pixel-by-pixel equality** | `np.array_equal(raw, dicom)` is `True` | Pixel mismatch detected |

When pixel values differ, the tool reports:
- **Maximum absolute difference**: The largest single-pixel deviation.
- **Mean pixel values**: For both arrays, helping identify systematic offsets.
- **Difference histogram**: Visual distribution of pixel-level differences, useful for diagnosing whether mismatches are random (noise/rounding) or systematic (byte-order, offset, scaling).

#### Use Case Example

A medical physicist receives flat-field images from a detector in proprietary RAW format and also as DICOM "FOR PROCESSING" images. Before running QA analyses, they want to verify that both files contain identical pixel data:

```
Upload: flat_field_001.raw + flat_field_001.dcm
       │
       ▼
 Shape check: 2048 × 2048 ✅
 Dtype check: uint16 ✅
 Pixel equality: ✅ All 4,194,304 pixels match
```

### `dicomizer.py` — RAW to DICOM Conversion

Converts a decoded RAW pixel array into a valid **DICOM Secondary Capture** file, creating all required DICOM metadata from scratch. This is useful for:

- Archiving RAW acquisitions in DICOM-compatible PACS systems.
- Sharing raw detector data with collaborators who only accept DICOM.
- Creating test DICOM files for software validation.

#### `generate_dicom_from_raw(image_array, pixel_spacing_row, pixel_spacing_col, original_filename) -> tuple[bytes, str]`

Creates a complete DICOM file in memory:

**File Meta Information:**

| Tag | Value |
|-----|-------|
| Media Storage SOP Class UID | Secondary Capture (`1.2.840.10008.5.1.4.1.1.7`) |
| Transfer Syntax UID | Explicit VR Little Endian |
| Implementation Class UID | Auto-generated |

**Dataset:**

| Tag | Source |
|-----|--------|
| Patient Name / ID | Placeholder values |
| Study Date / Time | Current timestamp |
| Rows, Columns | From `image_array.shape` |
| Pixel Spacing | From function arguments |
| Image Type | `['ORIGINAL', 'PRIMARY']` |
| Photometric Interpretation | `MONOCHROME2` |
| Samples Per Pixel | `1` |
| Bits Allocated / Stored / High Bit | Derived from `image_array.dtype` |
| Pixel Representation | `0` (unsigned) |
| Pixel Data | `image_array.tobytes()` |

**Returns**: A tuple of `(dicom_bytes, suggested_filename)` where the suggested filename replaces the original extension with `.dcm`. The bytes can be directly offered for download via Streamlit's download button.

---

## Design Decisions

- **Secondary Capture SOP Class**: Used because the converted file is not a direct acquisition — it's a reconstruction from RAW data. This is the semantically correct SOP class per the DICOM standard for non-acquisition images.
- **Placeholder patient metadata**: The dicomizer intentionally uses placeholder values for patient-identifying fields. Real clinical metadata should be added by the user's PACS or anonymization pipeline.
- **In-memory generation**: The DICOM file is created entirely in memory (using `BytesIO`) and never written to disk, avoiding file system side effects in the web application.
- **Explicit VR Little Endian**: The most widely supported DICOM transfer syntax, ensuring maximum compatibility with PACS and viewers.

---

## Integration with the Application

Both tools are routed through `workflow.py`:

- **Convert to DICOM**: The workflow loads the RAW image via the centralized decoder (`load_single_image()`), then passes the decoded array to `generate_dicom_from_raw()`. The resulting bytes are offered as a Streamlit download button.
- **RAW vs DICOM Comparison**: The workflow passes the list of preloaded payloads directly to `display_comparison_tool_section()`, which extracts the decoded arrays from each payload.

---

## Typical Workflows

### Converting a RAW File

```
Upload 1 RAW file
       │
       ▼
 Centralized decoder: resolve endian → decode bytes → reshape
       │
       ▼
 generate_dicom_from_raw():
   • Build DICOM File Meta Information
   • Build DICOM Dataset (dimensions, spacing, pixel data)
   • Write to in-memory buffer via pydicom.dcmwrite()
       │
       ▼
 Download button: "flat_field_001.dcm"
```

### Comparing RAW vs DICOM

```
Upload 1 RAW + 1 DICOM
       │
       ▼
 Both decoded via centralized loader (shared RAW params for the RAW file)
       │
       ▼
 display_comparison_tool_section():
   • Validate exactly 1 RAW + 1 DICOM
   • Compare shapes, dtypes
   • np.array_equal() for pixel-level comparison
   • If mismatch: compute max difference, render histogram
       │
       ▼
 Results: ✅ Match or ❌ Mismatch with diagnostics
```
