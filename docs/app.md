# Application Layer — `src/app/menu_analyzer/`

## Purpose

The `app` package is the **presentation and orchestration layer** of the X-ray Imaging Analysis Toolkit. It implements the entire user-facing web interface using [Streamlit](https://streamlit.io/) and coordinates the interaction between the user and the underlying analysis engines. Every action a user performs—uploading files, selecting analyses, configuring parameters, and viewing results—is mediated by modules in this package.

---

## Architecture Overview

```
src/app/menu_analyzer/
├── app.py              ← Main Streamlit entry point (UI layout, session state)
├── catalog.py          ← Registry of all available analyses and their metadata
├── file_detection.py   ← Automatic DICOM vs RAW file classification
├── image_loader.py     ← Centralized image decoding (DICOM & RAW)
├── raw_params.py       ← RAW file parameter inference & UI
└── workflow.py         ← Analysis routing and orchestration engine
```

### Data Flow

```
User uploads files
       │
       ▼
 file_detection.py ──► Auto-classify each file (DICOM / RAW / unknown)
       │
       ▼
 image_loader.py ──► Build preloaded payloads (read bytes once, lazy-decode)
       │
       ▼
 raw_params.py ──► If RAW files present: render parameter UI in sidebar
       │                (dtype, dimensions, pixel spacing, endian, extra bytes)
       ▼
 workflow.py ──► Route to the appropriate QA or developer tool module
       │
       ▼
 Analysis module returns results → displayed in Streamlit UI
```

---

## Module Reference

### `app.py` — Main UI & Session State

This is the Streamlit entry point (`streamlit run src/app/menu_analyzer/app.py`). It is responsible for:

- **Session state initialization**: Manages persistent state across Streamlit reruns for detector conversion fits, MTF/NPS caches, RAW parameter defaults, and UI refresh counters.
- **Sidebar rendering**: File type overrides, RAW endian toggle, saved analysis data status indicators (detector fit, MTF, NPS caches), and a "Clear All Saved Analysis Data" button.
- **Category and test selection UI**: Renders a grid of analysis categories (Flat Panel QA, Developer Tools, DICOM Analysis) and, upon selection, the individual tests within each category.
- **Info cards**: Displays the selected test's description, requirements, and computation formula (LaTeX-rendered) below the selector.
- **File upload trigger**: Once a test is selected, shows the appropriate Streamlit file uploader and hands off to `workflow.py`.

**Key session state keys:**

| Key | Type | Purpose |
|-----|------|---------|
| `detector_conversion` | `dict` | Cached detector response fit, EI fit, σ² fit, results |
| `mtf_cache` | `dict` | Cached MTF results and geometric mean |
| `nps_cache` | `dict` | Cached NPS results and kerma value |
| `selected_category` | `str` | Currently selected analysis category |
| `selected_test` | `str` | Currently selected test name |
| `raw_little_endian_default` | `bool` | Global little-endian default for RAW decoding |
| `shared_raw_params_current_test` | `dict` | Shared RAW decoder params for the active test |

### `catalog.py` — Analysis Registry

Returns a nested dictionary describing all available analyses. Each test entry contains:

- **`icon`**: Unicode emoji for UI display.
- **`description`**: One-line summary.
- **`files_needed`** and **`file_types`**: Number and type of files required.
- **`requirements`**: List of prerequisite conditions.
- **`computation`**: LaTeX-formatted formula and methodology description.

**Analysis categories:**

| Category | Tests |
|----------|-------|
| **Flat Panel QA** | Detector Response Curve, Uniformity, MTF, NPS, DQE, TCDD |
| **Developer Tools** | Convert to DICOM, RAW vs DICOM Comparison |
| **DICOM Analysis** | Post-processing Analysis (SNR) |

### `file_detection.py` — File Type Classification

```python
def detect_file_type(file_bytes: bytes, filename: str) -> Literal['dicom', 'raw', 'unknown']
```

Automatic classification logic:

1. Attempt `pydicom.dcmread(force=True, stop_before_pixels=True)`.
2. Check DICOM tag `(0008,0068) PresentationIntentType`:
   - `"FOR PRESENTATION"` → `'dicom'` (post-processed).
   - `"FOR PROCESSING"` → `'raw'` (unprocessed detector data).
3. Fallback to `(0008,0008) ImageType`:
   - Contains `ORIGINAL` / `PRIMARY` → `'raw'`.
   - Contains `DERIVED` / `SECONDARY` → `'dicom'`.
4. If Rows/Columns DICOM tags exist → `'dicom'`.
5. Default → `'raw'`.

This classification is critical because the toolkit treats **"FOR PROCESSING"** DICOM files (raw detector data wrapped in DICOM headers) as RAW data for flat panel QA, while **"FOR PRESENTATION"** files (post-processed) are used in DICOM analysis workflows.

### `image_loader.py` — Centralized Image Decoding

The **single point of truth** for decoding uploaded files into NumPy arrays. All analysis modules receive their pixel data through this module.

**Key functions:**

- **`build_preloaded_payloads()`**: Creates `ImagePayload` dicts for each uploaded file (reads bytes once, stores file type and detected type).
- **`ensure_payload_loaded()`**: Lazy-loads `image_array` into an existing payload using the shared RAW parameters from session state. Caches the decoded result.
- **`load_single_image()`**: Core decoder that handles both DICOM and RAW paths:
  - **DICOM path**: `pydicom.dcmread()` → `pixel_array`, extracts `ImagerPixelSpacing`.
  - **RAW path**: Uses shared RAW params (dtype, dimensions, extra bytes) and `frombuffer_with_endian()` from `core.io.raw_endian`.

This centralized approach ensures consistent decoding across all analyses and prevents UI-blocking failures from propagating.

### `raw_params.py` — RAW File Parameter UI

When RAW files are uploaded, this module renders a sidebar UI for configuring the decode parameters shared across all files in the current test:

1. **Pixel data type** (`uint8`, `uint16`, `float32`) — auto-preset from DICOM header hints if embedded metadata is available.
2. **Pixel spacing** (mm/pixel for row and column) — auto-filled from DICOM `ImagerPixelSpacing` tag when present.
3. **Image dimensions** — inferred from file size (lists valid factor pairs with reasonable aspect ratios) or extracted from DICOM Rows/Columns tags. Manual override available.
4. **Extra bytes handling** — detects header/trailer bytes not part of pixel data; user specifies location (start/end) and skip amount.

### `workflow.py` — Analysis Routing Engine

The central orchestrator that connects the UI to the analysis back-end. After files are uploaded and classified:

1. Renders per-file type override dropdowns in the sidebar.
2. Builds preloaded payloads and shared RAW parameters.
3. Validates the file mix (e.g., no mixing RAW + DICOM for most tests).
4. Routes to the appropriate analysis function based on the selected test:

| Test | Routed To |
|------|-----------|
| Detector Response Curve | `detector_conversion.display_detector_conversion_section()` |
| Uniformity | `uniformity.display_uniformity_analysis_section()` |
| MTF | `mtf.display_mtf_analysis_section()` |
| NPS | `nps.display_nps_analysis_section()` |
| DQE | `dqe.display_dqe_analysis_section()` |
| TCDD | `threshold_contrast.display_threshold_contrast_section()` |
| DICOM Post-processing | `postprocessing_constancy.display_dicom_postprocessing_analysis_section()` |
| Convert to DICOM | `dicomizer.generate_dicom_from_raw()` |
| RAW vs DICOM Comparison | `comparison_tool.display_comparison_tool_section()` |

---

## Design Decisions

- **Lazy decoding**: Image bytes are read once at upload time, but pixel array decoding is deferred until needed. This keeps the UI responsive for multi-file uploads.
- **Shared RAW parameters**: All RAW files within a single test share the same decoder configuration. This simplifies the UI and is physically valid (files from the same detector acquisition share dimensions and encoding).
- **Session-state caching**: Expensive results (detector fits, MTF, NPS) persist across Streamlit reruns, enabling the DQE computation to consume MTF and NPS caches without re-running those analyses.
- **Non-blocking previews**: File preview failures (e.g., decode errors) do not prevent the analysis UI from rendering; users can still configure parameters and retry.

---

## Project Evolution

The application layer evolved significantly during development:

- **v0 — Initial prototype**: Simple single-file DICOM uploader with basic SNR computation.
- **RAW file support**: Added auto-detection of FOR PROCESSING vs FOR PRESENTATION DICOM images, enabling flat panel QA on raw detector data.
- **Centralized decoding**: Refactored from per-module decoding to a single `image_loader.py`, eliminating inconsistencies and improving reliability.
- **Session caching**: Introduced session state management for cross-test data sharing (detector fit → uniformity kerma domain, MTF + NPS → DQE).
- **Containerization**: Docker multi-stage build with conda for reproducible deployment.
