# Core I/O Layer — `src/core/io/`

## Purpose

The `core` package provides the **foundational data types and low-level I/O primitives** that every other module in the project depends on. It defines the canonical data schema for passing image data between modules and implements the byte-level decoding logic for RAW detector files, including endianness detection and resolution.

This layer has **zero UI dependencies** — it knows nothing about Streamlit, session state, or user interactions. It is a pure data manipulation library.

---

## Architecture Overview

```
src/core/io/
├── analysis_payload.py   ← ImagePayload TypedDict + file utility
└── raw_endian.py         ← Endian detection, byte-order resolution, buffer decoding
```

---

## Module Reference

### `analysis_payload.py` — Canonical Image Data Schema

Defines `ImagePayload`, a `TypedDict` that serves as the **standard data contract** between the application layer and every analysis module. All uploaded files are wrapped in this schema before being passed downstream.

```python
class ImagePayload(TypedDict, total=False):
    name: str                                          # Original filename
    bytes: bytes                                       # Raw file bytes (read once at upload)
    file_type: Literal['raw', 'dicom', 'unknown']      # User-selected / overridden type
    detected_type: Literal['raw', 'dicom', 'unknown']  # Auto-detected type
    image_array: np.ndarray                            # Decoded 2D pixel array (lazy-loaded)
    pixel_spacing_row: float                           # Row spacing in mm/pixel
    pixel_spacing_col: float                           # Column spacing in mm/pixel
```

**Design rationale:**

- **`total=False`**: Not all fields are required at creation time. The `image_array` and pixel spacing fields are populated lazily, only when the analysis actually needs decoded pixel data.
- **Bytes read once**: The `bytes` field stores the complete file content at upload time, avoiding repeated I/O operations. Subsequent decoding operates on this in-memory buffer.
- **Dual type fields**: `detected_type` captures the auto-classification result, while `file_type` stores the user's override (if any). This allows the UI to show the auto-detection while respecting manual corrections.

**Utility function:**

```python
def file_name_and_bytes(file_obj) -> tuple[str, bytes]
```

Normalizes access to file name and bytes, handling both Streamlit `UploadedFile` objects (with `.name` and `.read()` attributes) and `ImagePayload` dicts (with `'name'` and `'bytes'` keys). This abstraction lets downstream code work uniformly regardless of the data source.

### `raw_endian.py` — Endian Detection & Byte-Order Decoding

Handles the endianness problem inherent to raw medical imaging files. Detector manufacturers store pixel data in either little-endian or big-endian byte order, and files wrapped in DICOM containers encode this information in the Transfer Syntax UID. Pure RAW files (no DICOM header) require a configurable default.

**Key functions:**

#### `infer_endian_from_dicom_header(raw_bytes: bytes) -> tuple[bool | None, str]`

Attempts to detect endianness from an embedded DICOM Transfer Syntax UID `(0002,0010)`:

- Parses the raw bytes looking for the DICOM meta-information header.
- If `TransferSyntaxUID` is `1.2.840.10008.1.2.2` (Explicit VR Big Endian), returns `False` (big endian).
- Otherwise returns `True` (little endian) for all other DICOM transfer syntaxes.
- Returns `None` if no DICOM header is found in the byte stream.

#### `dtype_with_endian(dtype, little_endian: bool = True) -> np.dtype`

Applies the appropriate byte-order marker to a NumPy dtype:

- For multi-byte types (e.g., `uint16`, `float32`): prefixes with `<` (little-endian) or `>` (big-endian).
- For single-byte types (`uint8`): returns the dtype unchanged (endianness is irrelevant).

#### `resolve_effective_endian(raw_bytes, default_little_endian, auto_endian_from_dicom) -> tuple[bool, str]`

The **resolution strategy** for determining effective byte order:

1. **If auto-detection is enabled**: attempt to read the DICOM Transfer Syntax UID from the file bytes.
2. **If a DICOM header is found**: use its endianness (with a descriptive source string like `"DICOM Transfer Syntax UID 1.2.840.10008.1.2.1"`).
3. **Fallback**: use the `default_little_endian` setting from the application's global sidebar toggle.

#### `frombuffer_with_endian(raw_bytes, dtype, default_little_endian, auto_endian_from_dicom) -> tuple[np.ndarray, bool, str]`

The **workhorse decoder**: converts a byte buffer into a typed NumPy array with correct endianness.

1. Calls `resolve_effective_endian()` to determine byte order.
2. Calls `dtype_with_endian()` to build the correctly ordered dtype.
3. Calls `np.frombuffer()` to decode the buffer.
4. Returns the decoded array, the endianness used, and the source of that determination.

This function is the sole byte-level decoder in the project; all RAW file decoding flows through it.

---

## Design Decisions

- **Separation of concerns**: The core layer contains no UI logic, no file I/O (it operates on in-memory buffers), and no domain-specific analysis. This makes it testable and reusable.
- **Auto-detection with override**: The endian resolution chain prioritizes embedded metadata (DICOM Transfer Syntax) but always allows the user to override via the global sidebar setting. This handles both pure RAW files and pseudo-RAW files (raw pixel data wrapped in DICOM containers with metadata headers).
- **TypedDict over dataclass**: `ImagePayload` uses `TypedDict` rather than a dataclass because payloads are frequently passed through Streamlit's session state and caching mechanisms, which serialize dicts more reliably than custom objects.
- **Lazy loading pattern**: Only `name`, `bytes`, `file_type`, and `detected_type` are populated at upload time. The expensive `image_array` decoding is deferred until needed, keeping the UI responsive for multi-file uploads.

---

## Integration Points

| Consumer | What it uses | Why |
|----------|-------------|-----|
| `image_loader.py` | `frombuffer_with_endian()` | Decodes RAW pixel buffers with correct endianness |
| `image_loader.py` | `ImagePayload` | Creates and populates payload dicts for all uploaded files |
| `file_detection.py` | `file_name_and_bytes()` | Extracts bytes from various file-like objects |
| `detector_conversion.py` | `frombuffer_with_endian()` | Direct RAW decoding for legacy square-reshape path |
| `workflow.py` | `ImagePayload` | Types the preloaded data flowing between app and QA modules |
| All analysis modules | `ImagePayload` | Receive decoded image data in a consistent schema |
