import numpy as np
import pydicom

def _detect_footer_rows(image_array: np.ndarray) -> int:
    """
    Heuristically detects the number of footer rows at the bottom of an image.
    by looking for a "change point" in row statistics.

    Args:
        image_array (np.ndarray): The full 2D image array.

    Returns:
        int: The number of detected footer rows from the bottom.
    """
    if image_array.ndim != 2 or image_array.shape[0] < 100:
        return 0  # Not enough rows for a reliable guess

    row_stds = np.std(image_array, axis=1)

    # 1. Establish a baseline from the central 50% of the image
    center_start = image_array.shape[0] // 4
    center_end = 3 * (image_array.shape[0] // 4)
    baseline_std = np.median(row_stds[center_start:center_end])
    # Use a tolerance band around the baseline
    std_tolerance = baseline_std * 0.5  # Allow 50% deviation
    # Ensure tolerance is not zero for uniform images
    std_tolerance = max(std_tolerance, 1.0)

    lower_bound = baseline_std - std_tolerance
    upper_bound = baseline_std + std_tolerance

    # 2. Scan from the bottom up to find the change point
    # We need a few consecutive rows to match the baseline to be confident
    consecutive_match_needed = 10
    consecutive_matches = 0
    footer_rows = 0

    # Start scanning from the bottom, leaving a margin for the consecutive check
    for i in range(image_array.shape[0] - 1, consecutive_match_needed - 1, -1):
        is_image_row = lower_bound <= row_stds[i] <= upper_bound
        
        if is_image_row:
            consecutive_matches += 1
        else:
            consecutive_matches = 0  # Reset if the row doesn't look like image data

        if consecutive_matches >= consecutive_match_needed:
            # We found the start of the image data at index `i`.
            # The footer consists of all rows below this index.
            footer_rows = image_array.shape[0] - (i + consecutive_match_needed)
            break

    # 3. Apply a safety cap to prevent over-trimming (e.g., max 20% of image)
    return min(footer_rows, image_array.shape[0] // 5)

def get_raw_pixel_array(ds: pydicom.Dataset, auto_trim_footer: bool = False) -> np.ndarray:
    """
    Extracts the raw, untransformed pixel data from a pydicom dataset.
    This bypasses pydicom's automatic application of Rescale/Intercept, etc.

    Args:
        ds (pydicom.Dataset): The DICOM dataset.
        auto_trim_footer (bool): If True, automatically detects and trims footer rows.
    """
    # Decompress only if the dataset is compressed.
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()

    # Determine the numpy dtype from DICOM tags
    bits_allocated = ds.BitsAllocated
    pixel_representation = ds.PixelRepresentation  # 0 = unsigned, 1 = signed

    if bits_allocated == 8:
        dtype = np.uint8 if pixel_representation == 0 else np.int8
    elif bits_allocated == 16:
        dtype = np.uint16 if pixel_representation == 0 else np.int16
    elif bits_allocated == 32:
        dtype = np.uint32 if pixel_representation == 0 else np.int32
    else:
        raise ValueError(f"Unsupported Bits Allocated: {bits_allocated}")

    raw_bytes = ds.PixelData
    shape = (ds.Rows, ds.Columns)

    # Create array from buffer and reshape
    image_array = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

    # Automatically detect and trim footer rows if requested
    if auto_trim_footer:
        rows_to_trim = _detect_footer_rows(image_array)
        if rows_to_trim > 0:
            image_array = image_array[:-rows_to_trim, :]

    return image_array.copy() # Return a copy to avoid memory issues