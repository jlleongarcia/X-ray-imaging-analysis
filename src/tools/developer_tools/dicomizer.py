import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import numpy as np
import datetime
import os
from io import BytesIO

def generate_dicom_from_raw(image_array: np.ndarray, pixel_spacing_row: float, pixel_spacing_col: float, original_filename: str) -> tuple[bytes, str]:
    """
    Converts a raw numpy image array into a DICOM file in memory.

    Args:
        image_array: The 2D numpy array containing pixel data.
        pixel_spacing_row: The pixel spacing in the row direction (mm/px).
        pixel_spacing_col: The pixel spacing in the column direction (mm/px).
        original_filename: The filename of the source RAW file.

    Returns:
        A tuple containing:
        - dicom_bytes (bytes): The generated DICOM file as a bytes object.
        - new_filename (str): The suggested filename for the new DICOM file.
    """
    # 1. Create File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # 2. Create the main dataset
    ds = Dataset()
    ds.file_meta = file_meta

    # 3. Add patient and study information (with placeholders)
    ds.PatientName = "RAW Import"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S')

    # 4. Set image-specific tags
    ds.Rows, ds.Columns = image_array.shape
    ds.PixelSpacing = [pixel_spacing_row, pixel_spacing_col]
    ds.ImageType = ['ORIGINAL', 'PRIMARY']
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = image_array.dtype.itemsize * 8
    ds.BitsStored = ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 0  # 0 for unsigned, 1 for signed

    # 5. Add the pixel data
    ds.PixelData = image_array.tobytes()

    # 6. Save to an in-memory bytes buffer
    buffer = BytesIO()
    pydicom.filewriter.dcmwrite(buffer, ds, write_like_original=False)
    buffer.seek(0)  # Go to the beginning of the buffer
    dicom_bytes = buffer.getvalue()
    new_filename = f"{os.path.splitext(original_filename)[0]}.dcm"

    return dicom_bytes, new_filename