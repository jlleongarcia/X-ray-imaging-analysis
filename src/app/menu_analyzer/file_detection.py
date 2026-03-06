import io

import pydicom


def detect_file_type(file_bytes, filename):
    """
    Detect file type by examining DICOM tag (0008,0068) Presentation Intent Type.
    Returns: 'dicom', 'raw', or 'unknown'

    Logic:
    - If DICOM tag (0008,0068) exists:
      - "FOR PRESENTATION" → true DICOM file
      - "FOR PROCESSING" → RAW file (even if has .dcm extension)
        - If no PresentationIntentType, inspect ImageType (0008,0008):
            - ORIGINAL/PRIMARY → RAW/STD file
            - DERIVED/SECONDARY → DICOM file
    """
    try:
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True, stop_before_pixels=True)

            if hasattr(ds, 'PresentationIntentType') and ds.PresentationIntentType:
                presentation_intent = str(ds.PresentationIntentType).upper().strip()

                if "FOR PRESENTATION" in presentation_intent:
                    return 'dicom'
                elif "FOR PROCESSING" in presentation_intent:
                    return 'raw'
                else:
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        return 'dicom'
            else:
                image_type = getattr(ds, 'ImageType', None)
                if image_type:
                    if isinstance(image_type, str):
                        image_type_values = [x.strip().upper() for x in image_type.replace('\\', '/').split('/') if x.strip()]
                    else:
                        image_type_values = [str(x).strip().upper() for x in image_type if str(x).strip()]

                    if 'ORIGINAL' in image_type_values or 'PRIMARY' in image_type_values:
                        return 'raw'
                    if 'DERIVED' in image_type_values or 'SECONDARY' in image_type_values:
                        return 'dicom'

                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    return 'dicom'

        except Exception:
            return 'raw'

        return 'raw'

    except Exception:
        return 'unknown'
