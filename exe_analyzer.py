import subprocess
import sys
import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import os
import tempfile

# Import functions from your analysis modules
from uniformity import display_uniformity_analysis_section
from NPS import display_nps_analysis_section
from MTF import display_mtf_analysis_section

def install_packages():
    """Install packages from requirements.txt."""
    print("Checking and installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def main_app_ui():
    """Defines the main Streamlit UI."""
    st.set_page_config(page_title="X-ray Image Analysis Toolkit", layout="wide")
    st.title("X-ray Image Analysis Toolkit")

    # --- File Upload and Initial Image Display ---
    uploaded_file = st.sidebar.file_uploader("Choose a DICOM file", type=["dcm", "dicom"])

    image_array = None
    pixel_spacing_row = None
    pixel_spacing_col = None
    dicom_filename = None

    if uploaded_file is not None:
        dicom_filename = uploaded_file.name
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            dicom_dataset = pydicom.dcmread(tmp_file_path)
            
            if 'PixelData' in dicom_dataset:
                image_array = dicom_dataset.pixel_array
            else:
                st.error("DICOM file does not contain pixel data.")
            
            if 'PixelSpacing' in dicom_dataset:
                pixel_spacing = dicom_dataset.PixelSpacing
                if len(pixel_spacing) == 2:
                    pixel_spacing_row = float(pixel_spacing[0])
                    pixel_spacing_col = float(pixel_spacing[1])
                else:
                    st.warning(f"Pixel Spacing tag (0028,0030) has unexpected format: {pixel_spacing}.")
            else:
                st.warning("Pixel Spacing tag (0028,0030) not found in DICOM header.")

        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            image_array = None # Ensure it's reset on error
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    # --- Main Area ---
    if image_array is not None:
        st.header(f"Uploaded Image: {dicom_filename}")
        if pixel_spacing_row and pixel_spacing_col:
            st.write(f"Pixel Spacing: {pixel_spacing_row:.3f} mm/px (row) x {pixel_spacing_col:.3f} mm/px (col)")
        else:
            st.write("Pixel Spacing: Not available or not applicable for selected analysis.")

        # Display original image
        display_array = image_array.astype(np.float32)
        if np.max(display_array) > np.min(display_array):
            display_array = (display_array - np.min(display_array)) / (np.max(display_array) - np.min(display_array)) * 255.0
        else:
            display_array = np.zeros_like(display_array)
        display_array = display_array.astype(np.uint8)

        if len(display_array.shape) == 2:
            img_pil = Image.fromarray(display_array, mode='L')
            st.image(img_pil, caption="Original Image (Normalized for Display)", use_container_width=True)
        else:
            st.warning(f"Image has unexpected shape {image_array.shape}. Cannot display directly.")
        
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ("Select an analysis...", "Uniformity Analysis", "NPS Analysis", "MTF Analysis")
        )

        if analysis_type == "Uniformity Analysis":
            display_uniformity_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "NPS Analysis":
            display_nps_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        elif analysis_type == "MTF Analysis":
            display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col)
        
    elif uploaded_file is None:
        st.info("Please upload a DICOM file using the sidebar to begin analysis.")


if __name__ == "__main__":
    print("Starting setup script...")

    # Install dependencies
    install_packages()
    
    # Run the main Streamlit UI
    print("Starting the Streamlit app UI from exe_analyzer.py...")
    main_app_ui()
