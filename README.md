# X-ray Image Analysis Toolkit

A Streamlit-based web application for performing common image quality analyses on medical DICOM X-ray images. This toolkit provides a user-friendly interface for uploading DICOM files and calculating key performance indicators for medical imaging systems.

## Overview

This application is designed for medical physicists, researchers, and engineers who need to evaluate the performance of X-ray imaging systems. It provides a suite of analysis tools that operate on DICOM files, including functionality to analyze single images or difference images created from two separate acquisitions.

The toolkit is built with modularity in mind, allowing for easy extension and addition of new analysis modules.

## Features

*   **DICOM File Handling**: Upload one or two `.dcm` files directly in the browser.
*   **Difference Imaging**: Automatically calculates a difference image if two files are uploaded, reverting to raw stored pixel values before subtraction.
*   **DICOM Header Inspection**: Displays key DICOM tags like `PixelSpacing`, `ImageType`, and `RescaleSlope`/`Intercept`.
*   **Raw Data Analysis**: Option to revert images to their original stored pixel values, undoing the Rescale Slope/Intercept transformation for analysis of the rawest data.
*   **Interactive Analysis Modules**:
    *   **Uniformity Analysis**: Calculates Global and Local Uniformity for both Pixel Value (PV) and Signal-to-Noise Ratio (SNR) using a sliding ROI method.
    *   **Noise Power Spectrum (NPS)**: Computes and displays the 1D radially averaged Normalized NPS (NNPS) with an interactive plot.
    *   **Modulation Transfer Function (MTF)**: Provides a simplified MTF calculation using the image's central row as a Line Spread Function (LSF).
    *   **Threshold Contrast**: Calculates the threshold contrast for a circular object, requiring data from both the MTF and NPS analyses.
*   **Data Persistence**: Uses Streamlit's session state to hold results from MTF and NPS analyses, making them available for dependent calculations like Threshold Contrast.

---

## Installation

To run this application locally, it is highly recommended to have installed uv, since it is far easier and straightforward.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/X-ray-imaging-analysis.git
    cd X-ray-imaging-analysis
    ```

2.  **Set up the environment and run:**

    ### Using `uv` (Recommended)
    The `uv run` command works across Windows, macOS, and Linux. It will automatically create a virtual environment, install dependencies, and launch the application.
    ```bash
    uv run main.py
    ```
    > **Note**: This requires a `pyproject.toml` or `requirements.txt` file in your project root.

    ### Using `pip` and `venv` (Manual)

    a. **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    b. **Run the app:**
    ```bash
    python main.py
    ```

---

## Usage

Once the setup is complete, you can either run the application using the main launcher script or create a .bat file for Windows Task Scheduler if you want to run it automatically at system startup.

### How to Use the App

1.  Launch the application.
2.  Use the sidebar to upload one or two DICOM files.
3.  The uploaded image (or the calculated difference image) will be displayed, along with key DICOM properties.
4.  If analyzing a single, non-difference image, you can choose to revert it to its raw stored pixel values.
5.  Select an analysis type from the sidebar dropdown menu.
6.  Click the "Run Analysis" button within the selected section to view the results.

---

## Analysis Modules

### Uniformity Analysis

This module assesses the uniformity of the image signal across a central region of interest (80% of the image area). It uses a sliding 30mm x 30mm ROI to calculate:
*   **Global Uniformity (GU_PV & GU_SNR)**: The maximum deviation of any local ROI's mean/SNR from the overall central ROI's mean/SNR.
*   **Local Uniformity (LU_PV & LU_SNR)**: The maximum deviation of any local ROI's mean/SNR from the average of its 8 immediate neighbors.

### Noise Power Spectrum (NPS) Analysis

Calculates the 2D Normalized Noise Power Spectrum (NNPS) and its 1D radial average. This is a fundamental measure of the noise texture and magnitude in an imaging system. The calculation is based on the IEC 62220-1 standard methodology.

*   The analysis is performed on a difference image to remove the low-frequency background signal.
*   The resulting 1D NNPS is plotted interactively using Altair.
*   The results are saved to the session state for use in other analyses.

### Modulation Transfer Function (MTF) Analysis

This module provides an estimate of the system's spatial resolution.

> **Note**: The current implementation is a simplified version that uses the central row of the image as a proxy for the Line Spread Function (LSF). For accurate results, an image of a sharp edge or a thin slit phantom is required.

*   Calculates the MTF from the FFT of the LSF.
*   Determines key metrics like **MTF50** and **MTF10**.
*   The results are saved to the session state.

### Threshold Contrast Analysis

This module calculates the minimum contrast an object must have to be detectable by an ideal observer, based on the Rose model. It incorporates the system's resolution (MTF), noise (NPS), and the human visual system's response (VTF).

*   **Dependencies**: This analysis is disabled until both MTF and NPS data have been calculated and saved in the current session.
*   **Inputs**: Requires user to input object radius and system magnification.
*   **Output**: Provides the threshold contrast value, a key indicator of low-contrast detectability.

---

## Contributing

Contributions are welcome! If you'd like to add a new analysis module, improve an existing one, or fix a bug, please feel free to fork the repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
