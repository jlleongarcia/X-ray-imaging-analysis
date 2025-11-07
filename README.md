# X-ray Image Analysis Toolkit

A Streamlit-based web application for performing common image quality analyses on medical DICOM X-ray images. This toolkit provides a user-friendly interface for uploading DICOM files and calculating key performance indicators for medical imaging systems.

## Overview

This application is designed for medical physicists, researchers, and engineers who need to evaluate the performance of X-ray imaging systems. It provides a suite of analysis tools that operate on DICOM files, including functionality to analyze single or multiple images.

The toolkit is built with modularity in mind, allowing for easy extension and addition of new analysis modules.

## Features

*   **DICOM File Handling**: Upload one or two `.dcm` files directly in the browser.
*   **DICOM Header Inspection**: Displays a small set of helpful DICOM tags. The app now only highlights whether the image is `DERIVED` (ImageType contains 'DERIVED').
*   **Raw Data Handling**: The previous helper that attempted DICOM→RAW conversion has been removed. The app uses `pydicom`'s `pixel_array` consistently for display and analysis.
*   **From RAW to DICOM tool**: Convert RAW files into DICOM ones.
*   **Interactive Analysis Modules**:
    *   **Uniformity Analysis**: Calculates Global and Local Uniformity for both Pixel Value (PV) and Signal-to-Noise Ratio (SNR).
    *   **Noise Power Spectrum (NPS)**: Computes and plots the 1D radially averaged Normalized NPS (NNPS).
    *   **Modulation Transfer Function (MTF)**: Provides an MTF estimate using the image's central row as a proxy LSF (note: a slit/edge phantom yields better MTF results).
    *   **Threshold Contrast**: Estimates threshold contrast using MTF and NPS results.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/X-ray-imaging-analysis.git
    cd X-ray-imaging-analysis
    ```

2.  **Set up the environment and run:**

    ### Using `uv` (Recommended)
    The `uv run` command will create a venv, install dependencies, and launch the app:
    ```bash
    uv run main.py
    ```

    ### Using `pip` and `venv` (Manual)

    a. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    b. **Run the app:**
    ```bash
    python main.py
    ```

> Note: Installing `pyarrow` (used indirectly by Streamlit) from source requires CMake and other build tools. For a reproducible environment that avoids native builds, use the Docker instructions below.

### Running with Docker Compose (recommended)

This repo contains a multi-stage Dockerfile that creates a conda-based builder stage (binary `pyarrow`) and copies the prepared conda installation into a slim runtime image. This avoids building heavy native packages during container runtime.

To build and run:

```bash
docker compose up -d --build
```

Then open: http://127.0.0.1:8502

If you see the container repeatedly restarting, inspect logs:

```bash
docker logs X-ray-imaging-analysis --follow
```

Quick troubleshooting tips:

- Confirm the `streamlit` binary exists inside the container at `/opt/conda/envs/xr_env/bin/streamlit`.
- If you encounter long builds or CMake/pyarrow errors when not using Docker, install a binary `pyarrow` via conda or use the provided Dockerfile.
- To inspect the runtime container interactively (if it stays up long enough):

```bash
docker run --rm -it x-ray-imaging-analysis-streamlit-app /bin/bash
# then inspect /opt/conda/envs/xr_env/bin and site-packages
ls -la /opt/conda/envs/xr_env/bin
```

## Usage

1.  Launch the application (uv or docker).
2.  Upload one or two DICOM files from the sidebar.
3.  The main view shows the selected image and a small set of DICOM tags.
4.  Select the analysis module from the sidebar and run the analysis in that module's panel.


## Contributing

Contributions are welcome. Follow the usual fork -> branch -> PR workflow.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
