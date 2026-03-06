def get_analysis_catalog():
    return {
        "Flat Panel QA": {
            "icon": "📊",
            "description": "Quality assessment tests for flat panel detectors",
            "tests": {
                "Detector Response Curve": {
                    "description": "Calibrate detector pixel values to radiation dose",
                    "files_needed": "3+ RAW images at different exposures",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Multiple RAW files", "Different exposure levels", "Same detector settings"],
                    "computation": "Assign kerma value to each uploaded RAW/STD file, compute central 100x100 MPV and σ.\n\n Uses least-squares fitting to establish: $MPV = f(K_{air})$ and $EI = f(K_{air})$. Enables conversion from detector units to air kerma for subsequent analyses.\n\n Provides detailed noise components analysis by Weighted Robust Linear Models to separate quantum noise, electronic noise, and structural noise. Forces parameters to be non-negative.",
                    "icon": "📈"
                },
                "Uniformity": {
                    "description": "Measure spatial uniformity across detector area",
                    "files_needed": "1 flat-field image",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "No phantom objects", "Covers full detector area"],
                    "computation": "Calculates uniformity metrics within a central 80% area ROI using a sliding 30mm × 30mm window.\n\n**MPV Global Uniformity:**\n$$U_{global} = max\\left(\\frac{|MPV_{ij}-MVP|}{MPV}\\right)$$\n\n**MPV Local Uniformity:**\n$$U_{local} = max\\left(\\frac{|MPV_{ij}-MVP_{8n}|}{MPV_{8n}}\\right)$$\n\n**SNR Global Uniformity:**\n$$SNR_{global} = max\\left(\\frac{|SNR_{ij}-SNR|}{SNR}\\right)$$\n\n**SNR Local Uniformity:**\n$$SNR_{local} = max\\left(\\frac{|SNR_{ij}-SNR_{8n}|}{SNR_{8n}}\\right)$$",
                    "icon": "🔲"
                },
                "Modulation Transfer Function (MTF)": {
                    "description": "Measure spatial resolution and sharpness",
                    "files_needed": "1-2 images with edge/slit phantom",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Edge or slit phantom", "Sharp edge visible", "2 orthogonal edges for DQE analysis"],
                    "computation": "IEC 62220-1-1:2015 slanted edge method. Uses Hough transform for edge detection, computes Edge Spread Function (ESF), differentiates to get Line Spread Function: $LSF = \\frac{dESF}{dx}$, then Fourier transform: $MTF(f) = |\\mathcal{F}\\{LSF\\}|$. Reports $MTF_{10\\%}$ and $MTF_{50\\%}$ as key metrics.",
                    "icon": "📐"
                },
                "Noise Power Spectrum (NPS)": {
                    "description": "Characterize noise power spectrum",
                    "files_needed": "1+ uniform images",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "Multiple images recommended", "Known air kerma value"],
                    "computation": "IEC 62220-1-1:2015 standard. Extracts $N \\times N$ ROIs, applies 2D FFT to each ROI, averages power spectra. Normalizes: $$NPS(u,v) = \\frac{\\Delta x \\Delta y}{N_x N_y} |FFT|^2$$ Computes radial profile and integrates for total noise.",
                    "icon": "📡"
                },
                "DQE (Detective Quantum Efficiency)": {
                    "description": "Calculate overall detector quality metric",
                    "files_needed": "Requires MTF + NPS results",
                    "file_types": "Uses cached results",
                    "requirements": ["MTF from orthogonal edges", "NPS computed", "Known air kerma value"],
                    "computation": "IEC 62220-1-1:2015 formula: $$DQE(f) = \\frac{MTF^2(f)}{NPS(f) \\cdot K_{air}}$$ \n\n Where: \n\n- $\\text{MTF}(f)$ = Modulation Transfer Function (geometric mean of orthogonal edges) \n\n - $\\text{NNPS}(f)$ = Normalized Noise Power Spectrum (radial average) \n\n - $K_{air}$ = Air kerma in μGy \n\n - $f$ = Spatial frequency in lp/mm",
                    "icon": "🎯"
                },
                "Threshold Contrast Detail Detectability (TCDD)": {
                    "description": "Measure low-contrast detectability",
                    "files_needed": "1 image with contrast phantom",
                    "file_types": "RAW/STD files only",
                    "requirements": ["Uniform exposure", "No phantom objects", "Covers full detector area"],
                    "computation": "Computes Threshold-Contrast Detail Detectability by statistical analysis, based on Paruccini et al. (2021, https://doi.org/10.1016/j.ejmp.2021.10.007) report.",
                    "icon": "🎭"
                }
            }
        },
        "Developer Tools": {
            "icon": "🛠️",
            "description": "File conversion and comparison utilities",
            "tests": {
                "Convert to DICOM": {
                    "description": "Convert RAW/image files to DICOM format",
                    "files_needed": "1 image file",
                    "file_types": "RAW, STD, or any image format",
                    "requirements": ["Image file to convert", "Pixel spacing (optional)", "Custom metadata (optional)"],
                    "computation": "Creates DICOM-compliant file using pydicom library. Embeds pixel data, image dimensions, and metadata. Sets PresentationIntentType='FOR PROCESSING' tag. Generates SOP Instance UID and dataset identifiers per DICOM standard.",
                    "icon": "🏥"
                },
                "RAW vs DICOM Comparison": {
                    "description": "Compare RAW and DICOM versions of same image",
                    "files_needed": "2 files (1 RAW + 1 DICOM)",
                    "file_types": "1 RAW + 1 DICOM file",
                    "requirements": ["Same image in both formats", "RAW parameters known", "DICOM has metadata"],
                    "computation": "Pixel-by-pixel comparison between RAW and DICOM arrays. Computes difference map, calculates statistics (mean, max, RMSE). Visualizes discrepancies via difference histogram and spatial difference map. Validates data integrity post-conversion.",
                    "icon": "⚖️"
                }
            }
        },
        "DICOM Analysis": {
            "icon": "🏥",
            "description": "Constancy checks for post-processed DICOM images",
            "tests": {
                "DICOM Post-processing Analysis": {
                    "description": "Compute central ROI SNR and key DICOM metadata across multiple images",
                    "files_needed": "1+ DICOM images",
                    "file_types": "DICOM files only (.dcm/.dicom)",
                    "requirements": [
                        "One or more DICOM files",
                        "Each image must be at least 100x100 pixels",
                        "Pixel data must be decodable from DICOM"
                    ],
                    "computation": "For each uploaded DICOM image, extracts a central $100 \\times 100$ ROI and computes Signal-to-Noise Ratio as $SNR = \\mu/\\sigma$, where $\\mu$ is ROI mean pixel value and $\\sigma$ is ROI standard deviation. The output table includes per-image SNR and DICOM tags (0018,1405) Relative X-ray Exposure and (0018,0015) Body Part Examined.",
                    "icon": "📋"
                }
            }
        }
    }
