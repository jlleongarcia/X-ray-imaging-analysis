# Flat Panel QA — `src/qa/flat_panel_qa/`

## Purpose

The `flat_panel_qa` package is the **scientific core** of the X-ray Imaging Analysis Toolkit. It implements a comprehensive suite of image quality metrics for characterizing flat panel digital X-ray detectors, following the methodologies defined in **IEC 62220-1-1:2015** (*Medical electrical equipment — Characteristics of digital X-ray imaging devices — Part 1-1: Determination of the detective quantum efficiency — Detectors used in radiographic imaging*) and related standards.

These modules operate on **raw, unprocessed detector data** ("FOR PROCESSING" images) to characterize the intrinsic performance of the detector hardware, independent of any manufacturer post-processing.

---

## Architecture Overview

```
src/qa/flat_panel_qa/
├── detector_conversion.py     ← Detector response characterization (MPV, EI, σ² vs kerma)
├── uniformity.py              ← Spatial uniformity (global & local, IEC method)
├── mtf.py                     ← Modulation Transfer Function (slanted edge, IEC 62220-1-1)
├── nps.py                     ← Noise Power Spectrum (2D FFT averaging, IEC 62220-1-1)
├── dqe.py                     ← Detective Quantum Efficiency (MTF² / NNPS·K, IEC 62220-1-1)
└── threshold_contrast.py      ← Threshold Contrast Detail Detectability (statistical method)
```

### Analysis Pipeline

The modules form a **logical pipeline** where earlier analyses feed into later ones:

```
 Detector Response Curve       ← Foundation: maps pixel values to physical dose
         │
         ├──► Uniformity       ← Optional kerma-domain analysis using inverse fit
         │
         ├──► MTF              ← Spatial resolution (independent of dose)
         │         │
         │         ▼
         ├──► NPS              ← Noise spectrum (uses kerma for normalization)
         │         │
         │         ▼
         └──► DQE              ← Synthesizes MTF + NPS + kerma into single figure of merit
                   
         TCDD                  ← Statistical contrast detectability (standalone)
```

---

## Module Reference

### `detector_conversion.py` — Detector Response Characterization

This module characterizes the detector's response curve — the fundamental relationship between incident radiation dose (air kerma) and the detector's digital output. It performs three distinct analyses on a series of flat-field exposures acquired at different dose levels.

#### Input Requirements

- **3 or more RAW flat-field images** acquired at different exposure levels (varying mAs or kV).
- For each image, the user provides the corresponding **air kerma value** (μGy) and optionally the **Exposition Index** (EI).

#### Analysis 1: Mean Pixel Value (MPV) vs Air Kerma

For each image, the module extracts a **100 × 100 pixel central ROI** and computes its mean pixel value. The relationship between MPV and air kerma is then fitted using one of three models:

| Method | Model | Formula |
|--------|-------|---------|
| Linear | $\text{MPV} = a \cdot K + b$ | First-order polynomial |
| Logarithmic | $\text{MPV} = a \cdot \ln(K) + b$ | Log-linear fit |
| Polynomial | $\text{MPV} = \sum_{i=0}^{n} c_i \cdot K^i$ | 2nd or 3rd degree polynomial |

The fit method can be **auto-selected** from embedded DICOM metadata — if the `(0028,1040) PixelIntensityRelationship` tag contains `"LOG"`, the logarithmic model is preselected.

The fit is characterized by $R^2$ (coefficient of determination) and maximum deviation. The resulting prediction function ($K \rightarrow \text{MPV}$) and its inverse ($\text{MPV} \rightarrow K$) are cached in session state for use by other modules (uniformity, NPS, σ² analysis).

#### Analysis 2: Exposition Index (EI) vs Air Kerma

Always fitted with a linear model: $\text{EI} = a \cdot K + b$. The EI is a DICOM-standardized exposure indicator defined in **IEC 62494-1** that allows comparison of exposure levels across different manufacturers.

#### Analysis 3: Noise Variance (σ²) vs Air Kerma

This is the most sophisticated analysis in the module. It decomposes the detector's noise into its physical components:

$$\sigma^2(K) = a \cdot K^2 + b \cdot K + c$$

where:
- **$a \cdot K^2$**: Structural noise (fixed pattern noise, independent of dose — gain non-uniformity, detector element defects).
- **$b \cdot K$**: Quantum noise (Poisson-limited, proportional to X-ray photon count).
- **$c$**: Electronic noise (dark current, readout electronics, independent of exposure).

**Processing pipeline for σ² analysis:**

1. **Domain conversion**: Each image's 100 × 100 ROI is converted from pixel values to kerma values using the inverse fit from Analysis 1.
2. **Detrending** (`_detrend_roi()`): A planar fit $Z = ax + by + c$ is subtracted from the ROI to remove the heel effect (anode heat gradient) and geometric intensity dome. This isolates noise from structural signal variations.
3. **Bootstrap variance estimation** (`_bootstrap_variance()`): The detrended ROI is resampled 500 times to estimate the variance of the variance — providing a weight for each data point in the subsequent fit.
4. **Constrained weighted fit** (`_constrained_weighted_fit()`): Non-negative constrained least squares with multi-start optimization ensures physically meaningful coefficients ($a, b, c \geq 0$).
5. **Quantum noise dominance interval** (`_compute_dominance_interval()`): Computes the kerma range where quantum noise dominates:
   - $K_{\min} = c / b$ (quantum noise exceeds electronic noise).
   - $K_{\max} = b / a$ (quantum noise exceeds structural noise).
   - This interval represents the optimal operating range for the detector.

### `uniformity.py` — Spatial Uniformity Metrics

Evaluates the spatial uniformity of the detector's response across its active area, following IEC methodology.

#### Input

- **1 flat-field RAW image** (uniform exposure).
- Pixel spacing (mm/pixel).
- Optional: detector response inverse fit for kerma-domain analysis.

#### Algorithm

1. **Central ROI extraction**: The central 80% of the image area is extracted (using $\sqrt{0.8}$ scaling factor on each dimension) to avoid edge artifacts.

2. **Moving ROI grid**: A 30 mm × 30 mm ROI slides across the central region with a 15 mm step, computing MPV, SD, and SNR for each position.

3. **Global Uniformity (GU)**:

$$\text{GU}_{\text{PV}} = \max_i \left( \frac{|\text{PV}_i - \overline{\text{PV}}|}{|\overline{\text{PV}}|} \right) \times 100\%$$

$$\text{GU}_{\text{SNR}} = \max_i \left( \frac{|\text{SNR}_i - \overline{\text{SNR}}|}{|\overline{\text{SNR}}|} \right) \times 100\%$$

4. **Local Uniformity (LU)**: Same metric but computed relative to the mean of each ROI's 8 immediate neighbors, detecting localized defects:

$$\text{LU}_{\text{PV}} = \max_i \left( \frac{|\text{PV}_i - \overline{\text{PV}}_{\text{neighbors}}|}{|\overline{\text{PV}}_{\text{neighbors}}|} \right) \times 100\%$$

5. **Kerma-domain option**: When the detector response fit is available, the image can be converted from pixel values to kerma values before computing uniformity, providing a physics-based assessment.

### `mtf.py` — Modulation Transfer Function

Measures the detector's spatial resolution using the **slanted edge method** as specified in IEC 62220-1-1:2015.

#### Input

- **1 or 2 RAW images** of an edge phantom (lead or tungsten edge, slightly tilted).
- For DQE computation: 2 orthogonal edges required (one near-vertical, one near-horizontal).

#### Algorithm

1. **ROI selection**: User selects a region containing the edge via percentage-based sliders (center position + ROI size).

2. **Edge detection**: Hough Transform determines the edge angle and whether it is primarily vertical or horizontal.

3. **IEC slanted edge method**:
   - **Edge Spread Function (ESF)**: Pixel values projected perpendicular to the edge, oversampled by the slant angle.
   - **Line Spread Function (LSF)**: Numerical derivative of the ESF ($\text{LSF} = d\text{ESF}/dx$).
   - **MTF**: Magnitude of the Fourier Transform of the LSF ($\text{MTF}(f) = |\mathcal{F}\{\text{LSF}\}|$).

4. **Key metrics**:
   - **MTF50**: Spatial frequency where MTF = 50% (practical resolution limit).
   - **MTF10**: Spatial frequency where MTF = 10% (high-frequency cutoff).
   - **Nyquist frequency**: $f_{\text{Ny}} = 1 / (2 \cdot \Delta x)$ where $\Delta x$ is pixel spacing.

5. **IEC angle compliance**: The standard requires edge angles of **3°–5°** (horizontal) or **85°–87°** (vertical) relative to the pixel grid. The UI reports whether the detected angle falls within the IEC-optimal range.

6. **Geometric mean MTF**: When two orthogonal edge images are provided, the isotropic MTF is computed as:

$$\text{MTF}_{\text{geom}}(f) = \sqrt{\text{MTF}_{\text{V}}(f) \times \text{MTF}_{\text{H}}(f)}$$

This geometric mean is required by IEC 62220-1-1 for the DQE computation.

### `nps.py` — Noise Power Spectrum

Quantifies the frequency-dependent noise characteristics of the detector, following IEC 62220-1-1:2015.

#### Input

- **1 or more uniform RAW images** (flat-field exposures).
- Pixel spacing (mm/pixel).
- Air kerma value (μGy) — can be auto-detected from the detector response fit.

#### Algorithm

1. **Multi-image support**: Multiple images of the same exposure increase the number of independent noise samples, improving the NNPS estimate.

2. **ROI extraction**:
   - A large central ROI (default 1024 px, adjustable 25–250 mm) is extracted from each image.
   - The large ROI is subdivided into small ROIs (default 128 px, adjustable 8–512 px) for ensemble averaging.

3. **2D FFT**: Each small ROI undergoes a 2D Fast Fourier Transform. The power spectra are averaged across all ROIs and all images.

4. **Normalization (NNPS)**:

$$\text{NNPS}(u, v) = \frac{\Delta x \cdot \Delta y}{N_x \cdot N_y} \cdot \frac{|\text{FFT}|^2}{\mu^2}$$

where $\Delta x, \Delta y$ are pixel spacings, $N_x, N_y$ are ROI dimensions, and $\mu$ is the mean pixel value. Units are converted to $\mu m^2$.

5. **1D profiles**:
   - **Radial average**: Isotropic 1D NNPS for DQE computation.
   - **X-axis and Y-axis components**: Directional profiles for detecting anisotropic noise (e.g., scan-line artifacts).

6. **IEC guideline**: A warning is issued if the total number of analyzed pixels is below 4 million (the IEC threshold for statistical reliability).

7. **Kerma-domain**: When the detector response fit is available, the NPS can be computed in physical dose units rather than pixel value units.

### `dqe.py` — Detective Quantum Efficiency

The **figure of merit** for a digital X-ray detector, synthesizing spatial resolution (MTF), noise (NNPS), and dose (kerma) into a single frequency-dependent metric. Defined by IEC 62220-1-1:2015 as:

$$\text{DQE}(f) = \frac{\text{MTF}^2(f)}{\text{NNPS}(f) \cdot K_{\text{air}}}$$

#### Prerequisites

DQE computation requires **both** MTF and NPS analyses to have been completed in the same session:

- **MTF cache**: Must contain the geometric mean MTF (two orthogonal edge images).
- **NPS cache**: Must contain the NNPS radial average and the kerma value used.

#### Algorithm

1. **Validation**: Both caches are checked for existence, completeness, and the presence of geometric mean MTF.
2. **Frequency alignment**: MTF and NNPS are interpolated to a common frequency grid (the intersection of their respective frequency ranges).
3. **DQE computation**: $\text{DQE}(f) = \text{MTF}^2(f) / (\text{NNPS}(f) \cdot K)$, with NNPS converted from $\mu m^2$ to $mm^2$.
4. **Physical clamping**: DQE values are clamped to $[0, 1]$ — a DQE above 1 is physically impossible (it would mean the detector creates information from noise).
5. **Key metrics**:
   - **DQE(0)**: Low-frequency detective quantum efficiency (overall dose efficiency).
   - **Frequency at 50% of DQE(0)**: Practical resolution limit when noise-limited.
   - **Frequency at 10% of DQE(0)**: High-frequency cutoff.

#### Interpretation

| DQE(0) Range | Interpretation |
|-------------|----------------|
| 0.6–0.8 | Excellent dose efficiency (modern CsI/a-Si or Se-based panels) |
| 0.4–0.6 | Good (typical for well-maintained systems) |
| 0.2–0.4 | Fair (may indicate aging or suboptimal configuration) |
| < 0.2 | Poor (investigation warranted) |

### `threshold_contrast.py` — Threshold Contrast Detail Detectability (TCDD)

Estimates the minimum contrast that can be detected in a uniform phantom as a function of detail size, using the **statistical method** described by Chao et al. (2000).

#### Input

- **1 uniform RAW image** (flat-field or uniform phantom exposure).
- Pixel spacing (mm/pixel).
- Optional: detector response fit from session state for kerma-domain analysis.

#### Algorithm

1. **Kerma-domain conversion** (optional): If the detector response fit is available in session state (`session_state['detector_conversion']['fit']`), the entire image is converted from pixel values to kerma values (μGy) using the cached `predict_mpv` inverse function. This ensures that the contrast thresholds are expressed in physical dose units rather than arbitrary pixel values. If no detector conversion is available, the analysis proceeds in pixel domain.

2. **Central ROI extraction**: A 1024 × 1024 pixel central ROI (adjustable).

3. **Normalization ROI**: A smaller 120 × 120 pixel central ROI provides the mean value (in kerma or pixel units, depending on step 1) for percentage normalization.

4. **Sub-ROI sampling**: For each tested detail size (2–128 pixels), a fixed number of sub-ROIs (default 49) are extracted from the central ROI using grid-based sampling with slight randomization.

4. **Statistical threshold contrast**:
   - For each detail size: compute the mean pixel value of each sub-ROI.
   - Compute the standard deviation of these means: $\sigma_\chi$.
   - Threshold contrast at 95% confidence: $C_T = 3.29 \times \sigma_\chi$.
   - The factor 3.29 corresponds to the one-sided 95th percentile of the normal distribution.

5. **Normalization**: $C_T(\%) = (C_T / \mu_{\text{norm}}) \times 100$, expressing threshold contrast as a percentage of the mean signal.

6. **Curve fitting**: The contrast-detail relationship is fitted to:

$$C_T(d) = \frac{c}{d^2} + \frac{b}{d} + a$$

where $d$ is the detail diameter in mm, and $a, b, c$ are fitted coefficients. Goodness of fit is reported as $R^2$ and RMSE.

7. **Key outputs**: $C_T$ at standard detail sizes (0.5 mm, 2.0 mm) for cross-system comparison.

---

## IEC 62220-1-1:2015 Compliance Summary

| IEC Requirement | Module | Implementation |
|----------------|--------|---------------|
| Slanted edge MTF | `mtf.py` | Hough-based edge detection, ESF → LSF → FFT |
| Edge angle 3°–5° or 85°–87° | `mtf.py` | Angle detection with compliance indicator |
| Geometric mean of orthogonal MTFs | `mtf.py` | Automatic when 2 images provided |
| NPS from ≥ 4M pixels | `nps.py` | Multi-image support with pixel count warning |
| NNPS normalization | `nps.py` | $\text{NNPS} = \text{NPS} / \mu^2$ with unit conversion |
| DQE = MTF² / (NNPS × K) | `dqe.py` | Full computation with physical clamping |
| Detector response linearity | `detector_conversion.py` | MPV vs kerma fit with multiple models |

---

## Session State Integration

The flat panel QA modules share data through Streamlit session state:

```
session_state['detector_conversion']
  ├── 'fit'      → MPV vs kerma fit (coefficients, prediction function)
  ├── 'ei_fit'   → EI vs kerma fit
  ├── 'sd2_fit'  → σ² vs kerma fit (noise decomposition)
  └── 'results'  → Per-file kerma, MPV, SD summaries

session_state['mtf_cache']
  ├── 'results'              → List of per-image MTF results
  ├── 'mtf_geometric_mean'   → Isotropic MTF (if orthogonal edges)
  └── 'timestamp'

session_state['nps_cache']
  ├── 'results'     → NNPS data arrays and metadata
  ├── 'kerma_ugy'   → Air kerma used for NPS normalization
  └── 'timestamp'
```

This caching architecture enables the DQE computation to consume pre-computed MTF and NPS results without re-running those analyses, and allows the uniformity and NPS modules to optionally operate in kerma-domain using the detector response inverse function.

---

## Project Evolution

The flat panel QA suite evolved through several phases visible in the commit history:

1. **Initial NPS and MTF** (early commits): Basic implementations, NPS initially for single images only.
2. **Detector response fitting**: Added MPV vs kerma fitting with linear and logarithmic models.
3. **Noise analysis**: Introduced σ² = aK² + bK + c noise decomposition with weighted fitting.
4. **Detrending**: Added heel effect removal via planar detrending for more accurate noise estimation.
5. **Bootstrap resampling**: Robust variance-of-variances weighting replaced simpler heuristics.
6. **Constrained optimization**: Non-negative multi-start fitting ensures physically meaningful noise coefficients.
7. **MTF overhaul**: Migrated to a custom pylinac fork for proper Hough-based edge detection and IEC-compliant angle handling.
8. **DQE synthesis**: Combined MTF geometric mean with NNPS and kerma for the complete IEC figure of merit.
9. **TCDD**: Added statistical contrast-detail analysis as a complementary metric independent of the IEC pipeline.
10. **Centralized decoding**: All modules now consume preloaded payloads from the centralized image loader, ensuring consistent RAW decoding.
