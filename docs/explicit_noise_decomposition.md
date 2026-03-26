# Explicit Noise Decomposition

**Based on:** Monnin et al., "Comparison of the polynomial model against explicit
measurements of noise components for different mammography systems", *Phys. Med. Biol.* 59 (2014).

---

## 1. Motivation

The polynomial model fits total variance as:

$$\sigma^2(k) = a \cdot k^2 + b \cdot k + c$$

where the coefficients *a*, *b*, and *c* are **indirectly** attributed to structural,
quantum, and electronic noise. However, this attribution relies on the assumption that
the quadratic decomposition cleanly separates the three components — which may not hold
if there are cross-terms or non-ideal detector behaviour.

The **explicit method** measures the structural (fixed-pattern) component directly from
repeated exposures at the same kerma level, providing a model-independent validation.

---

## 2. Data Requirements

| Image set | Count | Purpose |
|-----------|-------|---------|
| Different kerma levels | ≥ 3 (typically 7) | Detector response curve + polynomial σ² fit |
| Same kerma level k₀ | ≥ 2 (recommended ≥ 6) | Explicit noise decomposition |

All images must be acquired under the same beam conditions (kVp, filtration, geometry).
The same-kerma images should be acquired at a clinically relevant dose level, ideally in
the mid-range of the detector's operating interval.

---

## 3. Pipeline

### 3.1 Convert to kerma domain

All pixel values are converted to kerma (µGy) using the detector response curve fitted
in the previous step. This follows the article's methodology:

> "Before noise analysis, all pixel values were converted to photon fluence values
> using the detector response curves."

We use kerma rather than photon fluence since the two are proportional and our detector
response is calibrated in kerma.

### 3.2 Detrending (optional, recommended)

Each frame is detrended by fitting and subtracting a first-order plane (Z = ax + by + c)
from a configurable fraction of the image area (default 80%). This removes low-frequency
spatial trends (heel effect, geometric dome) that would inflate variance measurements.

### 3.3 Extract central ROI

All subsequent steps operate on the central 100×100 ROI of each frame.  Extracting a
small ROI first keeps the structural-noise estimation local and avoids edge artefacts.

### 3.4 Total noise

For each of the N ROIs, compute the spatial sample variance (ddof = 1):

$$S^2_i = \mathrm{Var}_{\text{ROI}}(\mathrm{ROI}_i)$$

The mean total noise across all frames:

$$\overline{S^2} = \frac{1}{N} \sum_{i=1}^{N} S^2_i$$

### 3.5 Pixel-wise mean and division images

For each pixel $(i,j)$ in the ROI, compute its mean across the N frames:

$$\bar{p}(i,j) = \frac{1}{N} \sum_{n=1}^{N} p_n(i,j)$$

Divide each frame's ROI by the pixel-wise mean:

$$d_n(i,j) = \frac{p_n(i,j)}{\bar{p}(i,j)}$$

This gives N **division images**, each with values close to 1.  The division
normalises out the multiplicative fixed-pattern (gain non-uniformity) — what remains
is purely stochastic (quantum + electronic) noise in relative units.

### 3.6 Stochastic noise (quantum + electronic)

Compute the spatial variance of each division image (still within the 100×100 ROI):

$$S^2_{\mathrm{div},n} = \mathrm{Var}_{\text{ROI}}(d_n)$$

Average the N division variances:

$$\overline{S^2_{\mathrm{div}}} = \frac{1}{N} \sum_{n=1}^{N} S^2_{\mathrm{div},n}$$

Apply the Bessel-like correction (the mean image contains 1/N of each frame's own
stochastic noise, introducing correlation):

$$S^2_{\mathrm{stoch,rel}} = \frac{N}{N-1} \cdot \overline{S^2_{\mathrm{div}}}$$

Convert from relative (dimensionless) to absolute units (µGy²):

$$S^2_{\mathrm{stoch}} = S^2_{\mathrm{stoch,rel}} \times \langle\bar{p}\rangle^2$$

where $\langle\bar{p}\rangle$ is the mean value of the pixel-wise mean ROI.

### 3.7 Fixed-pattern (structural) noise

By subtraction:

$$S^2_{\mathrm{fp}}(k_0) = S^2_{\mathrm{total}}(k_0) - S^2_{\mathrm{stoch}}(k_0)$$

### 3.8 Predict structural noise at all kerma levels

After detrending, the residual structural variance scales as k²:

$$S^2_{\mathrm{fp}}(k) = \frac{S^2_{\mathrm{fp}}(k_0)}{k_0^2} \cdot k^2$$

This allows predicting S²_fp at each of the different kerma levels used in the
polynomial fit.

### 3.9 Electronic noise (extrapolation to k → 0)

Compute stochastic variance at each kerma level:

$$S^2_{\mathrm{stoch}}(k) = S^2_{\mathrm{total}}(k) - S^2_{\mathrm{fp,predicted}}(k)$$

Fit a weighted linear model (weights = 1/S²_stoch):

$$S^2_{\mathrm{stoch}}(k) = \beta \cdot k + S^2_e$$

The intercept S²_e is the electronic noise variance (eq. 10 in Monnin et al.):

$$S^2_e = \lim_{k \to 0} S^2_{\mathrm{stoch}}(k)$$

### 3.10 Quantum noise

By subtraction:

$$S^2_q(k) = S^2_{\mathrm{stoch}}(k) - S^2_e = \beta \cdot k$$

---

## 4. Comparison with Polynomial Model

| Component | Polynomial model | Explicit method |
|-----------|-----------------|-----------------|
| Structural | a·k² (fitted) | Measured directly from N frames |
| Electronic | c (fitted intercept) | Extrapolation to k → 0 |
| Quantum | b·k (fitted) | By subtraction |
| Assumption | Clean quadratic decomposition | Structural scales as k² after detrending |
| Data needed | 1 image per kerma level | N ≥ 2 images at same kerma |

The explicit method serves as **independent validation** of the polynomial model.
If the polynomial a·k² matches the explicitly measured S²_fp, and the polynomial c
matches the extrapolated S²_e, the polynomial model is well-calibrated for that
detector.

---

## 5. Key Physics Notes

- **Why division, not subtraction?** The fixed-pattern noise is multiplicative
  (gain non-uniformity), so dividing by the mean image normalises it out. Subtraction
  would only remove additive patterns.

- **N/(N−1) correction:** The mean image contains 1/N of each frame's own stochastic
  noise, which is correlated with that frame. The correction compensates for this,
  analogous to Bessel's correction for sample variance.

- **Kerma domain:** Working in kerma domain (rather than pixel domain) ensures the
  variance has physical units (µGy²) and is directly comparable across detector types
  and operating modes.

- **Stochastic noise = quantum + electronic:** Both quantum and electronic noise are
  random and frame-independent. Quantum variance scales linearly with k, while
  electronic variance is constant — hence the linear extrapolation to separate them.

---

## 6. Implementation

The implementation is in `src/qa/flat_panel_qa/detector_conversion.py`:

- `_explicit_noise_decomposition()` — Core computation (steps 3.3–3.5)
- `_extrapolate_electronic_noise()` — Linear extrapolation (steps 3.7–3.8)
- UI section "Explicit Noise Decomposition" in `display_detector_conversion_section()`

### UI Workflow

1. Fit detector response curve using images at different kerma levels
2. Run the σ² vs Kerma polynomial fit
3. Upload N ≥ 2 images at the same kerma level k₀
4. Enter k₀ and click "Run explicit noise decomposition"
5. View results: S² total, stochastic, fixed-pattern, electronic and quantum decomposition

---

## 7. Reference

Monnin, P., et al. "Comparison of the polynomial model against explicit measurements
of noise components for different mammography systems." *Physics in Medicine and
Biology*, 59(5), 2014.
