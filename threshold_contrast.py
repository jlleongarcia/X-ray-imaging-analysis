import streamlit as st
import numpy as np
from scipy.integrate import quad
from scipy.special import j1
from scipy.interpolate import interp1d

# --- Component Functions (from our previous discussion) ---

def signal_spectrum_circular(f: np.ndarray, R: float, m: float) -> np.ndarray:
    """
    Calculates the signal spectrum S(f) for a circular object.
    S(f) = (R/f) * J1(2*pi*R*f/m)
    """
    # Avoid division by zero at f=0. The limit of S(f) as f->0 is pi*R^2/m.
    # However, the integrand includes f, so the value at f=0 is 0.
    # We can safely replace f=0 with a small number to avoid warnings.
    f_safe = np.where(f == 0, 1e-9, f)
    argument = 2 * np.pi * R * f_safe / m
    return (R / f_safe) * j1(argument)

def visual_transfer_function(f: np.ndarray, m: float) -> np.ndarray:
    """
    Calculates the Visual Transfer Function (VTF) of the human eye.
    VTF(f) = 29 * (f/m)^2 * exp(-4*f/m)
    """
    fm = f / m
    return 29 * (fm**2) * np.exp(-4 * fm)

# --- Main Calculation Function ---

def calculate_threshold_contrast_advanced(
    object_radius: float,
    magnification: float,
    mtf_data: np.ndarray,
    nnps_data: np.ndarray,
    nyquist_freq: float, # This is f_n, the upper limit for the numerator integral
    snr_t: float = 3.0
) -> float:
    """Calculates threshold contrast using a comprehensive imaging model."""
    # 1. Create interpolation functions for MTF and NNPS
    mtf_interp = interp1d(mtf_data[:, 0], mtf_data[:, 1], bounds_error=False, fill_value=0)
    nnps_interp = interp1d(nnps_data[:, 0], nnps_data[:, 1], bounds_error=False, fill_value=0)

    # 2. Define the integrands
    def numerator_integrand(f):
        S_f = signal_spectrum_circular(f, object_radius, magnification)
        MTF_f = mtf_interp(f)
        VTF_f = visual_transfer_function(f, magnification)
        NNPS_f = nnps_interp(f)
        return S_f**2 * MTF_f**2 * VTF_f**4 * NNPS_f * f

    def denominator_integrand(f):
        S_f = signal_spectrum_circular(f, object_radius, magnification)
        MTF_f = mtf_interp(f)
        VTF_f = visual_transfer_function(f, magnification)
        return S_f**2 * MTF_f**2 * VTF_f**2 * f

    # 3. Perform numerical integrations (denominator integral uses max frequency from MTF data)
    max_freq = np.max(mtf_data[:, 0])
    numerator_integral, _ = quad(numerator_integrand, 0, nyquist_freq, limit=100)
    denominator_integral, _ = quad(denominator_integrand, 0, max_freq, limit=100) # Reverted to max_freq

    if denominator_integral == 0:
        return float('inf')

    # 4. Assemble the final formula
    ct = (snr_t * np.sqrt(numerator_integral)) / (np.sqrt(2 * np.pi) * denominator_integral)
    return ct

# --- Streamlit Display Function ---

def display_threshold_contrast_section():
    """Renders the Threshold Contrast section in the Streamlit app."""
    st.header("Threshold Contrast (C_t) Calculation")

    # --- Step 1: Check for necessary data in session_state ---
    st.markdown("This calculation requires MTF and NNPS data. Please run those analyses first.")

    mtf_ready = 'mtf_data' in st.session_state and st.session_state['mtf_data'] is not None
    nnps_ready = 'nnps_data' in st.session_state and st.session_state['nnps_data'] is not None

    col_status1, col_status2 = st.columns(2)
    with col_status1:
        st.metric("MTF Data Status", "LOADED ✅" if mtf_ready else "MISSING ⚠️")
    with col_status2:
        st.metric("NNPS Data Status", "LOADED ✅" if nnps_ready else "MISSING ⚠️")

    st.markdown("---")

    # --- Step 2: Get user inputs ---
    st.subheader("Object and System Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        sid = st.number_input("Source-to-Image Distance (SID) in cm", min_value=1.0, value=100.0, step=0.1, format="%.1f")
    with col2:
        sod = st.number_input("Source-to-Object Distance (SOD) in cm", min_value=1.0, value=90.0, step=0.1, format="%.1f")
    with col3:
        object_radius = st.number_input("Object Radius (R) in mm", min_value=0.01, value=0.5, step=0.01, format="%.2f")

    nyquist_freq = st.number_input("Nyquist Frequency (cy/mm)", min_value=0.1, value=5.0, help="Typically 1/(2*pixel_spacing)")

    # Display the Nyquist frequency and max MTF frequency as outputs
    st.subheader("Key Frequencies for Calculation")
    st.metric("Nyquist Frequency (f_n)", f"{nyquist_freq:.2f} cy/mm")
    if mtf_ready:
        st.metric("Max MTF Frequency (f_max)", f"{np.max(st.session_state['mtf_data'][:, 0]):.2f} cy/mm")

    # --- Step 3: Run calculation ---
    if st.button("Calculate Threshold Contrast", disabled=not (mtf_ready and nnps_ready)):
        with st.spinner("Calculating..."):
            magnification = sid / sod
            ct = calculate_threshold_contrast_advanced(
                object_radius=object_radius, magnification=magnification, mtf_data=st.session_state['mtf_data'],
                nnps_data=st.session_state['nnps_data'], nyquist_freq=nyquist_freq
            )
            st.metric(label=f"Threshold Contrast for a {object_radius*2:.1f} mm Object", value=f"{ct:.4f}")
            st.info(f"A lower value indicates better detectability. This object needs to be at least {ct:.2%} brighter or darker than its background to be seen.")