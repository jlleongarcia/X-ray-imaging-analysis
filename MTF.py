import streamlit as st
import numpy as np
import pandas as pd
from pylinac.core.mtf import MTF as PylinacMTF # Renaming to avoid confusion

def calculate_mtf_metrics(image_array, pixel_spacing_col):
    """
    Calculates MTF from a given 1D profile.
    For this initial version, it uses the central row of the image_array as a simplified LSF.

    Args:
        image_array (np.ndarray): The input 2D image data.
        pixel_spacing_col (float): Pixel spacing for columns (e.g., mm/pixel), used for dpmm.

    Returns:
        dict: A dictionary containing MTF results.
    """
    if image_array is None or image_array.ndim != 2 or image_array.shape[0] < 1 or image_array.shape[1] < 2:
        # Need at least 2 pixels for a profile
        st.error("Valid 2D image_array with at least 2 columns is required for MTF calculation.")
        return {"MTF_Status": "Error: Invalid image array"}

    if pixel_spacing_col is not None and pixel_spacing_col <= 0:
        st.warning("Pixel spacing (col) is not valid (<=0). MTF spatial frequencies will be in cycles/pixel.")
        dpmm = None
        x_axis_unit = "cycles/pixel"
    elif pixel_spacing_col is None:
        st.warning("Pixel spacing (col) is not available. MTF spatial frequencies will be in cycles/pixel.")
        dpmm = None
        x_axis_unit = "cycles/pixel"
    else:
        dpmm = 1.0 / pixel_spacing_col
        x_axis_unit = "lp/mm"

    # Extract the central row as a simplified LSF.
    profile_values = image_array[image_array.shape[0] // 2, :].astype(float)
    
    if profile_values.size < 2: # MTF calculation needs at least 2 points
        st.error("Extracted profile for MTF has fewer than 2 points.")
        return {"MTF_Status": "Error: Profile too short"}

    try:
        # Assuming the extracted profile is an LSF.
        mtf_obj = PylinacMTF(values=profile_values, dpmm=dpmm, esf=False, normalize=True)

        frequencies = mtf_obj.frequencies.tolist()
        mtf_values = mtf_obj.mtf.tolist()
        
        # Prepare data for st.line_chart
        mtf_data_for_chart = np.array([frequencies, mtf_values]).T

        results = {
            "frequencies": frequencies,
            "mtf_values": mtf_values,
            "mtf_chart_data": mtf_data_for_chart,
            "x_axis_unit": x_axis_unit,
            "source_profile": "Central Row (Simplified LSF)"
        }

        # Calculate MTF at specific percentages (MTF50%, MTF10%)
        try:
            mtf50 = mtf_obj.relative_resolution(50) # pylinac uses percentage
            results["MTF50"] = float(mtf50) if not np.isnan(mtf50) else "N/A"
        except Exception: # pylint: disable=broad-except
            results["MTF50"] = "N/A (calculation error)"
        
        try:
            mtf10 = mtf_obj.relative_resolution(10)
            results["MTF10"] = float(mtf10) if not np.isnan(mtf10) else "N/A"
        except Exception: # pylint: disable=broad-except
            results["MTF10"] = "N/A (calculation error)"

        return results

    except Exception as e:
        st.error(f"Error during MTF calculation: {e}")
        return {"MTF_Status": f"Error: {e}"}

def display_mtf_analysis_section(image_array, pixel_spacing_row, pixel_spacing_col):
    st.subheader("Modulation Transfer Function (MTF) Analysis")
    st.markdown("""
    **Note:** This initial MTF calculation uses the **central row** of the image as a simplified Line Spread Function (LSF).
    For accurate MTF, a well-defined edge or slit from a phantom image is typically required.
    """)

    if st.button("Run MTF Analysis (using central row)"):
        if image_array is None:
            st.error("Please upload an image first.")
            return

        with st.spinner("Calculating MTF..."):
            mtf_results_dict = calculate_mtf_metrics(image_array, pixel_spacing_col)
            
            if "MTF_Status" in mtf_results_dict and "Error" in mtf_results_dict["MTF_Status"]:
                st.error(f"MTF Calculation Failed: {mtf_results_dict['MTF_Status']}")
            elif mtf_results_dict and "mtf_chart_data" in mtf_results_dict:
                st.success("MTF Analysis Complete!")
                st.write(f"MTF calculated from: **{mtf_results_dict.get('source_profile', 'N/A')}**")
                
                st.subheader(f"MTF Curve ({mtf_results_dict['x_axis_unit']})")
                df_mtf = pd.DataFrame(mtf_results_dict["mtf_chart_data"], columns=[mtf_results_dict['x_axis_unit'], 'MTF'])
                st.line_chart(df_mtf.set_index(mtf_results_dict['x_axis_unit']))

                st.write(f"**MTF50% ({mtf_results_dict.get('x_axis_unit','')}):** {mtf_results_dict.get('MTF50', 'N/A')}")
                st.write(f"**MTF10% ({mtf_results_dict.get('x_axis_unit','')}):** {mtf_results_dict.get('MTF10', 'N/A')}")
                
                st.json(mtf_results_dict) # Display all results for inspection
            else:
                st.error("MTF calculation did not return expected results.")

