import numpy as np
import streamlit as st


def render_metadata_summary(
    image_array,
    pixel_spacing_row,
    pixel_spacing_col,
    domain='pixel',
    filename=None,
    title='Image Metadata Summary',
):
    """Render a consistent metadata summary block across analysis sections."""
    st.markdown(f"### {title}")
    if filename:
        st.write(f"**Filename:** {filename}")

    if isinstance(image_array, np.ndarray) and image_array.ndim == 2:
        st.write(f"**Dimensions:** {image_array.shape[1]} x {image_array.shape[0]} pixels")
        st.write(f"**Data Type:** {image_array.dtype}")
    else:
        st.write("**Dimensions:** N/A")
        st.write("**Data Type:** N/A")

    if pixel_spacing_row and pixel_spacing_col:
        st.write(f"**Pixel Spacing:** {pixel_spacing_row:.3f} x {pixel_spacing_col:.3f} mm/px")
    else:
        st.write("**Pixel Spacing:** N/A")

    st.write(f"**Domain:** {domain}")
