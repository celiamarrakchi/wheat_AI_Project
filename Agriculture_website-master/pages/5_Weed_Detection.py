import streamlit as st
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import io

# Page configuration - must be run first
st.set_page_config(
    page_title="Weed Detection",
    page_icon="🌿",
    layout="wide"
)

# Add root directory to PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from srcHechmi.weed_agent.tools.custom_tool import WeedClassifierTool

# Custom CSS for agricultural theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5dc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    /* Sidebar styling - fixed dark color */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1e3a32 !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("🌿 Weed Detection")

# Description
st.markdown("""
This page allows you to detect the presence of weeds in your agricultural field images.  
Upload an image to begin the analysis.
""")

# Image upload area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Display the image
        st.image(image, caption="Selected Image", use_container_width=True)

        # Button to start analysis
        if st.button("Run Analysis"):
            try:
                # Create an instance of WeedClassifierTool
                classifier = WeedClassifierTool()
                
                # Save the image temporarily for analysis
                temp_dir = os.path.join(root_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save the image
                image.save(temp_path)
                
                # Run the classification
                result = classifier._run(temp_path)
                
                # Convert JSON result to dictionary
                import json
                result_dict = json.loads(result)

                # Display the results
                if "error" in result_dict:
                    st.error(result_dict["error"])
                else:
                    st.success(f"Prediction: {'Weed detected' if result_dict['prediction'] == 1 else 'No weed detected'}")
                    st.info("Recommendations:")
                    st.write(result_dict["solution"])

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred while loading the image: {str(e)}")
else:
    st.warning("Please upload an image to analyze.")
