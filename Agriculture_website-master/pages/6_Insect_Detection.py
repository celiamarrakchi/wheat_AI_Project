import streamlit as st
import os
import sys
import tempfile

# Streamlit configuration - this must be the first Streamlit command in the script
st.set_page_config(
    page_title="🪲 Insect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the absolute path to the srcInsect folder to prioritize the correct crew.py
SRCINSECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "srcInsect"))
if SRCINSECT_PATH not in sys.path:
    sys.path.insert(0, SRCINSECT_PATH)

# Import from srcInsect
from crew import run_insect_classification_agent
from tools.custom_tool import recommend_insect_info  # No need for full module path due to sys.path

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

# App title
st.title("🪲 Insect Detection & Information")
st.write("Upload an image of an insect to classify and get helpful information.")

# File uploader for insect image
uploaded_file = st.file_uploader("Choose an insect image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            temp_file_path = tmp_file.name
            tmp_file.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        # Classify the insect when the button is clicked
        if st.button("Classify Insect"):
            with st.spinner("Classifying insect..."):
                insect_name = run_insect_classification_agent(temp_file_path)

                if insect_name:
                    st.success(f"🧠 Predicted Insect: {insect_name}")
                    info = recommend_insect_info(insect_name)
                    st.text_area("📘 Insect Information:", info, height=300)
                else:
                    st.error("❌ Insect classification failed. Please try again with a different image.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
