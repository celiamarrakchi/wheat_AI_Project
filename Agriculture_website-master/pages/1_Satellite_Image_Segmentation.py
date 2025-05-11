import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.colors as mcolors
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Agricultural Image Segmentation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agricultural theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5dc;  /* Beige background */
    }
    .stButton button {
        background-color: #4CAF50;  /* Green */
        color: white;
    }
    h1, h2, h3 {
        color: #2e7d32;  /* Dark green */
    }
    /* Sidebar styling - fixed dark color */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1e3a32 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Define custom metric (same as in your model.py)


def jaccard_coef(y_true, y_pred):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
    union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat) - intersection
    return (intersection + 1.0) / (union + 1.0)

# Create a colorful custom colormap for agriculture


def create_agriculture_colormap():
    # Colors representing different agricultural features
    colors = [
        '#8c510a',  # Brown for soil
        '#01665e',  # Dark green for trees
        '#35978f',  # Teal for crops
        '#80cdc1',  # Light teal for young crops
        '#c7eae5',  # Very light teal for sprouting
        '#f6e8c3',  # Tan for dry grass
        '#dfc27d',  # Light brown for harvested fields
        '#bf812d',  # Medium brown for tilled soil
        '#543005',  # Dark brown for wet soil
        '#003c30',  # Very dark green for dense vegetation
        '#01665e',  # Dark teal for irrigation
        '#35978f',  # Medium teal for healthy crops
        '#80cdc1',  # Light blue-green for water features
        '#c7eae5',  # Very light blue for mist/fog
        '#f5f5f5',  # White for structures/buildings
    ]
    return mcolors.ListedColormap(colors)


# App title and description
st.title("🌾 Agricultural Image Segmentation")
st.markdown("""
This app uses a deep learning model to segment agricultural satellite imagery.
Upload your image to see the segmentation results!
""")

# Add a back button to return to home
if st.button("← Back to Home"):
    st.switch_page("app.py")

# Sidebar for options
st.sidebar.header("Options")

# Model loading


@st.cache_resource
def load_segmentation_model():
    model_path = r'C:\Users\merye\Desktop\Agriculture_website-master\Agriculture_website-master\model_epoch_150.h5'

    # nahi eli foukha

    # model_path = 'model_epoch_150.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please make sure it exists in the current directory.")
        return None

    model = load_model(model_path, custom_objects={'jaccard_coef': jaccard_coef}, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[jaccard_coef])
    return model


# Load model
model = load_segmentation_model()

if model is None:
    st.warning("Please place the model file 'model_epoch_150.h5' in the app directory and restart.")
    st.stop()

# Image upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Visualization options
st.sidebar.header("Visualization")
colormap_option = st.sidebar.selectbox(
    "Select colormap for segmentation",
    ["Agriculture Theme", "jet", "viridis", "plasma", "inferno", "magma", "cividis"]
)

alpha_value = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.7)
show_legend = st.sidebar.checkbox("Show Legend", True)

# Process the uploaded image
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Check if image is grayscale and convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Convert RGBA to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Display original image
    st.header("Original Image")
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    target_size = (256, 256)
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    with st.spinner("Generating segmentation mask..."):
        pred_mask = model.predict(img_input)

        # Check if the prediction has a channel dimension
        if pred_mask.shape[-1] > 1:
            # Multi-class segmentation
            pred_mask_class = np.argmax(pred_mask, axis=-1).squeeze()
            num_classes = pred_mask.shape[-1]
        else:
            # Binary segmentation
            pred_mask_class = (pred_mask.squeeze() > 0.5).astype(np.uint8)
            num_classes = 2

    # Choose colormap
    if colormap_option == "Agriculture Theme":
        cmap = create_agriculture_colormap()
    else:
        cmap = plt.get_cmap(colormap_option)

    # Create figure for visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax[0].imshow(img_resized)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Display segmentation mask
    im = ax[1].imshow(pred_mask_class, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 1))
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    # Add colorbar if requested
    if show_legend:
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

    # Create overlay of segmentation on original image
    st.header("Segmentation Overlay")

    # Create a new figure for the overlay
    fig_overlay, ax_overlay = plt.subplots(figsize=(8, 8))

    # Show original image
    ax_overlay.imshow(img_resized)

    # Overlay segmentation with transparency
    masked = np.ma.masked_where(pred_mask_class == 0, pred_mask_class)
    ax_overlay.imshow(masked, cmap=cmap, alpha=alpha_value, vmin=0, vmax=max(num_classes - 1, 1))

    ax_overlay.set_title("Segmentation Overlay")
    ax_overlay.axis("off")

    st.pyplot(fig_overlay)

    # Add download buttons for the visualizations
    st.header("Download Results")

    # Save figures to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    buf_overlay = io.BytesIO()
    fig_overlay.savefig(buf_overlay, format="png", dpi=300, bbox_inches="tight")
    buf_overlay.seek(0)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Segmentation Results",
            data=buf,
            file_name="segmentation_result.png",
            mime="image/png"
        )

    with col2:
        st.download_button(
            label="Download Overlay Image",
            data=buf_overlay,
            file_name="segmentation_overlay.png",
            mime="image/png"
        )

    # Display some statistics
    st.header("Segmentation Statistics")

    # Calculate some basic statistics
    if num_classes > 2:
        # For multi-class segmentation
        class_counts = [np.sum(pred_mask_class == i) for i in range(num_classes)]
        class_percentages = [count / (target_size[0] * target_size[1]) * 100 for count in class_counts]

        # Create a bar chart of class distributions
        fig_stats, ax_stats = plt.subplots(figsize=(10, 6))
        bars = ax_stats.bar(range(num_classes), class_percentages)

        # Color the bars according to the colormap
        for i, bar in enumerate(bars):
            bar.set_color(cmap(i / max(1, num_classes - 1)))

        ax_stats.set_xlabel("Class")
        ax_stats.set_ylabel("Percentage of Image (%)")
        ax_stats.set_title("Class Distribution")
        ax_stats.set_xticks(range(num_classes))

        st.pyplot(fig_stats)
    else:
        # For binary segmentation
        foreground_pixels = np.sum(pred_mask_class == 1)
        total_pixels = target_size[0] * target_size[1]
        foreground_percentage = foreground_pixels / total_pixels * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Foreground Pixels", f"{foreground_pixels:,}")
            st.metric("Total Pixels", f"{total_pixels:,}")

        with col2:
            st.metric("Foreground Percentage", f"{foreground_percentage:.2f}%")
            st.metric("Background Percentage", f"{100 - foreground_percentage:.2f}%")

else:
    # Display sample images if no upload
    st.info("Please upload an image to see the segmentation results.")

    # Show a sample of what the app does
    st.header("Sample Visualization")

    # Try to load the sample image from the model.py
    sample_image_path = 'image_part_005.jpg'
    sample_mask_path = 'image_part_005.png'

    if os.path.exists(sample_image_path) and os.path.exists(sample_mask_path):
        # Load sample image and mask
        sample_img = cv2.imread(sample_image_path)
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        sample_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        target_size = (256, 256)
        sample_img_resized = cv2.resize(sample_img, target_size)
        sample_mask_resized = cv2.resize(sample_mask, target_size)

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display sample image
        ax[0].imshow(sample_img_resized)
        ax[0].set_title("Sample Image")
        ax[0].axis("off")

        # Display sample mask with agriculture colormap
        ax[1].imshow(sample_mask_resized, cmap=create_agriculture_colormap())
        ax[1].set_title("Sample Segmentation")
        ax[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Sample images not found. Upload your own image to use the app.")

# Footer
st.markdown("---")
st.markdown("Agricultural Image Segmentation App | Created with Streamlit")

# Add sidebar navigation back to home
st.sidebar.markdown("---")
if st.sidebar.button("Back to Home"):
    st.switch_page("app.py")
