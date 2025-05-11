import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Agricultural Analysis Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better alignment and visual consistency
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f5f5dc;
        padding: 1.5rem;
    }
    
    /* Buttons styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        width: 100%;
        padding: 0.5rem;
        margin-bottom: 1rem;
        border: none;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #3e8e41;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling - fixed dark color */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1e3a32 !important;
    }
    
    /* Sidebar text color */
    .css-1d391kg .st-emotion-cache-16idsys p, 
    [data-testid="stSidebar"] .st-emotion-cache-16idsys p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] .st-emotion-cache-16idsys {
        color: #ffffff !important;
    }
    
    /* Feature card container */
    .feature-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Feature title styling */
    .feature-title {
        color: #2e7d32;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Feature description styling */
    .feature-description {
        color: #333333;
        margin-bottom: 15px;
        flex-grow: 1;
    }
    
    /* Feature button container */
    .feature-button {
        margin-top: auto;
    }
    
    /* Coming soon feature styling */
    .coming-soon-container {
        background: linear-gradient(135deg, #43a047 0%, #1b5e20 100%);
        color: white;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    
    .coming-soon-container:hover {
        transform: translateY(-5px);
    }
    
    .coming-soon-container::after {
        content: "COMING SOON";
        position: absolute;
        top: 10px;
        right: -35px;
        transform: rotate(45deg);
        background-color: #ff9800;
        color: white;
        padding: 5px 40px;
        font-size: 12px;
        font-weight: bold;
        z-index: 1;
    }
    
    .coming-soon-title {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .coming-soon-description {
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    
    /* Equal height rows */
    .equal-height {
        display: flex;
        flex-wrap: wrap;
    }
    
    .equal-height > div {
        display: flex;
        flex-direction: column;
    }
    
    /* Footer styling */
    .footer {
        font-size: 12px;
        color: #555555;
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #dddddd;
    }
    
    /* Custom container for intro */
    .intro-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 30px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .feature-title {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🌾 Agricultural Analysis Platform")

# Introduction section with improved styling
st.markdown("""
<div class="intro-container">
    <p>Welcome to our comprehensive Agricultural Analysis Platform! This platform provides various tools and features
    to help analyze agricultural data, satellite imagery, and much more.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with renamed title
st.sidebar.title("Smartify")
st.sidebar.info(
    """
    This platform is developed by a collaborative team of agricultural data scientists and engineers.
    For inquiries, feel free to contact us!
    """
)

# Coming Soon Features - Now at the top and more exciting
st.header("Exciting Features Coming Soon!")
coming_soon_col1, coming_soon_col2 = st.columns(2)

with coming_soon_col1:
    st.markdown("""
    <div class="coming-soon-container">
        <div class="coming-soon-title">🌦️ Advanced Weather Analysis</div>
        <div class="coming-soon-description">
            Our cutting-edge weather analysis system will predict patterns and provide actionable insights for optimal crop management.
            Integrate with local weather stations and satellite data for hyper-local forecasting.
        </div>
    </div>
    """, unsafe_allow_html=True)

with coming_soon_col2:
    st.markdown("""
    <div class="coming-soon-container">
        <div class="coming-soon-title">🌱 AI-Powered Soil Health Monitoring</div>
        <div class="coming-soon-description">
            Revolutionary soil analysis that combines sensor data with AI to provide real-time monitoring of soil nutrients, pH levels, and moisture content.
            Receive custom fertilizer recommendations tailored to your specific soil conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Available Features
st.header("Available Features")

# Using row and column layout for better organization
col1, col2 = st.columns(2)

with col1:
    # Feature 1: Satellite Image Segmentation
    st.markdown("""
    <div class="feature-container">
        <div class="feature-title">🛰️ Satellite Image Segmentation</div>
        <div class="feature-description">
            Segment agricultural satellite imagery to identify different crop types, soil conditions, and more.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Satellite Image Segmentation", key="satellite_button"):
        st.switch_page("pages/1_Satellite_Image_Segmentation.py")

    # Feature 2: Crop Yield Prediction
    st.markdown("""
    <div class="feature-container">
        <div class="feature-title">📊 Crop Yield Prediction</div>
        <div class="feature-description">
            Detect the crop yield condition of wheat and provide tailored agricultural advice.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Crop Yield Prediction", key="harvest_wheat_disease_button"):
        st.switch_page("pages/4_Harvest_Wheat_Disease_Detection.py")

    # Feature 3: Wheat Disease Detection
    st.markdown("""
    <div class="feature-container">
        <div class="feature-title">🌾 Wheat Disease Detection</div>
        <div class="feature-description">
            Detect and classify wheat leaf conditions into five categories using AI.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Wheat Disease Detection", key="wheat_disease_button"):
        st.switch_page("pages/2_Wheat_Disease_Detection.py")

with col2:
    # Feature 4: Wheat Growth Stage Detection
    st.markdown("""
    <div class="feature-container">
        <div class="feature-title">🌾 Wheat Growth Stage Detection</div>
        <div class="feature-description">
            Detect the growth stage of wheat and get tailored advice.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Wheat Growth Stage Detection", key="wheat_growth_stage_button"):
        st.switch_page("pages/3_Wheat_Growth_Stage_Detection.py")

    # Feature 5: Weed Detection
    st.markdown(""" 
    <div class="feature-container">
        <div class="feature-title">🌿 Weed Detection</div>
        <div class="feature-description">
            Detect and classify weeds in agricultural fields using AI.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Weed Detection", key="weed_detection_button"):
        st.switch_page("pages/5_Weed_Detection.py")

    # Feature 6: Insect Detection
    st.markdown(""" 
    <div class="feature-container">
        <div class="feature-title">🐞 Insect Detection</div>
        <div class="feature-description">
            Upload a field image to detect insect type and get recommendations.
        </div>
        <div class="feature-button"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Insect Detection", key="insect_detection_button"):
        st.switch_page("pages/6_Insect_Detection.py")

# Footer
st.markdown("""
<div class="footer">
    © 2025 Agricultural Analysis Platform presented by DATA BRAINS. All rights reserved.
</div>
""", unsafe_allow_html=True)