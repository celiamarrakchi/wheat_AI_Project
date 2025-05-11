import streamlit as st 
import os
import shutil
from datetime import datetime
from PIL import Image
import subprocess
import json
import yaml

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

# 1. Paths
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
AGENT_DIR     = os.path.normpath(os.path.join(BASE_DIR, "..", "src1", "new_project"))
CONFIG_DIR    = os.path.join(AGENT_DIR, "config")
IMAGES_DIR    = os.path.join(CONFIG_DIR, "images")
TASKS_YAML    = os.path.join(CONFIG_DIR, "tasks.yaml")
AGENT_TASKS   = os.path.join(AGENT_DIR, "tasks.yaml")
RESULTS_JSON  = os.path.join(AGENT_DIR, "results.json")

# 2. Prepare image directory
os.makedirs(IMAGES_DIR, exist_ok=True)

# 3. User Interface
st.title("🌾 Wheat Growth Stage Detection")
st.markdown("Upload a wheat image to detect its growth stage and receive tailored advice.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Run Growth Stage Prediction"):
        try:
            # 4. Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{timestamp}_{uploaded_file.name}"
            dest_path = os.path.join(IMAGES_DIR, filename)
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 5. Update config/tasks.yaml
            with open(TASKS_YAML, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            for task in cfg.get("tasks", []):
                if task.get("name") == "PredictWheatStage":
                    task["input"]["image_path"] = dest_path
            with open(TASKS_YAML, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            # 6. Copy to new_project/tasks.yaml for the agent
            shutil.copy(TASKS_YAML, AGENT_TASKS)

            # 7. Run main.py
            proc = subprocess.run(
                ["python", "main.py"],
                cwd=AGENT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            # 8. Read and display results
            if os.path.exists(RESULTS_JSON):
                with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                    results = json.load(f)

                    if isinstance(results, list) and results:
                        last_result = results[-1]
                        growth_stage = str(last_result.get("growth_stage", "Unknown"))
                        advice = last_result.get("advice", "No advice available.")
                    elif isinstance(results, dict):
                        growth_stage = str(results.get("PredictWheatStage", {}).get("growth_stage", "Unknown"))
                        advice = results.get("GiveWheatAdvice", {}).get("advice", "No advice available.")
                    else:
                        st.error("❌ Unrecognized format in results.json.")
                        st.stop()

                    # Display final result
                    st.success(f"🌱 **Predicted Growth Stage:** {growth_stage}")
                    st.markdown("### 📋 Practical advice for the farmer:")
                    st.markdown(advice)
            else:
                st.error("❌ results.json file not found after execution.")

        except Exception as e:
            st.error(f"❌ Error during execution: {e}")
