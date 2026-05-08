import os
import urllib.request
import streamlit as st
import numpy as np
import time
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# The order must be exactly the same as your training folder structure
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]
# ===============================
# CONFIG & STYLING
# ===============================
st.set_page_config(page_title="Plant Disease Classifier", layout="wide", page_icon="🌿")

# ===============================
# LOAD MODELS (CACHED)
# ===============================
@st.cache_resource
def load_all_models():
    # 1. Configuration
    ann_id = "1supC2FhVCobl5weItc5g1ggNYEy-3ods"
    ann_url = f"https://drive.google.com/uc?export=download&id={ann_id}"
    
    model_files = {
        "cnn": "simple_cnn_best.keras",
        "mobilenet": "mobilenet_best_model.h5",
        "ann": "ann_best_model.keras"
    }

    # 2. Check and Download ANN (the 150MB model)
    if not os.path.exists(model_files["ann"]):
        # Using a status container to avoid warnings
        with st.status("📥 Initializing Deep Learning Models...", expanded=True) as status:
            st.write("Downloading ANN Weights (150MB)...")
            try:
                urllib.request.urlretrieve(ann_url, model_files["ann"])
                
                # Check for "Virus Scan" or "Permission" pages from Google Drive
                if os.path.getsize(model_files["ann"]) < 100000: 
                    st.error("Download failed: The file is too small. Check Drive permissions!")
                    if os.path.exists(model_files["ann"]): os.remove(model_files["ann"])
                    return None, None, None
                
                st.write("Finalizing file structure...")
                time.sleep(2) # Shortened to 2 seconds to reduce the warning impact
                status.update(label="✅ ANN Ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Download error: {e}")
                return None, None, None

    # 2. Loading Phase
    try:
        # We load them individually so we can see which one fails in the logs
        m1 = tf.keras.models.load_model(model_files["cnn"])
        m2 = tf.keras.models.load_model(model_files["mobilenet"])
        m3 = tf.keras.models.load_model(model_files["ann"])
        return m1, m2, m3
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        st.info("Tip: If you see 'File not found', try clicking 'Rerun' in the top right menu.")
        return None, None, None
    
# Execute loading
model1, model2, model3 = load_all_models()

# Final Specs Mapping
if all([model1, model2, model3]):
    MODEL_SPECS = {
        "Model 1 (CNN)": {"model": model1, "size": (224, 224)},
        "Model 2 (MobileNet)": {"model": model2, "size": (224, 224)},
        "Model 3 (ANN)": {"model": model3, "size": (64, 64)},
    }
else:
    st.warning("⚠️ Some models failed to load. Please check your file paths or Drive IDs.")


# ===============================
# HELPER FUNCTIONS
# ===============================
def preprocess_image(image, target_size):
    """Resizes and normalizes image."""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image, target_size):
    # 1. Preprocess
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2. Prediction
    pred = model.predict(img_array, verbose=0)
    
    # 3. Get Index and Confidence
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    
    # 4. Map ID to Name
    class_name = CLASS_NAMES[class_id] # This converts 29 -> "Tomato___Early_blight"
    
    return class_name, confidence

def get_image_input():
    """Centralized image input logic."""
    option = st.radio("Select Input Source:", ["Upload File", "Image URL"], horizontal=True)
    if option == "Upload File":
        file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        if file: return Image.open(file)
    else:
        url = st.text_input("Paste Image URL:")
        if url:
            try:
                res = requests.get(url)
                return Image.open(BytesIO(res.content))
            except:
                st.error("Could not load image from URL.")
    return None

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox("Go to:", [
    "Home", "Try Model", "Compare Models", "Dataset Analysis", "About Models"
])

# ===============================
# PAGE ROUTING
# ===============================

if page == "Home":
    # 1. Main Header with a Hero Image or Logo
    st.title("🌱 Plant Disease Classifier")
    st.markdown("---")
    # 2. Layout using columns for a professional feel
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Automated Disease Diagnosis for for plants")
        st.write("""
            Welcome to **plant disease classification**, a state-of-the-art diagnostic tool designed to help farmers, 
            gardeners, and researchers identify plant diseases instantly. 
            
            By leveraging three distinct Deep Learning architectures—**Custom CNN**, **MobileNetV2**, 
            and **ANN**—this system provides a robust consensus on plant health, ensuring high 
            accuracy and reliability.
        """)
        
        st.info("📍 **Ready to start?** Navigate to the **'Try Model'** tab in the sidebar.")

    with col2:
        # A nice visual placeholder or project logo
       st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzJjbzE2eHM2OGNwbDBxd2hoZTZ2b3I1bWgyZDVibzE5MnBjcnN4ZCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oriNN5kkARo7ZAhuE/giphy.gif", 
                 caption="AI Analysis Engine Active", 
                 use_container_width=True)

    st.markdown("---")

    # 3. Features Section (Using 3 columns)
    st.subheader("Key System Capabilities")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("### 🔍 Precise")
        st.write("Identifies 38 different plant-disease combinations across multiple crop species.")

    with f_col2:
        st.markdown("### 📊 Comparative")
        st.write("Compare predictions from three separate AI models to reduce false positives.")

    with f_col3:
        st.markdown("### 📁 Batch Ready")
        st.write("Upload an entire dataset of images to get a bulk health report in seconds.")

    # 4. Instructions for the user
    with st.expander("📖 How to use this application"):
        st.write("""
            1. **Select a Mode:** Choose between 'Try Model' or 'Compare Model' .
            2. **Input Image:** Upload a file from your device or paste a URL of a plant leaf.
            3. **Review Results:** See the predicted disease name and the confidence level for each model.
            4. **Consult Knowledge:** Visit 'About Models' to understand the technology behind each prediction.
        """)

    # 5. Quick Warning/Disclaimer
    st.warning("⚠️ **Disclaimer:** This tool is for educational and diagnostic support purposes only. Always consult an agricultural expert for critical treatment decisions.")
    st.markdown("---")
    #
    st.write("""This project was made by:""")
    c1, c2,c3 = st.columns(3)

    with c1:
        st.subheader("Paula Moukhtar")
        
    with c2:
        st.subheader("Bavly Hany")
    with c3:
        st.subheader("Aya Naseer")
    c1, c2,c3 = st.columns(3)

    with c1:
        st.subheader("Toka Alaa")
        
    with c2:
        st.subheader("Toka Nasr")
    with c3:
        st.subheader("Rahma Osama")
#

elif page == "Try Model":
    st.title("🔍 Single Model Prediction")
    
    # 1. Let the user choose the input image first
    img = get_image_input()
    
    if img:
        # Create two columns: one for the image, one for model selection
        col_img, col_setup = st.columns([1, 1])
        
        with col_img:
            st.image(img, caption="Target Image", use_container_width=True)
            
        with col_setup:
            st.subheader("Configuration")
            # 2. Add the dropdown for selecting the model
            # We use MODEL_SPECS.keys() to get ["Model 1 (CNN)", "Model 2 (MobileNet)", "Model 3 (ANN)"]
            selected_model_name = st.selectbox(
                "Select the model you want to test:",
                options=list(MODEL_SPECS.keys())
            )
            
            # 3. Predict button
            if st.button("Run Prediction"):
                # Get the specific model and its required input size
                spec = MODEL_SPECS[selected_model_name]
                
                if spec['model']:
                    with st.spinner(f"Analyzing with {selected_model_name}..."):
                        # Run the prediction
                        name, conf = predict(spec['model'], img, spec['size'])
                        
                        # Display the result in a nice highlighted box
                        st.success(f"**Result:** {name}")
                        st.metric("Confidence Score", f"{conf*100:.2f}%")
                else:
                    st.error("The selected model is not loaded correctly.")
elif page == "Compare Models":
    st.title("📊 Model Comparison View")
    img = get_image_input()
    
    if img:
        col_img, col_info = st.columns([1, 2])
        
        with col_img:
            st.image(img, caption="Input Leaf", use_container_width=True)
        
        with col_info:
            st.info("This mode runs the image through all three architectures simultaneously to find a consensus.")

        if st.button("Generate Comparative Report"):
            results = []
            
            # 1. Run predictions and collect data
            for name, spec in MODEL_SPECS.items():
                if spec['model']:
                    # We modify predict to return the raw probabilities too
                    class_name, conf = predict(spec['model'], img, spec['size'])
                    results.append({
                        "Model": name,
                        "Prediction": class_name,
                        "Confidence": conf * 100
                    })

            # 2. Display as a clean Dataframe/Table
            import pandas as pd
            df = pd.DataFrame(results)
            st.table(df)

            # 3. Visual Confidence Comparison (Bar Chart)
            st.subheader("Confidence Comparison")
            # Creating a horizontal bar chart
            st.bar_chart(df.set_index('Model')['Confidence'])

            # 4. Smart Consensus Logic
            st.subheader("System Verdict")
            unique_predictions = df['Prediction'].unique()
            
            if len(unique_predictions) == 1:
                st.success(f"✅ **High Certainty:** All models agree that this is **{unique_predictions[0]}**.")
            elif len(unique_predictions) < len(results):
                # Find the most common prediction
                majority_vote = df['Prediction'].mode()[0]
                st.warning(f"⚠️ **Partial Agreement:** Majority of models suggest **{majority_vote}**, but there is variance. Please check image quality.")
            else:
                st.error("❌ **High Disagreement:** Every model returned a different result. This image might be too blurry or contain an unknown disease.")



elif page == "Dataset Analysis":
    st.title("📁 Batch Processing & Health Audit")
    st.markdown("""
        Upload a collection of plant leaf images to generate a comprehensive health report. 
        This is ideal for analyzing multiple plants from the same field at once.
    """)

    # 1. File Uploader
    files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if files:
        st.success(f"📂 **{len(files)}** images detected.")
        
        # 2. Model Selection for Batch
        st.subheader("Batch Configuration")
        selected_model_name = st.selectbox(
            "Select the model to use for this analysis:",
            options=list(MODEL_SPECS.keys()),
            help="MobileNet is recommended for the highest accuracy in batch reports."
        )

        # 3. Action Button
        if st.button(f"Start Analysis with {selected_model_name}"):
            results = []
            # Create a placeholder for the progress bar and status text
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Get specs for the chosen model
            spec = MODEL_SPECS[selected_model_name]
            
            # 4. Processing Loop
            for i, f in enumerate(files):
                status_text.text(f"Processing image {i+1} of {len(files)}...")
                
                # Open and predict
                img = Image.open(f)
                class_name, conf = predict(spec['model'], img, spec['size'])
                
                results.append({
                    "Filename": f.name,
                    "Prediction": class_name,
                    "Confidence": conf
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(files))
            
            status_text.text("✅ Analysis Complete!")

            # 5. Data Presentation
            import pandas as pd
            df_results = pd.DataFrame(results)

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📈 Summary Report", "📋 Detailed Data"])

            with tab1:
                col1, col2 = st.columns(2)
                
                # Calculate counts
                summary = df_results["Prediction"].value_counts().reset_index()
                summary.columns = ["Disease", "Count"]

                with col1:
                    st.write("### Disease Distribution")
                    st.dataframe(summary, use_container_width=True)
                
                with col2:
                    st.write("### Visual Statistics")
                    st.bar_chart(summary.set_index("Disease"))

            with tab2:
                st.write("### Individual Image Results")
                # Format confidence as percentage for the table
                df_display = df_results.copy()
                df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_display, use_container_width=True)
                
elif page == "About Models":
    st.title("📚 Model Architectures & Deep Performance Analysis")
    st.markdown("Detailed breakdown of layers, statistical metrics, and visual evaluations.")

    # --- MODEL 1: CUSTOM CNN ---
    st.header("1. Custom Simple CNN")
    
    # Top Row: Architecture vs Metrics
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Model Architecture")
        st.code("""
        - Conv2D (64 filters, 3x3) + ReLU
        - MaxPooling2D (2x2)
        - Conv2D (128 filters, 3x3) + ReLU
        - GlobalAveragePooling2D
        - Dense (128 units) + Dropout (0.5)
        - Output: Dense (38 units, Softmax)
        """)
    with col2:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "94.20%")
        m_col1, m_col2 = st.columns(2)
        m_col1.write("**Precision:** 0.94")
        m_col1.write("**Recall:** 0.93")
        m_col2.write("**F1-Score:** 0.94")
        m_col2.write("**Test Loss:** 0.18")

    # Bottom Row: Visuals (Confusion Matrix and Curves)
    st.write("#### Performance Visualizations")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # Use your Confusion Matrix Image ID
        st.image("test1.png", caption="CNN Confusion Matrix", use_container_width=True)
    with v_col2:
        # Use your Training Curves Image ID
        st.image("curve1.png", caption="CNN Training Curves", use_container_width=True)

    st.divider()

    # --- MODEL 2: MOBILENETV2 ---
    st.header("2. MobileNetV2 (Transfer Learning)")
    
    col3, col4 = st.columns([1, 1])
    with col3:
        st.subheader("Model Architecture")
        st.code("""
        - Base: MobileNetV2 (ImageNet)
        - Layer: GlobalAveragePooling2D
        - Layer: Dropout (0.2)
        - Output: Dense (38 units, Softmax)
        """)
    with col4:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "97.85%", delta="Best Performer")
        m_col3, m_col4 = st.columns(2)
        m_col3.write("**Precision:** 0.98")
        m_col3.write("**Recall:** 0.97")
        m_col4.write("**F1-Score:** 0.98")
        m_col4.write("**Test Loss:** 0.09")

    st.write("#### Performance Visualizations")
    v_col3, v_col4 = st.columns(2)
    with v_col3:
        st.image("test2.png", caption="MobileNetV2 Confusion Matrix", use_container_width=True)
    with v_col4:
        st.image("curve2.png", caption="MobileNetV2 Training Curves", use_container_width=True)

    st.divider()

    # --- MODEL 3: ANN ---
    st.header("3. Artificial Neural Network (ANN)")
    
    col5, col6 = st.columns([1, 1])
    with col5:
        st.subheader("Model Architecture")
        st.code("""
        - Input: Flatten (64x64x3)
        - Dense (1024) + ReLU
        - Dropout (0.4)
        - Dense (512) + ReLU
        - Output: Dense (38 units, Softmax)
        """)
    with col6:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "82.40%")
        m_col5, m_col6 = st.columns(2)
        m_col5.write("**Precision:** 0.81")
        m_col5.write("**Recall:** 0.80")
        m_col6.write("**F1-Score:** 0.80")
        m_col6.write("**Test Loss:** 0.65")

    st.write("#### Performance Visualizations")
    v_col5, v_col6 = st.columns(2)
    with v_col5:
        st.image("test3.png", caption="ANN Confusion Matrix", use_container_width=True)
    with v_col6:
        st.image("curve3.png", caption="ANN Training Curves", use_container_width=True)

