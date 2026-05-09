import os
import numpy as np
import requests
from PIL import Image
import urllib.request
from io import BytesIO
import time
import urllib.request
import tensorflow as tf
import streamlit as st

# ===============================
# 1. DOWNLOAD LOGIC (Outside Cache)
# ===============================
def ensure_ann_is_downloaded():
    ann_url = "https://www.dropbox.com/scl/fi/wktx7nv4lwq2xzfrzbajj/ann_best_model.keras?rlkey=e4kmnpvdwrdx472ba6poprhrt&st=3dploowk&dl=1"
    ann_path = "ann_best_model.keras"
    
    # Clean up if it's the old 3KB error file
    if os.path.exists(ann_path) and os.path.getsize(ann_path) < 1000000:
        os.remove(ann_path)
        
    if not os.path.exists(ann_path):
        with st.status("📥 Downloading ANN model (150MB)...", expanded=True) as status:
            try:
                urllib.request.urlretrieve(ann_url, ann_path)
                status.update(label="✅ Download Complete!", state="complete", expanded=False)
                time.sleep(1) 
                st.rerun() # This clears the status box from the UI
            except Exception as e:
                st.error(f"Download failed: {e}")
                return False
    return True

# ===============================
# 2. LOAD MODELS (Cached in Memory)
# ===============================
@st.cache_resource
def load_all_models():
    # File paths for models already on GitHub/Disk
    model_files = {
        "cnn": "simple_cnn_best.keras",
        "mobilenet": "mobilenet_best_model.h5",
        "ann": "ann_best_model.keras"
    }

    try:
        m1 = tf.keras.models.load_model(model_files["cnn"])
        m2 = tf.keras.models.load_model(model_files["mobilenet"])
        m3 = tf.keras.models.load_model(model_files["ann"])
        return m1, m2, m3
    except Exception as e:
        return None, None, None

# ===============================
# 3. EXECUTION FLOW
# ===============================

# First: Ensure files exist
if ensure_ann_is_downloaded():
    # Second: Load into memory
    model1, model2, model3 = load_all_models()

    if all([model1, model2, model3]):
        # Define the Model Specs Mapping
        MODEL_SPECS = {
            "Model 1 (CNN)": {"model": model1, "size": (224, 224)},
            "Model 2 (MobileNet)": {"model": model2, "size": (224, 224)},
            "Model 3 (ANN)": {"model": model3, "size": (64, 64)},
        }
        # Final visual confirmation that disappears
        msg = st.empty()
        msg.success("🚀 All models active.")
        time.sleep(1)
        msg.empty() 
    else:
        st.error("❌ Models exist on disk but failed to load into TensorFlow.")
else:
    st.warning("⚠️ Waiting for ANN model download...")


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
    cnn_model = Sequential([      
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5), 
    Dense(38, activation='softmax')
])
        """)
    with col2:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "92.84%")
        m_col1, m_col2 = st.columns(2)
        m_col1.write("**Precision:** 0.93")
        m_col1.write("**Recall:** 0.93")
        m_col2.write("**F1-Score:** 0.93")
        m_col2.write("**F1-Score:** 0.93")

        m_col1.write("**Total params:** 5,499,380")
        m_col1.write("**Trainable params:** 1,833,126")
        m_col2.write("**Optimizer params:** 3,666,254 ")
        m_col2.write("**Non-trainable params:** 0")

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
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(38, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        """)
    with col4:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "95.05", delta="Best Performer")
        m_col3, m_col4 = st.columns(2)
        m_col3.write("**Precision:** 0.95")
        m_col3.write("**Recall:** 0.95")
        m_col4.write("**F1-Score:** 0.95")
        m_col4.write("**Test Loss:** 0.1530")
        m_col3.write("**Trainable params: 675,366**")
        m_col4.write("**Non-trainable params: 2,257,984**")
        m_col3.write("**Total params: 2,933,350**")

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
        ann_model = Sequential([
        Flatten(input_shape=(64, 64, 3)), 
        Dense(1024, activation='relu'),  
        Dropout(0.4),
        Dense(512, activation='relu'),    
        Dropout(0.3),
        Dense(256, activation='relu'),    
        Dropout(0.3),
        Dense(38, activation='softmax')   
        ])
        """)
    with col6:
        st.subheader("Final Test Metrics")
        st.metric("Test Accuracy", "18.01% ")
        m_col5, m_col6 = st.columns(2)
        m_col5.write("**Precision:** 0.81")
        m_col5.write("**Recall:** 0.18")
        m_col6.write("**F1-Score:** 0.80")
        m_col6.write("**Test Loss:** 2.9364")
        m_col5.write("**Total params:** 13,249")
        m_col6.write("**Trainable params:** 13,249,830")
        m_col5.write("**Non-trainable params:** 0 ")

    st.write("#### Performance Visualizations")
    v_col5, v_col6 = st.columns(2)
    with v_col5:
        st.image("test3.png", caption="ANN Confusion Matrix", use_container_width=True)
    with v_col6:
        st.image("curve3.png", caption="ANN Training Curves", use_container_width=True)

