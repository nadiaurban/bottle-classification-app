import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# ---------------------------
# Placeholder variables and model loading
# ---------------------------

st.set_page_config(
    page_title="Bottle Classifier",  # This shows up in the browser tab
    page_icon="♻️",                   # Optional: Emoji or URL to an icon
    layout="centered",                # Optional: 'centered' or 'wide'
    initial_sidebar_state="auto"     # Optional: 'auto', 'expanded', 'collapsed'
)

# Update the class labels if needed (should match your model's output encoding)
CLASS_NAMES = ["glass bottle", "plastic bottle"] # <-- UPDATE: Replace with actual class names from "labels.txt"

# Load the model using TFSMLayer
model = keras.layers.TFSMLayer(
    "model.savedmodel",        # <-- UPDATE: Replace with the actual path to your model
    call_endpoint="serving_default"
)

# ---------------------------
# Utility Function to Resize Images
# ---------------------------
def resize_image(image_path, width):
    """Loads an image from a file path and resizes it to the given width while maintaining aspect ratio."""
    try:
        img = Image.open(image_path)
        img.thumbnail((width, width))
        return img
    except Exception as e:
        st.error("Error loading image: " + str(e))
        return None

# ---------------------------
# Sidebar: Model Information, Example Images, and Author Details
# ---------------------------
with st.sidebar:
    # Title for the sidebar
    st.title("ℹ️ Glass and plasic bottle scanner")
    
    # Placeholder model description; update the text below with your model's detai5Y0ls.
    st.write(
    """
    Bottle Classification App  
    This app helps to classify glass and plastic bottles:

    1. Plastic bottle.
    2. Glass bottle.


    **Model Design**  
    - **Goal:** 🎯 We wanted to develop an AI model that can recognize plastic bottles from glass bottles. 
    - **Data Type:** 🖼️ Images of bottles in 2 categories (1) plastic (2) glass.
    - **Data Source:** 🌐 The images were collected online from bing.com.      
    - **Training:** 🏋️ Model trained using Teachable Machine.  
    - **Model Type:** 🧠 CNN (Convolutional Neural Network).
    """
    )

    # Example Images section – replace placeholder images and captions with actual files/paths.
    st.write("Class 1: Glass bottle")
    img1 = resize_image("example1.jpg", 300)  
    if img1:
        st.image(img1, caption="Training data for class 1- glass bottle: 196 pictures")

    st.write("Class 2: plastic bottle")
    img2 = resize_image("example2.jpg", 300)  
    if img2:
        st.image(img2, caption="Training data for class 2 - plastic bottle: 2217 pictures")


    # Model Authors Section
    st.write("### Model Authors")
    st.write(
        """
        - **Name:** <隋屹宸 Eric>   
        - **Name:** <陈弈楷 Steven>
        """
    )
    
    st.caption("📝 Use the file uploader or camera input on the main panel to analyze an image.")

# ---------------------------
# Optional: Custom CSS for Sidebar and Main Area Styling
# ---------------------------
st.markdown(
    """
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #FFFDD0;  /* Light yellow */
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span {
            color: black !important;  /* Ensure text readability */
        }
        /* Header styling */
        .header-container {
            text-align: center;
            padding: 40px 0;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #2E5A4D;
            margin-bottom: 10px;
        }
        .main-text {
            font-size: 18px;
            color: #3A5A40;
        }
        /* Input section styling */
        .input-container {
            text-align: center;
            padding: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Function to Preprocess Image for Recognition
# ---------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # or use ImageOps.fit if you prefer center crop
    image = image.convert("RGB")
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1  # Match TM normalization
    normalized_image_array = np.expand_dims(normalized_image_array, axis=0)
    return normalized_image_array

# ---------------------------
# Main Area: Header and Image Recognition Toolbar
# ---------------------------
# Header Section with title and instructions (center-aligned)
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🔍 Bottle Classifier </h1>', unsafe_allow_html=True)  # <-- Change to the name of your tool
st.markdown('<p class="main-text">Upload or capture an image below to see the model predictions.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize camera visibility state if it doesn't exist.
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# Input section: Use a container to vertically stack the file uploader and then the "Take Picture" button.
st.markdown('<div class="input-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image for analysis", type=["jpg", "png", "jpeg"])

# Place the "Take Picture" button below the uploader.
if st.button("Take Picture"):
    st.session_state.show_camera = True

# If the button has been clicked, show the camera input below the button.
if st.session_state.show_camera:
    camera_image = st.camera_input("📸 Capture an image")
else:
    camera_image = None
st.markdown('</div>', unsafe_allow_html=True)

# Process the image from either method.
if uploaded_file is not None or camera_image is not None:
    # Use the camera image if available; otherwise use the uploaded file.
    if camera_image is not None:
        image = Image.open(camera_image)
    else:
        image = Image.open(uploaded_file)
    
    # Resize the image for display purposes.
    max_width = 300  # Set display width for preview (adjust as needed)
    image.thumbnail((max_width, max_width))
    st.image(image, caption="📸 Selected Image", use_container_width=False)
    
    # Preprocess the image and get a prediction from the model.
    processed_image = preprocess_image(image)
    raw_output = model(processed_image)

    # Some model output types can be dictionaries; automatically extract predictions if needed.
    if isinstance(raw_output, dict):
        raw_output = raw_output[list(raw_output.keys())[0]]

    # Apply softmax to convert logits to probabilities
    prediction = tf.nn.softmax(raw_output).numpy()

    # Get predicted class and confidence
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the prediction result.
    st.write("### Prediction:")
    st.success(f"**Class: {CLASS_NAMES[predicted_class]}** (Confidence: {confidence:.2%})")


# ---------------------------
# Footer Section: Additional Info or Branding
# ---------------------------
st.markdown(
    """
    <div style='text-align: left; padding-top: 40px;'>
        <p>©Created by Nadia Urban for Shanghai Thomas School.<br>CNN model trained with Teachable Machine</p>
    </div>
     <div style='text-align: left; padding-top: 40px;'>
        <p>Disclamer: This app is made for education purposes ONLY.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# School Logo
st.image("school_logo.png", width=150)  # <-- UPDATE: Replace with your logo file path if needed
