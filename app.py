import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page title and icon
st.set_page_config(page_title="Cataract Detection", page_icon="👁️")

# Title and Description
st.title("👁️ Cataract Detection App")
st.write("Upload an eye image to detect if it's **Normal** or has **Cataract**.")

# Path to the model
MODEL_PATH = 'Trained_models/cataract.h5'

@st.cache_resource
def load_prediction_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please make sure the path is correct.")
        return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model()

def preprocess_image(img):
    """Preprocesses the image for the model."""
    # Resize to (64, 64) as required by the model
    img = img.resize((64, 64))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dims to match batch shape (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale pixel values (1./255) as done in training
    img_array /= 255.0
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        if model is not None:
            with st.spinner('Analyzing...'):
                # Preprocess
                processed_image = preprocess_image(image_display)
                
                # Predict
                result = model.predict(processed_image)
                
                # Logic from TrainCNN_Model.py
                # if result[0][0] == 1: prediction = 'normal' else: prediction = 'cataract'
                # Note: 'cataract' is 0, 'normal' is 1 based on the training script logic
                
                prediction_score = result[0][0]
                
                if prediction_score > 0.5:
                    label = "Normal"
                    confidence = prediction_score
                    color = "green"
                else:
                    label = "Cataract"
                    confidence = 1 - prediction_score
                    color = "red"
                
                st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                st.write(f"Confidence: {confidence:.2%}")
        else:
            st.error("Model could not be loaded, cannot predict.")
