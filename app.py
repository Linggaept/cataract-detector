import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set page title and icon
st.set_page_config(page_title="Cataract Detection", page_icon="👁️")

# Title and Description
st.title("👁️ Cataract Detection App")

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
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define the Video Processor Class
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Copto for processing (Resize & Normalize)
        # Convert BGR (OpenCV) to RGB (PIL/Model expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 64x64
        img_resized = cv2.resize(img_rgb, (64, 64))
        
        # Normalize
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Predict
        if model is not None:
            result = model.predict(img_array)
            prediction_score = result[0][0]
            
            if prediction_score > 0.5:
                label = "Normal"
                prob = prediction_score
                color = (0, 255, 0) # Green in BGR
            else:
                label = "Cataract"
                prob = 1 - prediction_score
                color = (0, 0, 255) # Red in BGR
            
            # 3. Display Result on Frame
            text = f"{label}: {prob:.2%}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            # Draw rectangle around the whole image or just text? 
            # Just text is enough for now as we don't have object detection (bounding box), just classification.
        
        return img

# Sidebar for Navigation
option = st.sidebar.selectbox("Choose Input Method", ("Upload Image", "Real-Time Camera"))

if option == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            if model is not None:
                with st.spinner('Analyzing...'):
                    processed_image = preprocess_image(image_display)
                    result = model.predict(processed_image)
                    
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
                st.error("Model could not be loaded.")

elif option == "Real-Time Camera":
    st.subheader("Real-Time Camera Detection")
    st.write("Click **Start** to open webcam. The model will predict in real-time.")
    
    if model is not None:
        webrtc_streamer(key="cataract-detection", 
                        video_transformer_factory=VideoProcessor,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                        )
    else:
        st.error("Model not loaded, cannot start camera.")
