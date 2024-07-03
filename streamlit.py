import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests

# Load the trained model
model = load_model('model/img_model_1.1.h5')

# Predict the class
def predict(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)[0], prediction

st.title("Image Classification App")

st.write("Upload an image or provide an image URL to classify.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

# Input image URL
image_url = st.text_input("Or enter an image URL:")

# Button to trigger image URL submission and classification
if st.button("Submit"):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        st.image(img, caption="Uploaded Image", use_column_width=True)
        label, prediction = predict(img)
        categories = ['Adult Content', 'Safe', 'Violent']
        predicted_category = categories[label]
        st.write(f"Classification: {predicted_category}")
        st.write(f"Confidence: {float(prediction[0][label]):.2f}")

    elif image_url:
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                response.raw.decode_content = True
                img = Image.open(BytesIO(response.content))
                
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                st.image(img, caption="Fetched Image", use_column_width=True)
                label, prediction = predict(img)
                categories = ['Adult Content', 'Safe', 'Violent']
                predicted_category = categories[label]
                st.write(f"Classification: {predicted_category}")
                st.write(f"Confidence: {float(prediction[0][label]):.2f}")
            else:
                st.error("Failed to retrieve image from the provided URL")
        except Exception as e:
            st.error(f"Error fetching image: {e}")
