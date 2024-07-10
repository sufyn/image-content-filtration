import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests

# Load models
model_paths = {
    "Model 1.1": 'model/img_model_1.1.h5',
    "Model 2.0": 'model/combined_model.h5',  # Assuming this is the combined model
    "Model 3.0": 'ml.h5'
}

# Load the selected model
def load_selected_model(selected_model_name):
    model = load_model(model_paths[selected_model_name])
    return model

# Prediction function for standard classification model
def predict_standard(img, model):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)[0], prediction

# Prediction function for combined model
def predict_combined(img, model):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    
    # Assuming the combined model outputs two sets of predictions: class and age
    class_prediction = predictions[0]
    age_prediction = predictions[1]
    
    predicted_class = np.argmax(class_prediction, axis=1)[0]
    predicted_age = age_prediction[0][0]  # Assuming age prediction is a single scalar value
    
    return predicted_class, class_prediction, predicted_age

st.title("Image Classification App")

st.write("Upload an image or provide an image URL to classify.")

# Select model
selected_model_name = st.selectbox("Select Model", list(model_paths.keys()))

# Load the selected model
model = load_selected_model(selected_model_name)

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
        
        if selected_model_name == "Model 2.0":
            label, class_prediction, age_prediction = predict_combined(img, model)
            categories = ['Adult Content', 'Safe', 'Violent']
            predicted_category = categories[label]
            st.write(f"Classification: {predicted_category}")
            st.write(f"Adult Confidence: {float(class_prediction[0][0]) * 100:.2f}%")
            st.write(f"Safe Confidence: {float(class_prediction[0][1]) * 100:.2f}%")
            st.write(f"Violent Confidence: {float(class_prediction[0][2]) * 100:.2f}%")
            st.write(f"Predicted Age: {age_prediction:.2f}")
        else:
            label, prediction = predict_standard(img, model)
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
                
                if selected_model_name == "Model 2.0":
                    label, class_prediction, age_prediction = predict_combined(img, model)
                    categories = ['Adult Content', 'Safe', 'Violent']
                    predicted_category = categories[label]
                    st.write(f"Classification: {predicted_category}")
                    st.write(f"Adult Confidence: {float(class_prediction[0][0]) * 100:.2f}%")
                    st.write(f"Safe Confidence: {float(class_prediction[0][1]) * 100:.2f}%")
                    st.write(f"Violent Confidence: {float(class_prediction[0][2]) * 100:.2f}%")
                    st.write(f"Predicted Age: {age_prediction:.2f}")
                else:
                    label, prediction = predict_standard(img, model)
                    categories = ['Adult Content', 'Safe', 'Violent']
                    predicted_category = categories[label]
                    st.write(f"Classification: {predicted_category}")
                    st.write(f"Adult Confidence: {float(prediction[0][0]) * 100:.2f}%")
                    st.write(f"Safe Confidence: {float(prediction[0][1]) * 100:.2f}%")
                    st.write(f"Violent Confidence: {float(prediction[0][2]) * 100:.2f}%")
            else:
                st.error("Failed to retrieve image from the provided URL")
        except Exception as e:
            st.error(f"Error fetching image: {e}")
