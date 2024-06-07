import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('ml2.h5')


# Define a function to predict the class of an uploaded image
def predict(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
    return np.argmax(prediction, axis=1)[0], prediction

# Streamlit app interface
st.title('Image Content Classification')
st.write('Upload an image to classify it into Violent, Adult Content, or Safe.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, prediction = predict(image)
    categories = ['Adult Content','Safe','Violent']
    predicted_category = categories[label]
    st.write(f'This image is classified as: {categories[label]}')
    st.write(f'Predicted: {predicted_category} (Confidence: {prediction[0][label]:.2f})')
