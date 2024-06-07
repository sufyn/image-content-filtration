import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('new.h5')

# Define a function to predict the class of an uploaded image
def predict(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]

# Streamlit app interface
st.title('Image Content Classification')
st.write('Upload an image to classify it into Violent, Adult Content, or Safe.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    
    categories = ['Violent', 'Adult Content', 'Safe']
    st.write(f'This image is classified as: {categories[label]}')
