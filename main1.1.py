import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('ml2.h5')


# predict the class 
def predict(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
    return np.argmax(prediction, axis=1)[0], prediction

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

    # Access individual class probabilities from prediction
    class_probabilities = {category: prob for category, prob in zip(categories, prediction[0])}
    Validation_Loss= 0.8010649681091309
    Validation_Accuracy= 0.719298243522644

    st.write("Model Statistics:")
    st.write(f"- Validation Accuracy: {Validation_Accuracy:.2f}")
    st.write(f"- Validation Loss: {Validation_Loss:.2f}")

    with st.expander("Detailed Prediction Breakdown"):
        for i, category in enumerate(categories):
            class_probability = prediction[0][i]
            st.write(f"- {category}: {class_probability:.2f}")

    
    # Display class probabilities as a table
    st.subheader('Class Probabilities')
    st.table(class_probabilities)        

    # Display prediction details
    fig, ax = plt.subplots()
    ax.bar(categories, prediction[0])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence for Each Category')
    st.pyplot(fig)
    
    # Display a pie chart of class probabilities
    fig, ax = plt.subplots()
    ax.pie(prediction[0], labels=categories, autopct='%1.1f%%')
    
    ax.set_title('Class Probabilities')
    st.pyplot(fig)

    # Display a bar chart of class probabilities
    fig, ax = plt.subplots()
    ax.barh(categories, prediction[0])
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence for Each Category')
    st.pyplot(fig)

    # Display model accuracy and loss
    history = np.load('model/ml2_history.npy', allow_pickle=True).item()
    
    st.write("### Model Performance")
    st.write(f"Validation Accuracy: {history['val_accuracy'][-1]:.2f}")
    st.write(f"Validation Loss: {history['val_loss'][-1]:.2f}")

    # Plot accuracy and loss over epochs
    st.write("### Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    st.pyplot(fig)


