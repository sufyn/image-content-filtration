import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
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
    # Access individual class probabilities from prediction
    class_probabilities = {category: prob for category, prob in zip(categories, prediction[0])}

    # Display class probabilities as a progress bar chart
    st.subheader('Class Probabilities')
    st.progress(class_probabilities['Adult Content'])
    st.progress(class_probabilities['Safe'])
    st.progress(class_probabilities['Violent'])

    # Display class probabilities as a table
    st.subheader('Class Probabilities')
    st.table(class_probabilities)

    # Display the raw prediction values
    st.subheader('Raw Prediction Values')
    st.write(prediction)

    Validation_Loss= 0.8010649681091309
    Validation_Accuracy= 0.719298243522644
    st.write("Model Statistics:")
    st.write(f"- Validation Accuracy: {Validation_Accuracy:.2f}")

    # Display precision and recall in a table
    st.dataframe(pd.DataFrame({'Category': categories, 'Validation Accuracy': Validation_Accuracy, 'Validation Loss': Validation_Loss}))

    # Display the model summary
    st.write("Model Summary:")
    st.write(model.summary())

    # Display the model architecture
    st.write("Model Architecture:")
    st.write(model.to_json())

    # Display the model weights
    st.write("Model Weights:")
    st.write(model.get_weights())

    # Display the model optimizer
    st.write("Model Optimizer:")
    st.write(model.optimizer)

    # Display the model loss function
    st.write("Model Loss Function:")

    st.write(model.loss)

    # Display the model metrics
    st.write("Model Metrics:")
    st.write(model.metrics)

    # Display the model learning rate
    st.write("Model Learning Rate:")
    st.write(model.optimizer.learning_rate)

    # Display the model loss function
    st.write("Model Loss Function:")
    st.write(model.loss)

      # Add a collapsible section for detailed prediction breakdown (optional)
    with st.expander("Detailed Prediction Breakdown"):
        for i, category in enumerate(categories):
            class_probability = prediction[0][i]
            st.write(f"- {category}: {class_probability:.2f}")
            
