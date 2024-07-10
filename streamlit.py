import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests
import cv2

# Path to models
faceProto = 'model/opencv_face_detector.pbtxt'
faceModel = 'model/opencv_face_detector_uint8.pb'
ageProto = 'model/age_deploy.prototxt'
ageModel = 'model/age_net.caffemodel'

# Load OpenCV models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Age categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load classification models
model_paths = {
    "Model 1.1 (small)": 'model/img_model_1.1.h5',
    "Model 2.0 (combined)": 'model/combined_model.h5',
    "Model 3.0 (large)": 'model/large_ml.h5'
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

# Function to detect faces and predict age using OpenCV
def detect_and_predict_age(net, age_net, frame, confidence_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    age_predictions = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1), max(0, x1-20):min(x2+20, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = ageList[age_preds[0].argmax()]
            age_predictions.append(age)
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
            cv2.putText(frame_opencv_dnn, f'{age}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return frame_opencv_dnn, face_boxes, age_predictions

# Streamlit app
st.title("Image Classification and Age Prediction App")

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
        
        if selected_model_name == "Model 2.0 (combined)":
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

    elif image_url:
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                response.raw.decode_content = True
                img = Image.open(BytesIO(response.content))
                
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                st.image(img, caption="Fetched Image", use_column_width=True)
                
                if selected_model_name == "Model 2.0 (combined)":
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
