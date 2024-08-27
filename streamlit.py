import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests
import cv2
from fastai.vision.learner import load_learner
from fastai.data.external import URLs
from fastai.data.transforms import get_image_files

# Custom functions or classes
def label_func(f): return f[0]

faceProto = 'model/opencv_face_detector.pbtxt'
faceModel = 'model/opencv_face_detector_uint8.pb'
ageProto = 'model/age_deploy.prototxt'
ageModel = 'model/age_net.caffemodel'
# classificationModelPath = 'model/img_model_1.1.h5'  

# Load OpenCV models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Load classification model
# classificationModel = load_model(classificationModelPath)

#Fast AI model
import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# Load FastAI model with custom functions
# fastai_model_path = 'model/img_model2.pkl'
# fastai_model = load_learner(fastai_model_path, cpu=True)

# Load FastAI model with custom functions
fastai_model_path = 'model/updated_6c.pkl'
fastai_model2 = load_learner(fastai_model_path, cpu=True)

# Age categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ageThresholds = {
    '(0-2)': 'Minor',
    '(4-6)': 'Minor',
    '(8-12)': 'Minor',
    '(15-20)': 'Minor',
    '(25-32)': 'Adult',
    '(38-43)': 'Adult',
    '(48-53)': 'Adult',
    '(60-100)': 'Adult'
}

# Prediction function for standard classification model
def predict_classification(img, model):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)[0], prediction

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

# Prediction function for FastAI model
def predict_fastai(img, model):
    img = img.convert('RGB')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    prediction = model.predict(img_bytes)
    label = prediction[0]
    probabilities = prediction[2]
    return label, probabilities

st.sidebar.title("Choose an option:")
option = st.sidebar.selectbox("Select a task", ("Image Classification", "Video Classification"))

if option == "Image Classification":
    st.header("Image Classification")

    # Streamlit app
    st.title("Image Classification and Age Prediction App")
    
    st.write("Upload an image or provide an image URL to classify.")
    
    operation_mode = st.selectbox("Select Operation Mode", ["Age and Classification", "Only Age Prediction"])
    
    # model_selection = st.selectbox("Select Model", ["model 1", "model 2","model 3"])
    
    model_selection = st.selectbox("Select Model", ["model 3"])
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    image_url = st.text_input("Or enter an image URL:")
    
    if st.button("Submit"):
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # if operation_mode == "Age and Classification":
            #     if model_selection == "model 1":
    
            #         label, prediction = predict_classification(img, classificationModel)
            #         categories = ['Adult Content', 'Safe', 'Violent']
            #         predicted_category = categories[label]
            #         st.write(f"Classification: {predicted_category}")
            #         st.write(f"Adult Confidence: {float(prediction[0][0]) * 100:.2f}%")
            #         st.write(f"Safe Confidence: {float(prediction[0][1]) * 100:.2f}%")
            #         st.write(f"Violent Confidence: {float(prediction[0][2]) * 100:.2f}%")
                    
            #         img_cv = np.array(img)
            #         result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
            #         st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
            #         for age in age_predictions:
            #             if age:
            #                 st.write(f"Age: {age} - {ageThresholds[age]}")
            #             else:
            #                 st.write("No age predicted")
            #     elif model_selection == "model 2":
    
            #         label, probabilities = predict_fastai(img, fastai_model)
            #         categories = fastai_model.dls.vocab
            #         st.write(f"Classification: {label}")
            #         st.write("Probabilities:")
            #         for i, prob in enumerate(probabilities):
            #             st.write(f"{categories[i]}: {prob *100:.2f}%")
                    
            #         img_cv = np.array(img)
            #         result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
            #         st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
            #         for age in age_predictions:
            #             if age:
            #                 st.write(f"Age: {age} - {ageThresholds[age]}")
            #             else:
            #                 st.write("No age predicted")
            if model_selection == "model 3":
    
                    label, probabilities = predict_fastai(img, fastai_model2)
                    categories = fastai_model2.dls.vocab
                    st.write(f"Classification: {label}")
                    st.write("Probabilities:")
                    for i, prob in enumerate(probabilities):
                        st.write(f"{categories[i]}: {prob *100:.2f}%")
                    
                    img_cv = np.array(img)
                    result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                    st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                    for age in age_predictions:
                        if age:
                            st.write(f"Age: {age} - {ageThresholds[age]}")
                        else:
                            st.write("No age predicted")
    
            elif operation_mode == "Only Age Prediction":
                img_cv = np.array(img)
                result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                for age in age_predictions:
                    if age:
                        st.write(f"Age: {age} - {ageThresholds[age]}")
                    else:
                        st.write("No age predicted")
    
        elif image_url:
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    response.raw.decode_content = True
                    img = Image.open(BytesIO(response.content))
                    
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    st.image(img, caption="Fetched Image", use_column_width=True)
                    
                    # if operation_mode == "Age and Classification":
                    #     if model_selection == "model 1":
    
                    #         label, prediction = predict_classification(img, classificationModel)
                    #         categories = ['Adult Content', 'Safe', 'Violent']
                    #         predicted_category = categories[label]
                    #         st.write(f"Classification: {predicted_category}")
                    #         st.write(f"Adult Confidence: {float(prediction[0][0]) * 100:.2f}%")
                    #         st.write(f"Safe Confidence: {float(prediction[0][1]) * 100:.2f}%")
                    #         st.write(f"Violent Confidence: {float(prediction[0][2]) * 100:.2f}%")
                            
                    #         img_cv = np.array(img)
                    #         result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                    #         st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                    #         for age in age_predictions:
                    #             if age:
                    #                 st.write(f"Age: {age} - {ageThresholds[age]}")
                    #             else:
                    #                 st.write("No age predicted")
                    #     elif model_selection == "model 2":
    
                    #         label, probabilities = predict_fastai(img, fastai_model)
                    #         categories = fastai_model.dls.vocab
                    #         st.write(f"Classification: {label}")
                    #         st.write("Probabilities:")
                    #         for i, prob in enumerate(probabilities):
                    #             st.write(f"{categories[i]}: {prob *100:.2f}%")
                            
                    #         img_cv = np.array(img)
                    #         result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                    #         st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                    #         for age in age_predictions:
                    #             if age:
                    #                 st.write(f"Age: {age} - {ageThresholds[age]}")
                    #             else:
                    #                 st.write("No age predicted")
                         if model_selection == "model 3":
    
                            label, probabilities = predict_fastai(img, fastai_model2)
                            categories = fastai_model2.dls.vocab
                            st.write(f"Classification: {label}")
                            st.write("Probabilities:")
                            for i, prob in enumerate(probabilities):
                                st.write(f"{categories[i]}: {prob *100:.2f}%")
                            
                            img_cv = np.array(img)
                            result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                            st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                            for age in age_predictions:
                                if age:
                                    st.write(f"Age: {age} - {ageThresholds[age]}")
                                else:
                                    st.write("No age predicted")
    
                    elif operation_mode == "Only Age Prediction":
                        img_cv = np.array(img)
                        result_img, face_boxes, age_predictions = detect_and_predict_age(faceNet, ageNet, img_cv)
                        st.image(result_img, caption="Image with Age Prediction", use_column_width=True)
                        for age in age_predictions:
                            if age:
                                st.write(f"Age: {age} - {ageThresholds[age]}")
                            else:
                                st.write("No age predicted")
            except Exception as e:
                st.error(f"An error occurred: {e}")


elif option == "Video Classification":
    st.header("Video Classification")
    import streamlit as st
    import cv2
    from fastai.vision.all import *
    from PIL import Image
    
    def classify_video(video_path, model_path, frame_interval=15):
        learn = load_learner(model_path)
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_count = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
    
            if frame_count % frame_interval == 0:
                # Convert frame to RGB (FastAI expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Classify the frame
                pred_class, _, _ = learn.predict(PILImage.create(frame_rgb))
                results.append((frame_count, str(pred_class)))
    
            frame_count += 1
    
        cap.release()
        return results
    
    # Streamlit UI
    st.title("Video Classification App")
    st.write("Upload a video and classify its content as WALLPAPER, SAFE, VIOLENT, or SPAM.")
    
    # Upload video
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    model_path = 'model/video_model.pkl'
    
    if uploaded_video is not None and model_path:
        st.video(uploaded_video)
        st.write("Classifying the video...")
        
        # Save uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        # Classify the video
        results = classify_video("temp_video.mp4", model_path)
        
        st.write("Classification Results:")
        for frame_number, predicted_class in results:
            st.write(f"Frame {frame_number}: {predicted_class}")
        
        # Determine if any frame is classified as Violent
        if any(pred_class == "Violent" for _, pred_class in results):
            st.error("Violence detected in the video.")
        else:
            st.success("No violence detected in the video.")
