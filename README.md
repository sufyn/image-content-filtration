### Image Content Classification Model with Streamlit Web Application

#### Objective

The objective of this project was to develop a machine learning model capable of classifying images into three categories: 'Violent', 'Adult Content', and 'Safe'. The model was to be integrated into a Streamlit web application to demonstrate its functionality in a live web interface.

#### Tools and Libraries Used

- **Python**: Main programming language used for the project.
- **TensorFlow/Keras**: Used for building and training the convolutional neural network (CNN).
- **NumPy**: For data manipulation and preprocessing.
- **Matplotlib**: For creating visualizations of the training process.
- **Streamlit**: For creating an interactive web app to showcase the model's capabilities.
- **Pillow**: For handling image processing.

#### Key Steps

1. **Environment Setup**
   - Installed necessary libraries: TensorFlow, NumPy, Matplotlib, Streamlit, and Pillow.

2. **Data Collection and Preprocessing**
   - Collected images from various sources, categorized them into 'Violent', 'Adult Content', and 'Safe'.
   - Used TensorFlow's `ImageDataGenerator` to resize, normalize, and augment images.
   - Split the data into training and validation sets with an 80-20 ratio.

3. **Model Development and Training**
   - Constructed a CNN using MobileNetV2 as the base model.
   - Added custom layers on top of MobileNetV2 for specific classification tasks.
   - Compiled the model with the Adam optimizer and categorical cross-entropy loss function.
   - Trained the model using early stopping and a learning rate scheduler to prevent overfitting.
   - Applied class weights to handle class imbalance.

4. **Model Evaluation**
   - Evaluated the model using validation data.
   - Visualized training and validation accuracy and loss over epochs to assess model performance.

5. **Streamlit Integration**
   - Developed a Streamlit web app allowing users to upload images and view classification results.
   - Displayed model accuracy, loss, and training history within the web app.

#### Model Performance

- **Training Results**:
  - Achieved a training accuracy of approximately 80% and a validation accuracy of approximately 50% after 11 epochs.
  - Validation loss and accuracy fluctuated, indicating possible overfitting, but were managed using data augmentation, dropout, and class weights.

- **Evaluation Metrics**:
  - Validation Loss: 1.6966
  - Validation Accuracy: 0.7132

#### Streamlit Web Application

The Streamlit web application provides an interactive interface for users to:

- Upload an image for classification.
- View the uploaded image and its predicted category.
- Display confidence scores for each category.
- Show model performance metrics such as validation loss and accuracy.
- Visualize training history for accuracy and loss if available.

#### Conclusion

The project successfully developed a machine learning model to classify images into 'Violent', 'Adult Content', and 'Safe' categories with a reasonable degree of accuracy. Integrating the model into a Streamlit web application provided an interactive platform to demonstrate the model's capabilities, allowing users to upload images and view real-time classification results.

Future work could focus on collecting more data, especially for underrepresented classes, and fine-tuning the model further to improve accuracy and reduce overfitting. Additionally, deploying the Streamlit app on a cloud platform would make it accessible to a broader audience.

#### Running the Streamlit App

To run the Streamlit app, use the following command:

```sh
streamlit run app.py
```

This will start the Streamlit server, and you can interact with the web app through your browser, uploading images and viewing classification results.
