import streamlit as st
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model and class indices
model = load_model('ResNet50_eurosat.h5')
class_indices = np.load('class_indices.npy', allow_pickle=True).item()

# Define a function to preprocess the input image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Define a function to make predictions on the input image
def predict_class(image_file):
    preprocessed_img = preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    class_index = np.argmax(predictions)
    predicted_class = list(class_indices.keys())[list(class_indices.values()).index(class_index)]
    confidence = predictions[0][class_index]
    return predicted_class, confidence

# Streamlit app
st.title('Image Classification')
st.write('Upload an image to classify its type of class')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image
    with open(os.path.join("temp.jpg"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    # Make prediction on the uploaded image
    predicted_class, confidence = predict_class("temp.jpg")
    st.write(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
