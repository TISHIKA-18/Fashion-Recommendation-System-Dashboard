import os
import glob
import numpy as np
import streamlit as st
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

# Function to extract images from zip file
def extract_images(zip_file_path, extraction_directory):
    if not os.path.exists(extraction_directory):
        os.makedirs(extraction_directory)

    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_directory)

# Function to preprocess images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function to extract features using VGG16 model
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract images from the dataset
zip_file_path = 'data/women-fashion.zip'  # Update this path as necessary
extraction_directory = 'data/women_fashion/'
extract_images(zip_file_path, extraction_directory)

# Load image paths
image_directory = os.path.join(extraction_directory, 'women fashion')
image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*'))
                    if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

# Extract features for all images in the dataset
all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))

# Function to recommend fashion items based on input image
def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    # Calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    # Filter out the input image index from similar_indices
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]

    return similar_indices

# Streamlit application layout
st.title("Fashion Recommendation System")
st.write("Upload an image of a clothing item to receive recommendations based on similar styles.")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    input_image_path = "temp_uploaded_image.jpg"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(input_image_path, caption="Uploaded Image", use_column_width=True)

    # Recommend similar fashion items based on the uploaded image
    similar_indices = recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model)

    # Display recommended images
    st.write("Recommended Fashion Items:")

    cols = st.columns(3)  # Create columns for displaying images

    for i, idx in enumerate(similar_indices):
        image_path = os.path.join(image_directory, all_image_names[idx])
        cols[i % 3].image(image_path, caption=all_image_names[idx], use_column_width=True)

