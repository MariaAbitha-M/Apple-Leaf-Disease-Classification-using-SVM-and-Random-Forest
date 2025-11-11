#%%import streamlit as st
import os
import streamlit as st
import numpy as np
import cv2
import joblib  # for loading the saved models
from PIL import Image


#%% #Load the trained models
svm_model = joblib.load('svm_model.pkl')   # SVM model
rf_model = joblib.load('rf_model.pkl')     # Random Forest model


disease_names = [
   'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_scab', ]


#%% #Feature extractor (simple version using flattening)
def extract_features(image):
    image = image.resize((128, 128))
    image_np = np.array(image)
    flat_features = image_np.flatten()  # Converts to 1D array
    return flat_features


#%% #Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("APPLE LEAF DISEASE CLASSIFICATION SYSTEM - ML Version (SVM/RF)")
    st.image("https://media.istockphoto.com/id/488152916/photo/organic-red-ripe-apples-on-the-orchard-tree-with-leaves.jpg?s=612x612&w=0&k=20&c=ro9CnL3qOX4Q4bMs6zD-1Zg8BpYoCCQsPJOWdJ9Y2SQ=", use_container_width=True)

    st.markdown("""###  Apple Leaf Disease Classification System""")  
    st.markdown("""Welcome to the Apple Leaf Disease Classification System — an intelligent tool 
                designed to help farmers, researchers, and agricultural Learners and
                diagnose common diseases affecting apple leaves.

    """,unsafe_allow_html=True)

#%% #About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
     ## About the Project""")
    st.markdown("""The Apple Leaf Disease Classification System is a Machine learning-based application designed to assist in the early 
    detection of diseases in apple plants by analyzing images of their leaves. This project aims to empower farmers, 
    agricultural researchers, and plant health enthusiasts by offering a simple, accurate, and user-friendly 
    solution for plant disease diagnosis.""")
    st.markdown("""##  Diseases Detected """)
    st.markdown("""This system can classify apple leaf images into the following categories:

1. Apple Scab

2. Black Rot

3. Cedar Apple Rust""")
    st.markdown(""" ##  How It Works""")
    st.markdown("""User uploads an image of an apple leaf.

The image is preprocessed and passed to a trained SVM and Random forest model.

The model analyzes and predicts the most likely disease category.

Results are displayed instantly to the user.
    """)

#%% #Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition (SVM / RF Models)")
    uploaded_file = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Check if file is an image by extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png']:  # Only allow these image formats
            try:
                # Try opening the image file
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_container_width=True)

                # Convert image to RGB before feature extraction
                img_rgb = img.convert("RGB")
                features = extract_features(img_rgb)

                # Display prediction results
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Predict using SVM"):
                        with st.spinner("Predicting..."):
                            try:
                                result_index = int(svm_model.predict([features])[0])
                                if 0 <= result_index < len(disease_names):
                                    disease_name = disease_names[result_index]
                                    st.success(f"✅ SVM Prediction: {disease_name}")

                                else:
                                    st.error("Prediction is outside known disease categories. Please check the input image.")
                            except Exception as e:
                                st.error(f"An error occurred during SVM prediction: {e}")

                with col2:
                    if st.button("Predict using Random Forest"):
                        with st.spinner("Predicting..."):
                            try:
                                result_index = int(rf_model.predict([features])[0])
                                if 0 <= result_index < len(disease_names):
                                    disease_name = disease_names[result_index]
                                    st.success(f"✅ Random Forest Prediction: {disease_name}")

                                else:
                                    st.error("Prediction is outside known disease categories. Please check the input image.")
                            except Exception as e:
                                st.error(f"An error occurred during Random Forest prediction: {e}")
            except Exception as e:
                st.error(f"Error loading the image: {e}")
        else:
            st.error("Please upload a valid image file (jpg, jpeg, or png).")
