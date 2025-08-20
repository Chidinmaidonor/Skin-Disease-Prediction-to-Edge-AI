import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model/disease_prediction.h5")
model.compile(optimizer='adam', 
loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels
class_names = ['Acne and Rosacea', 'Actinic Keratosis','Atopic Dermatitis','Bullous Disease',
               'Cellulitis Impetigo','Eczema','Exanthems','Herpes HPV','Light disease','Lupus',
               'Melanoma Skin Cancer','Poison Ivy','Psoriasis','Seborrheic','Systemic Disease',
               'Tinea Ringworm','Urticaria Hives','Vascular Tumors','Vasculitis','Warts Molluscum']

# Streamlit UI
st.title("🩺 Chidinma Skin Disease Prediction App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def predict_image(img):
    # Ensure image is in correct format
    if not isinstance(img, Image.Image):
        img = Image.open(img)

    # Preprocess
    img = img.resize((128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))  # Highest probability
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

# Handle uploaded image
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    result, confidence = predict_image(uploaded_file)
    st.write(f"### ✅ Prediction: {result}")
    st.write(f"### 🔎 Confidence: {confidence:.2f}")
