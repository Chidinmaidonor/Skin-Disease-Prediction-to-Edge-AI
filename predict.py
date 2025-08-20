import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("model/disease_prediction.h5")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels
class_names = ['Acne and Rosacea', 'Actinic Keratosis','Atopic Dermatitis','Bullous Disease',
               'Cellulitis Impetigo','Eczema','Exanthems','Herpes HPV','Light disease','Lupus',
               'Melanoma Skin Cancer','Poison Ivy','Psoriasis','Seborrheic','Systemic Disease',
               'Tinea Ringworm','Urticaria Hives','Vascular Tumors','Vasculitis','Warts Molluscum']  

def predict_image(img_path, threshold=0.5):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Check confidence threshold
    if confidence >= threshold:
        print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
    else:
        print(f"Prediction rejected: Low confidence ({confidence:.2f})")

if __name__ == "__main__":
    img_path = input("Enter path to image: ")
    predict_image(img_path)
