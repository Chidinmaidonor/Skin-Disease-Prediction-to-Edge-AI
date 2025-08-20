import numpy as np
import tensorflow as tf
from PIL import Image
import requests

# Class labels
class_names = [
    'Acne and Rosacea', 'Actinic Keratosis', 'Atopic Dermatitis', 'Bullous Disease',
    'Cellulitis Impetigo', 'Eczema', 'Exanthems', 'Herpes HPV', 'Light disease', 'Lupus',
    'Melanoma Skin Cancer', 'Poison Ivy', 'Psoriasis', 'Seborrheic', 'Systemic Disease',
    'Tinea Ringworm', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts Molluscum'
]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/disease_prediction.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img_path, threshold=0.5):
    try:
        # Load and preprocess image
        img = Image.open(img_path).resize((128, 128))
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Show result
        if confidence >= threshold:
            result = {
                "prediction": class_names[predicted_class],
                "confidence": float(confidence)
            }
            print(f"\n✅ Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})\n")
        else:
            result = {
                "prediction": "Low confidence",
                "confidence": float(confidence)
            }
            print(f"\n⚠️ Prediction rejected: Low confidence ({confidence:.2f})\n")

        # Send result to Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/receive", json=result)
            print("📡 Sent to Flask API, response:", response.text)
        except Exception as e:
            print("⚠️ Could not send to Flask API:", e)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    img_path = input("📂 Enter path to image: ").strip()
    predict(img_path)
