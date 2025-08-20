from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load once at startup
model = load_model("model/disease_prediction.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

class_names = ['Acne and Rosacea', 'Actinic Keratosis','Atopic Dermatitis','Bullous Disease',
               'Cellulitis Impetigo','Eczema','Exanthems','Herpes HPV','Light disease','Lupus',
               'Melanoma Skin Cancer','Poison Ivy','Psoriasis','Seborrheic','Systemic Disease',
               'Tinea Ringworm','Urticaria Hives','Vascular Tumors','Vasculitis','Warts Molluscum']

@app.route("/receive", methods=["POST"])
def receive():
    data = request.json
    img_path = data.get("image_path")

    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        result = {
            "prediction": class_names[predicted_class],
            "confidence": float(confidence)
        }

        return jsonify({"status": "success", "received": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
