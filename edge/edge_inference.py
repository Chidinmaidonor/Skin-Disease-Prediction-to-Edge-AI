import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/disease_prediction.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img_path):
    img = Image.open(img_path).resize((128, 128))  # adjust if your model uses different size
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

if __name__ == "__main__":
    result = predict("C:/Users/DELL/Documents/Skin-Disease-Prediction-to-Edge-AI/data/train/Eczema Photos/eczema-acute-21.jpg")  # replace with your test image
    print("✅ Prediction:", result)
