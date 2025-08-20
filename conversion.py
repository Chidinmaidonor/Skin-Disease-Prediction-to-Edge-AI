import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model/disease_prediction.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open("model/disease_prediction.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: disease_prediction.tflite saved.")
