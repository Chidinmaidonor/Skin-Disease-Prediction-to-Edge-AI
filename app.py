import streamlit as st
import requests
import os

st.title("🧑‍⚕️ Skin Disease Prediction (via Flask API)")

# Upload image
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporarily
    temp_path = os.path.join("temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            # 🔗 Send request to Flask API
            response = requests.post(
                "http://127.0.0.1:5000/receive",
                json={"image_path": temp_path}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Prediction: {result['received']['prediction']} "
                           f"(Confidence: {result['received']['confidence']:.2f})")
            else:
                st.error(f"❌ API Error: {response.text}")

        except Exception as e:
            st.error(f"⚠️ Could not connect to Flask API: {e}")
