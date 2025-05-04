import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_cifar10_optimized.h5")

# Label CIFAR-10
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload gambar berukuran 32x32 px (CIFAR-10)")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((32, 32))
    st.image(img, caption='Gambar yang di-upload', use_column_width=True)

    # Preprocessing
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"**Prediksi: {predicted_class}**")
