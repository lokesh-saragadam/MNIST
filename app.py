import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('trained_model.h5')

st.title("Handwritten Digit Recognition 🎨")
st.markdown("Draw a digit (0–9) below and let the model predict it!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black ink
    stroke_width=12,
    stroke_color="#FFFFFF",  # White strokes
    background_color="#000000",  # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Process image
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    st.image(img.reshape(28, 28), caption='Resized Input', width=150)

    # Predict
    if st.button("Predict"):
        pred = model.predict(img)
        st.write(f"### Predicted Digit: {np.argmax(pred)}")