import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load trained model
model = load_model("my_model.keras")

st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0-9) below and click **Predict**")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # black
    stroke_width=15,
    stroke_color="#FFFFFF",  # white ink
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Preprocess image
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        st.success(f"Predicted Digit: **{predicted_class}**")
        st.bar_chart(prediction[0])
    else:
        st.error("Please draw a digit first.")
