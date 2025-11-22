import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# --------------------
# Load Model
# --------------------
model = tf.keras.models.load_model("mnist_cnn_new.keras")

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="centered")

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit below")

# Canvas settings (black background, white strokes)
canvas_result = st_canvas(
    fill_color="#000000",      
    stroke_width=18,
    stroke_color="#FFFFFF",     # white ink
    background_color="#000000", # black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:

        # Convert to PIL
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode="RGBA")

        # Convert RGBA to grayscale properly
        img = img.convert("L")

        # Resize to MNIST format
        img = img.resize((28, 28))

        # Convert to array
        img_arr = np.array(img)

        # Normalize
        img_arr = img_arr / 255.0

        # In MNIST: white digit = high value, black bg = low value.
        # Our canvas matches this, so no need to invert.

        # Reshape for CNN
        img_arr = img_arr.reshape(1, 28, 28, 1)

        # Model prediction
        prediction = np.argmax(model.predict(img_arr))

        st.subheader(f"Predicted Digit: **{prediction}**")




