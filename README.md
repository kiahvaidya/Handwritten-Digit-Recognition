🖊️ MNIST Digit Recognition App (Streamlit + TensorFlow)

This project is a simple yet powerful web app that allows users to draw digits (0–9) on a canvas and get predictions using a trained Convolutional Neural Network (CNN).
The app is built using Streamlit, TensorFlow and Pillow.

🚀 Features

🖌️ Interactive drawing canvas (streamlit-drawable-canvas)

🔢 Digit classification using a trained CNN model

⚡ Instant predictions

🧠 Uses a modern .keras model format for compatibility

📦 Clean project setup with Conda environment

🛠️ Tech Stack

Python 3.10

TensorFlow 2.16.1

Streamlit

NumPy

Pillow

streamlit-drawable-canvas


▶️ Run the App

python -m streamlit run app.py

This ensures Streamlit runs inside the correct Conda environment.

🧠 Model Info

The CNN model is trained on the MNIST dataset:

Conv2D → ReLU

MaxPooling

Conv2D → ReLU

Flatten

Dense (10 classes, Softmax)

If you use an old .h5 model, convert it with:

python convert.py



MIT License
