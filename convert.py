import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.h5", compile=False)
model.save("mnist_cnn_new.keras")
