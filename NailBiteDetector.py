from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model = load_model(
    "/Users/12salty/Documents/Coding/Python Projects/Anti-NailBiter/model/keras_model.h5",
    compile=False,
)


def analyzeImage(image, class_names):
    global model
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    # Predicts the model
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score
