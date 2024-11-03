import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('my_model')

# Define the class labels
class_names = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Eczema', 'Melanocytic Nevi', 
    'Melanoma', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Warts Molluscum and other Viral Infections'
]

# Preprocess the image
def preprocess_image(image):
    img = image.convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Classify the image
def classify_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = class_names[predicted_class]
    confidence = np.max(predictions) * 100
    return class_name, confidence
