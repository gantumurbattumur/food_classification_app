import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np

# Load model
model = keras.models.load_model("food_classification_model.keras")

# Class names (must match training order)
class_names = ['breakfast', 'dessert', 'meat', 'rice', 'flour', 'other_main']

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # same size as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    predictions = model.predict(img_array)[0]  # take first (only) prediction
    return predictions

if __name__ == "__main__":
    test_img = "test_image.jpeg"
    probs = predict_food(test_img)

    # Get top prediction
    top_idx = np.argmax(probs)
    print(f"\n Predicted: {class_names[top_idx]} ({probs[top_idx]*100:.2f}%)\n")

    # Show all class probabilities
    print(" Class probabilities:")
    for cls, prob in zip(class_names, probs):
        print(f" - {cls}: {prob*100:.2f}%")
