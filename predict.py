"""
predict.py — Classify a single image as 'cattle' or 'buffalo' using a trained model.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image data/test_image/some_photo.webp --model models/cattle_vs_buffalo_model.keras
    python predict.py   (will prompt for an image path interactively)
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["buffalo", "cattle"]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "cattle_vs_buffalo_model.keras"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Classify an image as cattle or buffalo.")
    parser.add_argument("--image", type=str, default=None, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained .keras model file.",
    )
    return parser.parse_args()


def predict(model_path, image_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\nTrain one first by running: python train.py"
        )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]
    predicted_index = int(round(prediction))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = prediction if predicted_label == "cattle" else 1 - prediction

    print(f"\nPrediction: this image is a {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2%}")
    return predicted_label, float(confidence)


def main():
    args = parse_args()
    image_path = args.image or input("\nEnter the full path of the test image: ").strip('"')
    predict(args.model, image_path)


if __name__ == "__main__":
    main()
