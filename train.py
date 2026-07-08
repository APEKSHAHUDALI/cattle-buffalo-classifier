"""
train.py — Train a cattle vs. buffalo image classifier.

Uses transfer learning on top of MobileNetV2 (pretrained on ImageNet) with a
small trainable classification head. Trains on images organized as:

    data/
        cattle/    (images of cattle)
        buffalo/   (images of buffalo)

Usage:
    python train.py
    python train.py --data-dir data --epochs 15 --batch-size 16
"""

import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["buffalo", "cattle"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cattle vs. buffalo classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Path to the training data directory (must contain 'cattle/' and 'buffalo/' subfolders).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument(
        "--model-out",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cattle_vs_buffalo_model.keras"),
        help="Where to save the trained model.",
    )
    return parser.parse_args()


def build_datasets(data_dir, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
    )

    print(f"Classes detected: {train_ds.class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


def build_model():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base model for transfer learning

    model = keras.Sequential(
        [
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),  # Binary classifier
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            "Expected a folder containing 'cattle/' and 'buffalo/' subfolders of images."
        )

    print(f"\nLoading dataset from: {args.data_dir}")
    train_ds, val_ds = build_datasets(args.data_dir, args.batch_size)

    print("\nBuilding model (MobileNetV2 transfer learning)...")
    model = build_model()
    model.summary()

    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

    final_val_acc = history.history["val_accuracy"][-1] * 100
    print(f"\nFinal validation accuracy: {final_val_acc:.2f}%")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"Model saved to '{args.model_out}'")


if __name__ == "__main__":
    main()
