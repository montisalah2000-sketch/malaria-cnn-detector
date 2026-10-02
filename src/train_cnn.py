import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def build_model(input_shape=(224, 224, 3)):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="features")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="malaria_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )
    return model


def get_class_names(data_dir):
    class_names = sorted(
        name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))
    )
    if len(class_names) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in {data_dir}, but found: {class_names}. "
            "Use a binary folder layout like 'Parasitized' and 'Uninfected'."
        )
    return class_names


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for malaria image classification.")
    parser.add_argument("--train-dir", type=str, required=True, help="Directory containing training images by class")
    parser.add_argument("--val-dir", type=str, required=True, help="Directory containing validation images by class")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for the CNN")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    class_names_train = get_class_names(args.train_dir)
    class_names_val = get_class_names(args.val_dir)

    if class_names_train != class_names_val:
        raise ValueError(
            f"Training and validation class names do not match: {class_names_train} vs {class_names_val}"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="binary",
        classes=class_names_train,
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="binary",
        classes=class_names_val,
        shuffle=False,
    )

    model = build_model(input_shape=(args.image_size, args.image_size, 3))
    model.summary()

    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        TensorBoard(log_dir=os.path.join(args.output_dir, "logs")),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nBest model saved to: {checkpoint_path}")
    print("Training complete.")

    metrics = {
        "train_loss": history.history["loss"][-1],
        "train_accuracy": history.history["accuracy"][-1],
        "val_loss": history.history["val_loss"][-1],
        "val_accuracy": history.history["val_accuracy"][-1],
    }
    print("Final metrics:", metrics)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    main()
