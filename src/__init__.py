import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def extract_features(model_path, data_dir, output_path, image_size=224, batch_size=32):
    model = keras.models.load_model(model_path, compile=False)

    if "features" not in [layer.name for layer in model.layers]:
        raise ValueError(
            "The model does not contain a layer named 'features'. "
            "Add a GlobalAveragePooling2D layer with name='features' in the model to enable feature extraction."
        )

    class_names = sorted(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)))
    if len(class_names) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in {data_dir}, but found: {class_names}. "
            "Use a binary dataset structure like 'Parasitized' and 'Uninfected'."
        )

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="binary",
        classes=class_names,
        shuffle=False,
    )

    feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("features").output)
    features = feature_extractor.predict(generator, verbose=1)
    labels = generator.classes

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        class_names=np.array(class_names),
        class_indices=np.array(generator.class_indices, dtype=object),
    )

    print(f"Saved feature matrix to: {output_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract learned CNN features from malaria image data.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .keras model")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data by class")
    parser.add_argument("--output-file", type=str, default="features/extracted_features.npz", help="Output .npz file")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size used by the model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during feature extraction")
    args = parser.parse_args()

    extract_features(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output_file,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    main()
