"""
test_multicrop_adapted.py

Improved evaluation script for the adapted MLP-Full model (30 classes)
on the Multi-Crop dataset.
Compatible with the upgraded training architecture in train_mlp_full_tfds.py.
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2

warnings.filterwarnings('ignore')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    'banana_bract_mosaic_virus', 'banana_cordana', 'banana_healthy', 'banana_insectpest', 'banana_moko',
    'banana_panama', 'banana_pestalotiopsis', 'banana_sigatoka', 'banana_yb_sigatoka',
    'cauliflower_Blackrot', 'cauliflower_bacterial _spot _rot', 'cauliflower_downy_mildew', 'cauliflower_healthy',
    'chilli_anthracnose', 'chilli_healthy', 'chilli_leafcurl', 'chilli_leafspot', 'chilli_whitefly', 'chilli_yellowish',
    'groundnut_early_leaf_spot', 'groundnut_early_rust', 'groundnut_healthy', 'groundnut_late_leaf_spot',
    'groundnut_nutrition_deficiency', 'groundnut_rust',
    'radish_black_leaf_spot', 'radish_downey_mildew', 'radish_flea_beetle', 'radish_healthy', 'radish_mosaic'
]
NUM_CLASSES = len(CLASS_NAMES)


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def build_feature_extractor():
    """Build feature extractor (single input, three backbones)."""
    input_shape = (*IMG_SIZE, 3)
    inputs = layers.Input(shape=input_shape)

    base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_resnet.trainable = False
    x_resnet = base_resnet(inputs)
    x_resnet = layers.GlobalAveragePooling2D()(x_resnet)

    base_eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_eff.trainable = False
    x_eff = base_eff(inputs)
    x_eff = layers.GlobalAveragePooling2D()(x_eff)

    base_mobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_mobile.trainable = False
    x_mobile = base_mobile(inputs)
    x_mobile = layers.GlobalAveragePooling2D()(x_mobile)

    concatenated = layers.Concatenate()([x_resnet, x_eff, x_mobile])
    model = models.Model(inputs=inputs, outputs=concatenated)
    return model


class MLPFullModelAdapted:
    """Improved MLP-Full model adapted to 30 classes."""

    def __init__(self, input_dim, num_classes, weights_path):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.model = self._build_model()
        self.model.load_weights(weights_path)
        print(f"✅ Loaded adapted model weights from {weights_path}")

    def _build_model(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def predict_classes(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1)


class TTAWrapper:
    """Simple feature-space test-time augmentation wrapper."""

    @staticmethod
    def predict_with_tta(model_wrapper, X, n_augments=5):
        all_preds = []
        for _ in range(n_augments):
            noise = np.random.normal(0, 0.01, X.shape)
            scale = np.random.uniform(0.98, 1.02, size=(X.shape[0], 1))
            X_aug = (X + noise) * scale
            preds = model_wrapper.predict(X_aug)
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)


def load_images_and_labels_yolo(images_dir, labels_dir, class_names, limit_per_class=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(images_dir.glob(ext))
    image_paths = sorted(image_paths)

    valid_paths = []
    labels = []
    class_counts = {}
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        if not first_line:
            continue
        parts = first_line.split()
        class_id = int(parts[0])
        if class_id >= len(class_names):
            continue

        if limit_per_class is not None:
            current = class_counts.get(class_id, 0)
            if current >= limit_per_class:
                continue
            class_counts[class_id] = current + 1

        valid_paths.append(str(img_path))
        labels.append(class_id)

    return valid_paths, np.array(labels)


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    mean, var = tf.nn.moments(img, axes=[0, 1])
    img = (img - mean) / (tf.sqrt(var) + 1e-7)
    return img


def extract_features(image_paths, feature_extractor, batch_size=BATCH_SIZE):
    all_features = []
    n = len(image_paths)
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [preprocess_image(path) for path in batch_paths]
        batch_tensor = tf.stack(batch_images)
        feats = feature_extractor.predict(batch_tensor, verbose=0)
        all_features.append(feats)
        print(f"  Processed {min(i + batch_size, n)}/{n}")
    return np.vstack(all_features)


def save_json_results(output_dir, accuracy, y_true, y_pred, target_names):
    present_labels = sorted(np.unique(y_true))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    summary = {
        'accuracy': float(accuracy),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'num_samples': int(len(y_true)),
        'classification_report': report_dict
    }
    json_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Evaluation summary saved to {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True, help='Test images directory')
    parser.add_argument('--labels_dir', type=str, required=True, help='Test labels directory')
    parser.add_argument('--model_weights', type=str, required=True, help='Adapted model weights (.h5)')
    parser.add_argument('--scaler_path', type=str, required=True, help='Scaler .pkl')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='test_results')
    parser.add_argument('--limit_per_class', type=int, default=None, help='Optional cap per class for quick evaluation')
    parser.add_argument('--use_tta', action='store_true', help='Enable simple feature-space TTA')
    parser.add_argument('--tta_augments', type=int, default=5, help='Number of TTA augmentations')
    args = parser.parse_args()

    set_memory_growth()
    os.makedirs(args.output_dir, exist_ok=True)

    print("📂 Loading test images...")
    image_paths, y_true = load_images_and_labels_yolo(
        args.images_dir,
        args.labels_dir,
        CLASS_NAMES,
        limit_per_class=args.limit_per_class
    )
    print(f"   Total: {len(image_paths)} images")

    print("🔧 Building feature extractor...")
    feat_extractor = build_feature_extractor()

    print("🚀 Extracting features...")
    X = extract_features(image_paths, feat_extractor, args.batch_size)
    print(f"   Features shape: {X.shape}")

    print("🧪 Loading scaler...")
    scaler = joblib.load(args.scaler_path)
    X_scaled = scaler.transform(X)

    print("🤖 Loading adapted MLP model...")
    mlp_model = MLPFullModelAdapted(X_scaled.shape[1], NUM_CLASSES, args.model_weights)

    if args.use_tta:
        print(f"✨ Running evaluation with TTA ({args.tta_augments} augmentations)...")
        y_pred_proba = TTAWrapper.predict_with_tta(mlp_model, X_scaled, n_augments=args.tta_augments)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = mlp_model.predict_classes(X_scaled)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n🎯 Accuracy: {acc:.4f}")
    print(f"🎯 Macro-F1: {macro_f1:.4f}")
    print(f"🎯 Weighted-F1: {weighted_f1:.4f}")

    present_labels = sorted(np.unique(y_true))
    target_names = [CLASS_NAMES[i] for i in present_labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
    print("\n📋 Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    plt.figure(figsize=(20, 16))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(target_names)), target_names, rotation=90, fontsize=8)
    plt.yticks(range(len(target_names)), target_names, fontsize=8)
    plt.title(f'Confusion Matrix - Adapted MLP-Full\nAccuracy: {acc:.2%}')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print("✅ Confusion matrix saved.")

    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    save_json_results(args.output_dir, acc, y_true, y_pred, target_names)


if __name__ == '__main__':
    main()
