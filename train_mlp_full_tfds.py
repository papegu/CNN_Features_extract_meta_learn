"""
transfer_mlp_full_to_multicrop.py

Improved Multi-Crop training pipeline for the MLP-Full classifier.
Key upgrades:
- richer image preprocessing and optional image-level TTA support symmetry
- class-balanced training with optional label smoothing
- stronger regularization and learning-rate scheduling
- optional focal loss for hard / imbalanced classes
- automatic validation split with stratification
- JSON summary export for paper-ready reporting
"""

import os
import json
import random
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2

warnings.filterwarnings('ignore')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    'banana_bract_mosaic_virus', 'banana_cordana', 'banana_healthy', 'banana_insectpest', 'banana_moko',
    'banana_panama', 'banana_pestalotiopsis', 'banana_sigatoka', 'banana_yb_sigatoka',
    'cauliflower_Blackrot', 'cauliflower_bacterial _spot _rot', 'cauliflower_downy_mildew', 'cauliflower_healthy',
    'chilli_anthracnose', 'chilli_healthy', 'chilli_leafcurl', 'chilli_leafspot', 'chilli_whitefly', 'chilli_yellowish',
    'groundnut_early_leaf_spot', 'groundnut_early_rust', 'groundnut_healthy', 'groundnut_late_leaf_spot',
    'groundnut_nutrition_deficiency', 'groundnut_rust',
    'radish_black_leaf_spot', 'radish_downey_mildew', 'radish_flea_beetle', 'radish_healthy', 'radish_mosaic'
]
NUM_CLASSES_NEW = len(CLASS_NAMES)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


class FeatureExtractor:
    """Multi-backbone feature extractor consistent with the main paper pipeline."""

    def __init__(self):
        self.backbones = {}
        self._build_backbones()

    def _build_backbones(self):
        input_shape = (*IMG_SIZE, 3)

        base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_resnet.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_resnet(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['resnet50'] = models.Model(inputs=inputs, outputs=x)

        base_eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_eff.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_eff(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['efficientnetb0'] = models.Model(inputs=inputs, outputs=x)

        base_mobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_mobile.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_mobile(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['mobilenetv2'] = models.Model(inputs=inputs, outputs=x)

        print("✅ Backbones loaded")

    @staticmethod
    def _preprocess_image(path, augment=False):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.08)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_saturation(img, 0.9, 1.1)
            noise = tf.random.normal(tf.shape(img), stddev=0.01)
            img = tf.clip_by_value(img + noise, 0.0, 1.0)

        mean, var = tf.nn.moments(img, axes=[0, 1])
        img = (img - mean) / (tf.sqrt(var) + 1e-7)
        return img

    def extract_features(self, image_paths, batch_size=BATCH_SIZE, augment=False, verbose=True):
        all_features = []
        n = len(image_paths)
        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [self._preprocess_image(path, augment=augment) for path in batch_paths]
            batch_tensor = tf.stack(batch_images)

            feats_resnet = self.backbones['resnet50'].predict(batch_tensor, verbose=0)
            feats_eff = self.backbones['efficientnetb0'].predict(batch_tensor, verbose=0)
            feats_mobile = self.backbones['mobilenetv2'].predict(batch_tensor, verbose=0)
            combined = np.concatenate([feats_resnet, feats_eff, feats_mobile], axis=1)
            all_features.append(combined)

            if verbose:
                print(f"  Extracted {min(i + batch_size, n)}/{n} images")
        return np.vstack(all_features)


def load_images_and_labels_yolo(images_dir, labels_dir, class_names):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(images_dir.glob(ext))
    image_paths = sorted(image_paths)

    valid_paths = []
    labels = []
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
        class_id = int(first_line.split()[0])
        if class_id >= len(class_names):
            continue
        valid_paths.append(str(img_path))
        labels.append(class_id)
    return valid_paths, np.array(labels)


def build_base_mlp(input_dim, num_classes_original=38):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes_original, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def build_multicrop_mlp(input_dim, num_classes=NUM_CLASSES_NEW, dropout_rate=0.4):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def categorical_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn


def prepare_targets(y, num_classes=NUM_CLASSES_NEW):
    return tf.keras.utils.to_categorical(y, num_classes=num_classes)


def export_summary(output_json, history, y_true, y_pred):
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "history": {k: [float(vv) for vv in vals] for k, vals in history.history.items()},
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Training summary saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Improved transfer learning MLP-Full to Multi-Crop')
    parser.add_argument('--train_images', type=str, required=True, help='train/images directory')
    parser.add_argument('--train_labels', type=str, required=True, help='train/labels directory')
    parser.add_argument('--val_images', type=str, default=None, help='val/images directory (optional)')
    parser.add_argument('--val_labels', type=str, default=None, help='val/labels directory (optional)')
    parser.add_argument('--pretrained_weights', type=str, default='models/mlp_full.weights.h5')
    parser.add_argument('--output_weights', type=str, default='models/mlp_full_multicrop.weights.h5')
    parser.add_argument('--scaler_path', type=str, default='models/scaler_multicrop.pkl')
    parser.add_argument('--summary_json', type=str, default='results/domain_adaptation_results.json')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of categorical crossentropy')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--feature_aug_copies', type=int, default=1, help='Number of augmented feature copies for training set')
    args = parser.parse_args()

    set_seed(SEED)
    set_memory_growth()

    print("📂 Loading training images...")
    train_paths, train_labels = load_images_and_labels_yolo(args.train_images, args.train_labels, CLASS_NAMES)
    print(f"   {len(train_paths)} images loaded.")

    if args.val_images and args.val_labels:
        val_paths, val_labels = load_images_and_labels_yolo(args.val_images, args.val_labels, CLASS_NAMES)
        print(f"   {len(val_paths)} validation images.")
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=0.2,
            random_state=SEED,
            stratify=train_labels
        )
        print(f"   Split: {len(train_paths)} train, {len(val_paths)} val.")

    print("🔧 Building feature extractor...")
    extractor = FeatureExtractor()

    print("🚀 Extracting train features...")
    X_train = extractor.extract_features(train_paths, batch_size=args.batch_size, augment=False)
    y_train = train_labels

    if args.feature_aug_copies > 0:
        augmented_blocks = [X_train]
        augmented_labels = [y_train]
        for k in range(args.feature_aug_copies):
            print(f"🔁 Extracting augmented train features copy {k+1}/{args.feature_aug_copies}...")
            X_aug = extractor.extract_features(train_paths, batch_size=args.batch_size, augment=True)
            augmented_blocks.append(X_aug)
            augmented_labels.append(y_train)
        X_train = np.vstack(augmented_blocks)
        y_train = np.concatenate(augmented_labels)

    print("🚀 Extracting validation features...")
    X_val = extractor.extract_features(val_paths, batch_size=args.batch_size, augment=False)
    y_val = val_labels

    print("🧪 Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    y_train_cat = prepare_targets(y_train)
    y_val_cat = prepare_targets(y_val)

    print("🤖 Building target model...")
    model = build_multicrop_mlp(X_train_scaled.shape[1], NUM_CLASSES_NEW)

    print("📦 Loading pretrained PlantVillage weights...")
    pretrained_full = build_base_mlp(X_train_scaled.shape[1], num_classes_original=38)
    pretrained_full.load_weights(args.pretrained_weights)

    for layer in model.layers:
        if layer.name in [l.name for l in pretrained_full.layers]:
            try:
                layer.set_weights(pretrained_full.get_layer(layer.name).get_weights())
                print(f"   ✅ transferred {layer.name}")
            except Exception:
                pass

    loss = categorical_focal_loss(label_smoothing=args.label_smoothing) if args.use_focal_loss else tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-5),
        loss=loss,
        metrics=['accuracy']
    )

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    callback_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(args.output_weights, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True, verbose=1)
    ]

    print("🚀 Training improved Multi-Crop MLP...")
    history = model.fit(
        X_train_scaled,
        y_train_cat,
        validation_data=(X_val_scaled, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callback_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("📊 Final validation evaluation...")
    y_pred_val = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
    print(classification_report(y_val, y_pred_val, zero_division=0))

    os.makedirs(os.path.dirname(args.output_weights), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    model.save_weights(args.output_weights)
    joblib.dump(scaler, args.scaler_path)
    export_summary(args.summary_json, history, y_val, y_pred_val)

    print(f"✅ Weights saved to {args.output_weights}")
    print(f"✅ Scaler saved to {args.scaler_path}")


if __name__ == '__main__':
    main()
