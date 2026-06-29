"""
transfer_mlp_full_to_multicrop.py

Adapte le modèle MLP-Full pré-entraîné sur PlantVillage (38 classes) au dataset Multi-Crop (30 classes).
Charge les poids, remplace la dernière couche, et ré-entraîne la nouvelle couche.
Nécessite les données d'entraînement de Multi-Crop au format YOLO (train/images, train/labels).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
import argparse
import joblib
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Classes du dataset Multi-Crop (30 classes)
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

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

class FeatureExtractor:
    """Extracteur de caractéristiques multi-backbones (identique à l'entraînement)"""
    def __init__(self):
        self.backbones = {}
        self._build_backbones()

    def _build_backbones(self):
        input_shape = (*IMG_SIZE, 3)
        # ResNet50
        base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_resnet.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_resnet(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['resnet50'] = models.Model(inputs=inputs, outputs=x)

        # EfficientNetB0
        base_eff = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_eff.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_eff(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['efficientnetb0'] = models.Model(inputs=inputs, outputs=x)

        # MobileNetV2
        base_mobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_mobile.trainable = False
        inputs = layers.Input(shape=input_shape)
        x = base_mobile(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        self.backbones['mobilenetv2'] = models.Model(inputs=inputs, outputs=x)

        print("✅ Backbones loaded")

    def extract_features(self, image_paths, batch_size=BATCH_SIZE):
        """Extrait les caractéristiques concaténées pour une liste de chemins d'images."""
        all_features = []
        n = len(image_paths)
        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, IMG_SIZE)
                img = tf.cast(img, tf.float32) / 255.0
                # Standardisation par image (identique à l'entraînement)
                mean, var = tf.nn.moments(img, axes=[0, 1])
                img = (img - mean) / (tf.sqrt(var) + 1e-7)
                batch_images.append(img)
            batch_tensor = tf.stack(batch_images)

            # Extraire les caractéristiques de chaque backbone
            feats_resnet = self.backbones['resnet50'].predict(batch_tensor, verbose=0)
            feats_eff = self.backbones['efficientnetb0'].predict(batch_tensor, verbose=0)
            feats_mobile = self.backbones['mobilenetv2'].predict(batch_tensor, verbose=0)
            combined = np.concatenate([feats_resnet, feats_eff, feats_mobile], axis=1)
            all_features.append(combined)
            print(f"  Extracted {min(i+batch_size, n)}/{n} images")
        return np.vstack(all_features)

def load_images_and_labels_yolo(images_dir, labels_dir, class_names):
    """Charge les chemins d'images et les étiquettes depuis le format YOLO"""
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
        with open(label_path, 'r') as f:
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
    """Construit le MLP complet avec 38 classes (pour charger les poids)"""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes_original, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def main():
    parser = argparse.ArgumentParser(description='Transfer learning MLP-Full to Multi-Crop')
    parser.add_argument('--train_images', type=str, required=True, help='Dossier train/images')
    parser.add_argument('--train_labels', type=str, required=True, help='Dossier train/labels')
    parser.add_argument('--val_images', type=str, default=None, help='Dossier val/images (optionnel)')
    parser.add_argument('--val_labels', type=str, default=None, help='Dossier val/labels (optionnel)')
    parser.add_argument('--pretrained_weights', type=str, default='models/mlp_full.weights.h5',
                        help='Poids pré-entraînés sur PlantVillage')
    parser.add_argument('--output_weights', type=str, default='models/mlp_full_multicrop.weights.h5')
    parser.add_argument('--scaler_path', type=str, default='models/scaler_multicrop.pkl')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_memory_growth()

    # 1. Charger les données d'entraînement
    print("📂 Chargement des images d'entraînement...")
    train_paths, train_labels = load_images_and_labels_yolo(args.train_images, args.train_labels, CLASS_NAMES)
    print(f"   {len(train_paths)} images chargées.")
    if args.val_images and args.val_labels:
        val_paths, val_labels = load_images_and_labels_yolo(args.val_images, args.val_labels, CLASS_NAMES)
        print(f"   {len(val_paths)} images de validation.")
    else:
        # Split train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=SEED, stratify=train_labels
        )
        print(f"   Split: {len(train_paths)} train, {len(val_paths)} val.")

    # 2. Construire l'extracteur de caractéristiques
    print("🔧 Construction de l'extracteur de caractéristiques...")
    extractor = FeatureExtractor()

    # 3. Extraire les caractéristiques pour train et val
    print("🚀 Extraction des caractéristiques pour train...")
    X_train = extractor.extract_features(train_paths, batch_size=args.batch_size)
    y_train = train_labels
    print("🚀 Extraction des caractéristiques pour val...")
    X_val = extractor.extract_features(val_paths, batch_size=args.batch_size)
    y_val = val_labels

    # 4. Normalisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Construire le modèle MLP avec 30 classes
    inputs = layers.Input(shape=(X_train_scaled.shape[1],))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(NUM_CLASSES_NEW, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    # 6. Charger les poids pré-entraînés (sur 38 classes)
    print("🤖 Chargement des poids pré-entraînés...")
    pretrained_full = build_base_mlp(X_train_scaled.shape[1], num_classes_original=38)
    pretrained_full.load_weights(args.pretrained_weights)
    print("✅ Poids pré-entraînés chargés.")

    # Transférer les poids des couches communes (toutes sauf la dernière)
    for layer in model.layers[:-1]:  # Toutes sauf la dernière couche Dense
        if layer.name in [l.name for l in pretrained_full.layers]:
            try:
                layer.set_weights(pretrained_full.get_layer(layer.name).get_weights())
                print(f"   ✅ Poids chargés pour {layer.name}")
            except Exception as e:
                print(f"   ⚠️ Impossible de charger {layer.name}: {e}")

    # Geler toutes les couches sauf la dernière
    for layer in model.layers[:-1]:
        layer.trainable = False

    # Compiler
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Modèle adapté, entraînement de la nouvelle couche...")

    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # Class weights (optionnel)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # Entraînement
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # 7. Sauvegarder le nouveau modèle et le scaler
    os.makedirs(os.path.dirname(args.output_weights), exist_ok=True)
    model.save_weights(args.output_weights)
    joblib.dump(scaler, args.scaler_path)
    print(f"✅ Nouveaux poids sauvegardés dans {args.output_weights}")
    print(f"✅ Scaler sauvegardé dans {args.scaler_path}")

if __name__ == '__main__':
    main()