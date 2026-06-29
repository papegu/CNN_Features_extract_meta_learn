"""
evaluate_mlp_full_real_world.py

Script pour évaluer le modèle MLP-Full (meilleur méta-apprenant) sur des données réelles
issues de PlantDoc (images de terrain). Utilise les backbones CNN pré-entraînés pour
extraire les caractéristiques, puis applique le classifieur MLP-Full sauvegardé.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Réduit pour éviter les OOM
SEED = 42

# Classes PlantDoc (38 classes similaires à PlantVillage)
PLANTDOC_CLASSES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy',
    'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bacterial Spot', 'Pepper Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Raspberry Healthy',
    'Soybean Healthy',
    'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus', 'Tomato Healthy'
]

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled")
        except RuntimeError as e:
            print(f"❌ GPU error: {e}")

class FeatureExtractor:
    """Extrait les caractéristiques des images via les trois backbones"""
    def __init__(self):
        self.backbones = {}
        self.build_backbones()
    
    def build_backbones(self):
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
        
        print("✅ Backbones chargés")
    
    def extract_features(self, image_paths, batch_size=BATCH_SIZE):
        """Extrait les caractéristiques concaténées pour une liste d'images"""
        n = len(image_paths)
        all_features = []
        
        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for path in batch_paths:
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, IMG_SIZE)
                img = tf.cast(img, tf.float32) / 255.0
                # Standardisation par image
                mean, var = tf.nn.moments(img, axes=[0, 1])
                img = (img - mean) / (tf.sqrt(var) + 1e-7)
                batch_images.append(img)
            
            batch_tensor = tf.stack(batch_images)
            
            # Extraire les caractéristiques de chaque backbone
            feats_resnet = self.backbones['resnet50'].predict(batch_tensor, verbose=0)
            feats_eff = self.backbones['efficientnetb0'].predict(batch_tensor, verbose=0)
            feats_mobile = self.backbones['mobilenetv2'].predict(batch_tensor, verbose=0)
            
            # Concaténer
            combined = np.concatenate([feats_resnet, feats_eff, feats_mobile], axis=1)
            all_features.append(combined)
            
            if (i//batch_size) % 10 == 0:
                print(f"  Traité {i}/{n} images")
        
        return np.vstack(all_features)

class MLPFullModel:
    """Charge le modèle MLP-Full entraîné et fait des prédictions"""
    def __init__(self, model_path, input_dim=4608, num_classes=38):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
        self.model.load_weights(model_path)
        print(f"✅ Modèle chargé depuis {model_path}")
    
    def _build_model(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def predict_classes(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

def download_plantdoc(data_dir='datasets/plantdoc'):
    """Télécharge et extrait le dataset PlantDoc"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "plantdoc.zip")
    
    if os.path.exists(os.path.join(data_dir, "PlantDoc-Dataset-master")):
        print("✅ PlantDoc déjà téléchargé")
        return os.path.join(data_dir, "PlantDoc-Dataset-master")
    
    print("📥 Téléchargement de PlantDoc...")
    urllib.request.urlretrieve(url, zip_path)
    print("✅ Téléchargé, extraction...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(zip_path)
    print("✅ Extraction terminée")
    return os.path.join(data_dir, "PlantDoc-Dataset-master")

def load_plantdoc_images(data_dir, split='test', max_per_class=20):
    """Charge les chemins d'images et les étiquettes pour le split donné"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        print(f"❌ Split {split} non trouvé dans {split_dir}")
        return [], []
    
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    # Filtrer pour ne garder que les classes présentes dans PlantVillage (optionnel)
    # Ici on garde toutes les classes
    image_paths = []
    labels = []
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Limiter le nombre d'images par classe
        if max_per_class:
            images = images[:max_per_class]
        image_paths.extend(images)
        labels.extend([idx] * len(images))
        print(f"  Classe {idx}: {class_name} -> {len(images)} images")
    
    return image_paths, labels, class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers les poids du modèle MLP-Full (.h5)')
    parser.add_argument('--data_dir', type=str, default='datasets/plantdoc', help='Répertoire de destination pour PlantDoc')
    parser.add_argument('--max_per_class', type=int, default=20, help='Nombre max d\'images par classe pour l\'évaluation')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Split à évaluer')
    args = parser.parse_args()
    
    set_memory_growth()
    
    # 1. Télécharger/charger PlantDoc
    plantdoc_root = download_plantdoc(args.data_dir)
    print("\n📂 Chargement des images...")
    image_paths, labels, class_names = load_plantdoc_images(
        plantdoc_root, split=args.split, max_per_class=args.max_per_class
    )
    
    if len(image_paths) == 0:
        print("❌ Aucune image trouvée.")
        return
    
    print(f"\n📊 Total images chargées : {len(image_paths)}")
    
    # 2. Extraire les caractéristiques
    print("\n🔧 Initialisation des extracteurs de caractéristiques...")
    extractor = FeatureExtractor()
    
    print("\n🚀 Extraction des caractéristiques (cela peut prendre quelques minutes)...")
    X = extractor.extract_features(image_paths, batch_size=BATCH_SIZE)
    y = np.array(labels)
    print(f"✅ Caractéristiques extraites : {X.shape}")
    
    # 3. Charger le modèle MLP-Full
    print("\n🤖 Chargement du modèle MLP-Full...")
    mlp_model = MLPFullModel(args.model_path, input_dim=X.shape[1], num_classes=len(class_names))
    
    # 4. Prédire et évaluer
    print("\n📈 Évaluation...")
    y_pred = mlp_model.predict_classes(X)
    
    accuracy = accuracy_score(y, y_pred)
    print(f"\n🎯 Accuracy: {accuracy:.4f}")
    
    # Rapport détaillé
    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
    
    # Matrice de confusion
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion - MLP-Full sur PlantDoc')
    plt.tight_layout()
    plt.savefig('confusion_matrix_plantdoc.png', dpi=300)
    print("✅ Matrice de confusion sauvegardée dans confusion_matrix_plantoc.png")
    
    # Sauvegarder les résultats
    results = {
        'accuracy': float(accuracy),
        'num_samples': len(y),
        'per_class': classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    }
    with open('results_plantdoc.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Résultats sauvegardés dans results_plantdoc.json")

if __name__ == '__main__':
    main()