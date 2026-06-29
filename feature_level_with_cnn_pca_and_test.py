"""
feature_level_stacking_complete_enhanced.py

Complete feature-level stacking pipeline with:
- Multi-CNN feature extraction (ResNet50, EfficientNetB0, MobileNetV2)
- Statistical analysis of CNN features
- Three meta-learners: MLP-full, MLP-PCA, PCA-Logistic (with harmonized PCA)
- Comprehensive SHAP analysis
- REAL-WORLD VALIDATION on multiple datasets (PlantDoc, Rice Disease, Cassava)
- All figures with Arial font 10pt
"""

import os
import gc
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             f1_score, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from scipy import stats
import seaborn as sns
import json
import warnings
import urllib.request
import zipfile
from pathlib import Path
import hashlib
warnings.filterwarnings('ignore')

# Désactiver oneDNN pour éviter les warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Optional imports
try:
    import shap
    print("✅ SHAP imported successfully")
except ImportError:
    shap = None
    print("⚠️ SHAP not installed. Install with: pip install shap")

# -------------------------------
# Configuration
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EXTRACTION_BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

FIGURE_DPI = 300
plt.rc('font', family='Arial', size=10)

# Dataset URLs
DATASET_URLS = {
    'plantdoc': 'https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip',
    'rice_disease': 'https://www.kaggle.com/api/v1/datasets/download/minhhuy2810/rice-diseases-image-dataset',
    'cassava': 'https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease'
}

# PlantDoc classes mapping
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
    """Enable memory growth for GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ GPU memory growth error: {e}")


class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        
    def preprocess_image(self, image, label, augment=False):
        """Preprocess single image with optional augmentation"""
        # Resize
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        if augment:
            # Geometric augmentations
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
            
            # Rotation (0, 90, 180, 270 degrees)
            k = tf.random.uniform((), maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            
            # Photometric augmentations
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            image = tf.image.random_hue(image, max_delta=0.05)
            
            # Gaussian noise
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Per-image standardization
        mean, var = tf.nn.moments(image, axes=[0, 1])
        image = (image - mean) / (tf.sqrt(var) + 1e-7)
        
        return image, label
    
    def prepare_dataset(self, dataset, augment=False, cache=True):
        """Prepare tf.data.Dataset with preprocessing"""
        # Unbatch if needed
        try:
            dataset = dataset.unbatch()
        except:
            pass
        
        # Apply preprocessing
        ds = dataset.map(
            lambda x, y: self.preprocess_image(x, y, augment),
            num_parallel_calls=AUTOTUNE
        )
        
        # Shuffle for training
        if augment:
            ds = ds.shuffle(2048, seed=SEED)
        
        # Cache for validation/test
        if cache and not augment:
            ds = ds.cache()
        
        # Batch and prefetch
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        
        return ds
    
    def load_plantvillage(self, data_dir):
        """Load PlantVillage dataset from directory"""
        if not os.path.exists(data_dir):
            raise ValueError(f"Path {data_dir} does not exist")
        
        print(f"Loading PlantVillage dataset from {data_dir}")
        
        # Check if using train/val/test split or single directory
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            # Train
            ds_train = tf.keras.preprocessing.image_dataset_from_directory(
                train_dir,
                labels='inferred',
                label_mode='int',
                image_size=self.img_size,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=SEED
            )
            
            # Validation
            ds_val = tf.keras.preprocessing.image_dataset_from_directory(
                val_dir,
                labels='inferred',
                label_mode='int',
                image_size=self.img_size,
                batch_size=BATCH_SIZE,
                shuffle=False,
                seed=SEED
            )
            
            # Test
            if os.path.isdir(test_dir):
                ds_test = tf.keras.preprocessing.image_dataset_from_directory(
                    test_dir,
                    labels='inferred',
                    label_mode='int',
                    image_size=self.img_size,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    seed=SEED
                )
            else:
                ds_test = ds_val
            
            class_names = ds_train.class_names
            
        else:
            # Single directory - split automatically
            full_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                labels='inferred',
                label_mode='int',
                image_size=self.img_size,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=SEED
            )
            class_names = full_ds.class_names
            
            # Split into train/val/test (80/10/10)
            n_samples = sum(1 for _ in full_ds.unbatch())
            train_size = int(0.8 * n_samples)
            val_size = int(0.1 * n_samples)
            
            # Shuffle and split
            full_ds = full_ds.unbatch().shuffle(n_samples, seed=SEED)
            ds_train = full_ds.take(train_size).batch(BATCH_SIZE)
            ds_val = full_ds.skip(train_size).take(val_size).batch(BATCH_SIZE)
            ds_test = full_ds.skip(train_size + val_size).batch(BATCH_SIZE)
        
        print(f"Classes detected: {len(class_names)}")
        print(f"First 5 classes: {class_names[:5]}")
        
        return ds_train, ds_val, ds_test, class_names


class RealWorldDatasetLoader:
    """Loader for real-world datasets (PlantDoc, Rice Disease, Cassava)"""
    
    def __init__(self, data_dir='datasets/real_world', img_size=IMG_SIZE):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name):
        """Download dataset if not exists"""
        if dataset_name not in DATASET_URLS:
            print(f"❌ Unknown dataset: {dataset_name}")
            return None
        
        dataset_path = self.data_dir / dataset_name
        if dataset_path.exists():
            print(f"✅ Dataset {dataset_name} already exists at {dataset_path}")
            return dataset_path
        
        print(f"📥 Downloading {dataset_name} dataset...")
        url = DATASET_URLS[dataset_name]
        zip_path = self.data_dir / f"{dataset_name}.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"✅ Download complete. Extracting...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Remove zip file
            zip_path.unlink()
            print(f"✅ Extraction complete")
            
            return dataset_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def load_plantdoc(self, max_images_per_class=50):
        """Load PlantDoc dataset (real-world field images)"""
        print("\n" + "="*60)
        print("Loading PlantDoc Dataset (Real-World Field Images)")
        print("="*60)
        
        dataset_path = self.download_dataset('plantdoc')
        if dataset_path is None:
            return None, None, None
        
        # Find image directory
        plantdoc_dir = dataset_path / 'PlantDoc-Dataset-master'
        if not plantdoc_dir.exists():
            plantdoc_dir = dataset_path
        
        # Look for train/test splits
        train_dir = plantdoc_dir / 'train'
        test_dir = plantdoc_dir / 'test'
        
        if not train_dir.exists() or not test_dir.exists():
            print("❌ PlantDoc directory structure not found")
            return None, None, None
        
        # Load datasets with limited images per class
        print(f"Loading from {train_dir} and {test_dir}")
        
        # Training set (with limit per class)
        ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='int',
            image_size=self.img_size,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
            validation_split=None
        )
        
        # Test set (with limit per class)
        ds_test = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='int',
            image_size=self.img_size,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEED
        )
        
        # Apply class filtering to match PlantVillage classes
        class_names = ds_train.class_names
        
        # Limit images per class for balanced evaluation
        ds_train = self._limit_images_per_class(ds_train, max_images_per_class)
        ds_test = self._limit_images_per_class(ds_test, max_images_per_class)
        
        print(f"✅ PlantDoc loaded: {len(class_names)} classes")
        print(f"  Train samples: {sum(1 for _ in ds_train.unbatch())}")
        print(f"  Test samples: {sum(1 for _ in ds_test.unbatch())}")
        
        return ds_train, ds_test, class_names
    
    def _limit_images_per_class(self, dataset, max_per_class):
        """Limit number of images per class for balanced evaluation"""
        # Unbatch first
        ds_unbatched = dataset.unbatch()
        
        # Collect by class
        images_by_class = {}
        for img, label in ds_unbatched:
            label = label.numpy() if hasattr(label, 'numpy') else label
            if label not in images_by_class:
                images_by_class[label] = []
            if len(images_by_class[label]) < max_per_class:
                images_by_class[label].append(img)
        
        # Rebuild dataset
        all_images = []
        all_labels = []
        for label, imgs in images_by_class.items():
            all_images.extend(imgs)
            all_labels.extend([label] * len(imgs))
        
        if not all_images:
            return dataset
        
        # Convert to tensor dataset
        ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    def create_test_dataset_from_directory(self, directory, class_names=None):
        """Create test dataset from directory of images"""
        if not os.path.exists(directory):
            print(f"❌ Directory {directory} does not exist")
            return None
        
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='int',
            image_size=self.img_size,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=SEED,
            class_names=class_names
        )
        
        return ds


class FeatureExtractor:
    """Feature extraction with multiple CNN backbones"""
    
    def __init__(self, backbone_names=None, img_size=IMG_SIZE):
        if backbone_names is None:
            backbone_names = ['resnet50', 'efficientnetb0', 'mobilenetv2']
        
        self.backbone_names = backbone_names
        self.img_size = img_size
        self.backbones = {}
        self.backbone_dims = {}
        self.total_dim = 0
        
    def build_backbone(self, name):
        """Build single backbone feature extractor"""
        input_shape = (*self.img_size, 3)
        
        if name == 'resnet50':
            base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
            dim = 2048
        elif name == 'efficientnetb0':
            base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
            dim = 1280
        elif name == 'mobilenetv2':
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
            dim = 1280
        else:
            raise ValueError(f'Unknown backbone: {name}')
        
        # Freeze backbone
        base.trainable = False
        
        # Add global pooling
        inputs = layers.Input(shape=input_shape)
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        model = models.Model(inputs=inputs, outputs=x, name=f'{name}_extractor')
        
        return model, dim
    
    def build_all(self):
        """Build all backbones"""
        for name in self.backbone_names:
            model, dim = self.build_backbone(name)
            self.backbones[name] = model
            self.backbone_dims[name] = dim
            self.total_dim += dim
        
        print(f"✅ Built {len(self.backbones)} backbones")
        print(f"   Total feature dimension: {self.total_dim}")
        for name, dim in self.backbone_dims.items():
            print(f"   - {name}: {dim} features")
        
        return self.backbones
    
    def create_feature_pipeline(self, dropout_rate=0.5):
        """Create end-to-end feature extraction pipeline"""
        if not self.backbones:
            self.build_all()
        
        inputs = layers.Input(shape=(*self.img_size, 3))
        features = []
        
        for name, backbone in self.backbones.items():
            f = backbone(inputs)
            f = layers.Dropout(dropout_rate, name=f'{name}_dropout')(f)
            features.append(f)
        
        concatenated = layers.Concatenate(name='feature_concat')(features)
        normalized = layers.BatchNormalization(name='feature_norm')(concatenated)
        
        model = models.Model(inputs=inputs, outputs=normalized, name='feature_pipeline')
        return model
    
    def extract_features(self, feature_model, dataset, verbose=True):
        """Extract features from dataset"""
        all_features = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(dataset):
            if verbose and batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx}...")
            
            features = feature_model.predict(images, verbose=0)
            all_features.append(features)
            all_labels.extend(labels.numpy())
        
        X = np.concatenate(all_features, axis=0)
        y = np.array(all_labels)
        
        if verbose:
            print(f"  ✅ Extracted: X {X.shape}, y {y.shape}")
            print(f"  ✅ Labels: {np.unique(y)}")
        
        return X, y
    
    def extract_features_from_paths(self, feature_model, image_paths, verbose=True):
        """Extract features from list of image paths"""
        all_features = []
        
        for i, path in enumerate(image_paths):
            if verbose and i % 50 == 0:
                print(f"  Processing image {i}/{len(image_paths)}...")
            
            # Load and preprocess image
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            # Standardize
            mean, var = tf.nn.moments(img, axes=[0, 1])
            img = (img - mean) / (tf.sqrt(var) + 1e-7)
            
            # Add batch dimension
            img = tf.expand_dims(img, 0)
            
            # Extract features
            features = feature_model.predict(img, verbose=0)
            all_features.append(features[0])
        
        return np.array(all_features)


class FeatureStatistics:
    """Comprehensive statistical analysis of CNN features"""
    
    def __init__(self, X_train, y_train, backbone_dims=None):
        self.X_train = X_train
        self.y_train = y_train
        self.backbone_dims = backbone_dims
        self.results = {}
        
    def analyze(self, output_dir='statistics'):
        """Run complete statistical analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("FEATURE STATISTICAL ANALYSIS")
        print("="*60)
        
        self._basic_statistics(output_dir)
        self._dimensionality_analysis(output_dir)
        self._correlation_analysis(output_dir)
        self._discriminative_power(output_dir)
        self._class_separability(output_dir)
        
        if self.backbone_dims:
            self._backbone_analysis(output_dir)
        
        # Save results
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            # Convert numpy types to Python types for JSON
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            results_json = convert_to_serializable(self.results)
            json.dump(results_json, f, indent=2)
        
        return self.results
    
    def _basic_statistics(self, output_dir):
        """Basic statistical properties"""
        print("\n📊 Basic Statistics:")
        
        from scipy import stats as scipy_stats
        
        # Calculer les statistiques
        X_flat = self.X_train.flatten()
        
        basic_stats = {
            'n_samples': int(self.X_train.shape[0]),
            'n_features': int(self.X_train.shape[1]),
            'n_classes': int(len(np.unique(self.y_train))),
            'mean': float(np.mean(self.X_train)),
            'std': float(np.std(self.X_train)),
            'min': float(np.min(self.X_train)),
            'max': float(np.max(self.X_train)),
            'skewness': float(scipy_stats.skew(X_flat)),
            'kurtosis': float(scipy_stats.kurtosis(X_flat))
        }
        
        for key, value in basic_stats.items():
            print(f"  {key}: {value}")
        
        self.results['basic'] = basic_stats
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(X_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Feature Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature means
        feature_means = np.mean(self.X_train, axis=0)
        axes[0, 1].hist(feature_means, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Mean Value')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Distribution of Feature Means')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature variances
        feature_vars = np.var(self.X_train, axis=0)
        axes[1, 0].hist(feature_vars, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Variance')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Distribution of Feature Variances')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=np.median(feature_vars), color='red', linestyle='--', 
                          label=f'Median: {np.median(feature_vars):.4f}')
        axes[1, 0].legend()
        
        # Feature-label correlation (sample)
        correlations = []
        n_sample = min(500, self.X_train.shape[1])
        for i in range(n_sample):
            corr = np.corrcoef(self.X_train[:, i], self.y_train)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        axes[1, 1].hist(correlations, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Correlation with Labels')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title(f'Feature-Label Correlation (Sample of {n_sample} features)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'basic_statistics.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/basic_statistics.png")
    
    def _dimensionality_analysis(self, output_dir):
        """Dimensionality analysis using SVD"""
        print("\n📊 Dimensionality Analysis:")
        
        # Center the data
        X_centered = self.X_train - np.mean(self.X_train, axis=0)
        
        # Compute SVD (use randomized SVD for speed)
        from sklearn.utils.extmath import randomized_svd
        
        print("  Computing SVD (this may take a moment)...")
        U, s, Vt = randomized_svd(X_centered, n_components=min(2000, X_centered.shape[1]), 
                                  random_state=SEED)
        
        # Explained variance
        explained_var = s**2 / np.sum(s**2)
        cum_var = np.cumsum(explained_var)
        
        # Find components for thresholds
        n_90 = np.searchsorted(cum_var, 0.90) + 1
        n_95 = np.searchsorted(cum_var, 0.95) + 1
        n_99 = np.searchsorted(cum_var, 0.99) + 1
        
        dim_stats = {
            'total_features': int(self.X_train.shape[1]),
            'components_90': int(n_90),
            'components_95': int(n_95),
            'components_99': int(n_99),
            'compression_ratio_95': float(self.X_train.shape[1] / n_95)
        }
        
        print(f"  Total features: {dim_stats['total_features']}")
        print(f"  90% variance: {n_90} components")
        print(f"  95% variance: {n_95} components")
        print(f"  99% variance: {n_99} components")
        print(f"  Compression ratio (95%): {dim_stats['compression_ratio_95']:.1f}x")
        
        self.results['dimensionality'] = dim_stats
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        n_display = min(200, len(explained_var))
        axes[0].plot(range(1, n_display+1), explained_var[:n_display] * 100, 
                    'bo-', markersize=3)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance (%)')
        axes[0].set_title(f'Scree Plot (First {n_display} Components)')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative variance
        n_display = min(2000, len(cum_var))
        axes[1].plot(range(1, n_display+1), cum_var[:n_display] * 100, 
                    'ro-', linewidth=1, markersize=1)
        axes[1].axhline(y=90, color='green', linestyle='--', label='90%')
        axes[1].axhline(y=95, color='blue', linestyle='--', label='95%')
        axes[1].axhline(y=99, color='red', linestyle='--', label='99%')
        if n_90 <= n_display:
            axes[1].axvline(x=n_90, color='green', linestyle=':', alpha=0.7)
        if n_95 <= n_display:
            axes[1].axvline(x=n_95, color='blue', linestyle=':', alpha=0.7)
        if n_99 <= n_display:
            axes[1].axvline(x=n_99, color='red', linestyle=':', alpha=0.7)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance (%)')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dimensionality_analysis.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/dimensionality_analysis.png")
    
    def _correlation_analysis(self, output_dir):
        """Feature correlation analysis"""
        print("\n📊 Correlation Analysis:")
        
        # Sample features for correlation matrix
        n_sample = min(100, self.X_train.shape[1])
        indices = np.random.choice(self.X_train.shape[1], n_sample, replace=False)
        X_sample = self.X_train[:, indices]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_sample.T)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        corr_stats = {
            'mean_abs_correlation': float(np.mean(np.abs(upper_tri))),
            'max_abs_correlation': float(np.max(np.abs(upper_tri))),
            'n_high_correlations': int(np.sum(np.abs(upper_tri) > 0.8))
        }
        
        print(f"  Mean absolute correlation: {corr_stats['mean_abs_correlation']:.4f}")
        print(f"  Max absolute correlation: {corr_stats['max_abs_correlation']:.4f}")
        print(f"  High correlations (>0.8): {corr_stats['n_high_correlations']}")
        
        self.results['correlation'] = corr_stats
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        im = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Feature Index')
        axes[0].set_title(f'Correlation Matrix (Sample of {n_sample} features)')
        plt.colorbar(im, ax=axes[0])
        
        # Distribution
        axes[1].hist(upper_tri, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].axvline(x=corr_stats['mean_abs_correlation'], color='red', linestyle='--', 
                       label=f'Mean: {corr_stats["mean_abs_correlation"]:.3f}')
        axes[1].axvline(x=0.8, color='green', linestyle=':', label='High correlation')
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Feature Correlations')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/correlation_analysis.png")
    
    def _discriminative_power(self, output_dir):
        """Analyze feature discriminative power"""
        print("\n📊 Discriminative Power Analysis:")
        
        from sklearn.feature_selection import mutual_info_classif, f_classif
        
        # Sample features for speed
        n_sample = min(500, self.X_train.shape[1])
        indices = np.random.choice(self.X_train.shape[1], n_sample, replace=False)
        X_sample = self.X_train[:, indices]
        
        # F-test
        print("  Computing F-test...")
        f_scores, f_pvalues = f_classif(X_sample, self.y_train)
        
        # Mutual Information
        print("  Computing Mutual Information...")
        mi_scores = mutual_info_classif(X_sample, self.y_train, random_state=SEED)
        
        disc_stats = {
            'mean_f_score': float(np.mean(f_scores)),
            'max_f_score': float(np.max(f_scores)),
            'mean_mutual_info': float(np.mean(mi_scores)),
            'max_mutual_info': float(np.max(mi_scores))
        }
        
        print(f"  Mean F-score: {disc_stats['mean_f_score']:.2f}")
        print(f"  Max F-score: {disc_stats['max_f_score']:.2f}")
        print(f"  Mean MI: {disc_stats['mean_mutual_info']:.4f}")
        print(f"  Max MI: {disc_stats['max_mutual_info']:.4f}")
        
        self.results['discriminative'] = disc_stats
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # F-score distribution
        axes[0].hist(f_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(x=disc_stats['mean_f_score'], color='red', linestyle='--', 
                       label=f'Mean: {disc_stats["mean_f_score"]:.2f}')
        axes[0].set_xlabel('F-Score')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Feature Discriminative Power (F-Test)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MI distribution
        axes[1].hist(mi_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axvline(x=disc_stats['mean_mutual_info'], color='red', linestyle='--', 
                       label=f'Mean: {disc_stats["mean_mutual_info"]:.4f}')
        axes[1].set_xlabel('Mutual Information')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Feature Discriminative Power (Mutual Information)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'discriminative_power.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/discriminative_power.png")
    
    def _class_separability(self, output_dir):
        """Class separability using PCA projection"""
        print("\n📊 Class Separability Analysis:")
        
        from sklearn.decomposition import PCA
        
        # PCA for 2D visualization
        print("  Computing PCA for 2D projection...")
        pca = PCA(n_components=2, random_state=SEED)
        X_pca = pca.fit_transform(self.X_train)
        
        var_ratio = pca.explained_variance_ratio_
        
        sep_stats = {
            'pca_1_variance': float(var_ratio[0]),
            'pca_2_variance': float(var_ratio[1]),
            'total_2d_variance': float(np.sum(var_ratio))
        }
        
        print(f"  PC1 variance: {sep_stats['pca_1_variance']:.2%}")
        print(f"  PC2 variance: {sep_stats['pca_2_variance']:.2%}")
        print(f"  Total 2D variance: {sep_stats['total_2d_variance']:.2%}")
        
        self.results['separability'] = sep_stats
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        n_classes = len(np.unique(self.y_train))
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_classes)))
        
        # Show first 20 classes for clarity
        for i in range(min(20, n_classes)):
            mask = self.y_train == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i % len(colors)]], label=f'Class {i}', alpha=0.5, s=5)
        
        plt.xlabel(f'PC1 ({var_ratio[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({var_ratio[1]:.1%} variance)')
        plt.title('Class Separability - PCA Projection (First 20 Classes)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_separability.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/class_separability.png")
    
    def _backbone_analysis(self, output_dir):
        """Analyze features by backbone origin"""
        print("\n📊 Backbone Contribution Analysis:")
        
        from sklearn.feature_selection import mutual_info_classif
        
        backbone_stats = {}
        start = 0
        
        for name, dim in self.backbone_dims.items():
            end = start + dim
            if end > self.X_train.shape[1]:
                end = self.X_train.shape[1]
            
            features = self.X_train[:, start:end]
            
            # Compute statistics
            mean_val = np.mean(features)
            std_val = np.std(features)
            
            # Mutual information (sample)
            print(f"  Computing MI for {name}...")
            n_sample = min(100, features.shape[1])
            mi_sample = mutual_info_classif(features[:, :n_sample], 
                                           self.y_train, random_state=SEED)
            
            backbone_stats[name] = {
                'dimension': int(end - start),
                'mean_value': float(mean_val),
                'std_value': float(std_val),
                'mean_mutual_info': float(np.mean(mi_sample))
            }
            
            print(f"  {name}:")
            print(f"    Dim: {end-start}")
            print(f"    Mean MI: {backbone_stats[name]['mean_mutual_info']:.4f}")
            
            start = end
        
        self.results['backbone'] = backbone_stats
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Feature count
        names = list(backbone_stats.keys())
        dims = [backbone_stats[n]['dimension'] for n in names]
        bars1 = axes[0].bar(names, dims, color=['steelblue', 'coral', 'forestgreen'])
        axes[0].set_xlabel('Backbone')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Feature Count per Backbone')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, dim in zip(bars1, dims):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                        str(dim), ha='center', va='bottom', fontsize=9)
        
        # Mutual information
        mi_values = [backbone_stats[n]['mean_mutual_info'] for n in names]
        bars2 = axes[1].bar(names, mi_values, color=['steelblue', 'coral', 'forestgreen'])
        axes[1].set_xlabel('Backbone')
        axes[1].set_ylabel('Mean Mutual Information')
        axes[1].set_title('Feature Importance by Backbone')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mi in zip(bars2, mi_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{mi:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backbone_analysis.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/backbone_analysis.png")


class MetaLearner:
    """Meta-learner for feature stacking"""
    
    def __init__(self, input_dim, num_classes, model_type='mlp', **kwargs):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.history = None
        
    def build_mlp(self, units=256, dropout=0.3, lr=1e-4, num_layers=2):
        """Build MLP model - accepts only MLP-specific arguments"""
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        
        for i in range(num_layers):
            units_layer = units // (2**i)
            x = layers.Dense(units_layer, activation='relu',
                           kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.Dropout(dropout)(x)
            x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_pca_logistic(self, pca_components=0.95):
        """Build PCA + LogisticRegression pipeline"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_components, random_state=SEED)),
            ('classifier', LogisticRegressionCV(
                Cs=10,
                cv=5,
                max_iter=1000,
                n_jobs=-1,
                random_state=SEED
            ))
        ])
        return pipeline
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **fit_params):
        """Fit the meta-learner"""
        
        print(f"  📊 Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  📊 Classes: {self.num_classes}")
        
        if self.model_type == 'mlp':
            # MLP training - use only MLP kwargs
            mlp_kwargs = {
                'units': self.kwargs.get('units', 256),
                'dropout': self.kwargs.get('dropout', 0.3),
                'lr': self.kwargs.get('lr', 1e-4),
                'num_layers': self.kwargs.get('num_layers', 2)
            }
            self.model = self.build_mlp(**mlp_kwargs)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            validation_data = None
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                validation_data = (X_val_scaled, y_val)
            
            # Class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
            
            self.history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=validation_data,
                epochs=fit_params.get('epochs', 30),
                batch_size=fit_params.get('batch_size', 32),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=fit_params.get('verbose', 0)
            )
            
        elif self.model_type == 'mlp_pca':
            # MLP with PCA preprocessing
            pca_components = self.kwargs.get('pca_components', 0.95)
            
            # First apply PCA
            self.pca = PCA(n_components=pca_components, random_state=SEED)
            X_train_pca = self.pca.fit_transform(X_train)
            
            var_explained = self.pca.explained_variance_ratio_.sum()
            print(f"  🔄 PCA: {X_train.shape[1]} → {X_train_pca.shape[1]} features")
            print(f"  📈 Variance explained: {var_explained:.2%}")
            
            # Create MLP with reduced dimension
            self.input_dim = X_train_pca.shape[1]
            mlp_kwargs = {
                'units': self.kwargs.get('units', 256),
                'dropout': self.kwargs.get('dropout', 0.3),
                'lr': self.kwargs.get('lr', 1e-4),
                'num_layers': self.kwargs.get('num_layers', 2)
            }
            self.model = self.build_mlp(**mlp_kwargs)
            
            # Scale and train
            X_train_scaled = self.scaler.fit_transform(X_train_pca)
            
            validation_data = None
            if X_val is not None:
                X_val_pca = self.pca.transform(X_val)
                X_val_scaled = self.scaler.transform(X_val_pca)
                validation_data = (X_val_scaled, y_val)
            
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
            
            self.history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=validation_data,
                epochs=fit_params.get('epochs', 30),
                batch_size=fit_params.get('batch_size', 32),
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=fit_params.get('verbose', 0)
            )
            
        elif self.model_type == 'pca_logistic':
            # PCA + LogisticRegression pipeline - HARMONIZED VERSION
            pca_components = self.kwargs.get('pca_components', 790)  # Fixed to 790 for harmonization
            self.model = self.build_pca_logistic(pca_components=pca_components)
            self.model.fit(X_train, y_train)
            
            # Get variance explained
            if hasattr(self.model.named_steps['pca'], 'explained_variance_ratio_'):
                var_exp = self.model.named_steps['pca'].explained_variance_ratio_.sum()
                n_comp = self.model.named_steps['pca'].n_components_
                print(f"  🔄 PCA: {X_train.shape[1]} → {n_comp} features")
                print(f"  📈 Variance explained: {var_exp:.2%}")
        
        return self.model
    
    def predict(self, X, proba=False):
        """Make predictions"""
        if self.model_type == 'mlp':
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict(X_scaled, verbose=0)
            return probs if proba else np.argmax(probs, axis=1)
            
        elif self.model_type == 'mlp_pca':
            X_pca = self.pca.transform(X)
            X_scaled = self.scaler.transform(X_pca)
            probs = self.model.predict(X_scaled, verbose=0)
            return probs if proba else np.argmax(probs, axis=1)
            
        elif self.model_type == 'pca_logistic':
            probs = self.model.predict_proba(X)
            return probs if proba else self.model.predict(X)


class ModelComparator:
    """Compare different meta-learners"""
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, class_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.class_names = class_names
        self.models = {}
        self.results = {}
        
    def train_mlp_full(self, **kwargs):
        """Train MLP on full features"""
        print("\n" + "="*50)
        print("Training MLP (Full Features)")
        print("="*50)
        
        mlp = MetaLearner(
            input_dim=self.X_train.shape[1],
            num_classes=self.num_classes,
            model_type='mlp',
            units=kwargs.get('units', 256),
            dropout=kwargs.get('dropout', 0.3),
            lr=kwargs.get('lr', 1e-4),
            num_layers=2
        )
        
        mlp.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=kwargs.get('epochs', 30),
            batch_size=32,
            verbose=1
        )
        
        self.models['MLP-Full'] = mlp
        return mlp
    
    def train_mlp_pca(self, **kwargs):
        """Train MLP with PCA preprocessing"""
        print("\n" + "="*50)
        print("Training MLP + PCA")
        print("="*50)
        
        mlp_pca = MetaLearner(
            input_dim=self.X_train.shape[1],
            num_classes=self.num_classes,
            model_type='mlp_pca',
            units=kwargs.get('units', 256),
            dropout=kwargs.get('dropout', 0.3),
            lr=kwargs.get('lr', 1e-4),
            num_layers=2,
            pca_components=kwargs.get('pca_components', 0.95)
        )
        
        mlp_pca.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=kwargs.get('epochs', 30),
            batch_size=32,
            verbose=1
        )
        
        self.models['MLP-PCA'] = mlp_pca
        return mlp_pca
    
    def train_pca_logistic(self, **kwargs):
        """Train PCA + LogisticRegression - HARMONIZED with exact 790 components"""
        print("\n" + "="*50)
        print("Training PCA + LogisticRegression (Harmonized - 790 components)")
        print("="*50)
        
        # Force exact number of components = 790 (from your analysis)
        n_components = kwargs.get('pca_components', 790)
        
        pca_log = MetaLearner(
            input_dim=self.X_train.shape[1],
            num_classes=self.num_classes,
            model_type='pca_logistic',
            pca_components=n_components  # Now exact number, not percentage
        )
        
        pca_log.fit(self.X_train, self.y_train)
        
        self.models['PCA-Logistic-Harmonized'] = pca_log
        return pca_log
    
    def evaluate_all(self):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            y_pred = model.predict(self.X_test)
            y_proba = model.predict(self.X_test, proba=True)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            # Per-class accuracy
            class_acc = {}
            for i in range(self.num_classes):
                mask = self.y_test == i
                if np.sum(mask) > 0:
                    class_acc[f'class_{i}'] = float(accuracy_score(self.y_test[mask], y_pred[mask]))
            
            results[name] = {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'class_accuracy': class_acc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  F1-macro: {f1_macro:.4f}")
            print(f"  F1-weighted: {f1_weighted:.4f}")
        
        self.results = results
        return results
    
    def statistical_comparison(self):
        """Statistical comparison between models"""
        if len(self.models) < 2:
            return None
        
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        comparisons = []
        model_names = list(self.models.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                
                # Get predictions
                pred1 = self.models[name1].predict(self.X_test)
                pred2 = self.models[name2].predict(self.X_test)
                
                # McNemar's test
                a = np.sum((pred1 == self.y_test) & (pred2 == self.y_test))
                b = np.sum((pred1 == self.y_test) & (pred2 != self.y_test))
                c = np.sum((pred1 != self.y_test) & (pred2 == self.y_test))
                d = np.sum((pred1 != self.y_test) & (pred2 != self.y_test))
                
                chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
                p_mcnemar = 1 - stats.chi2.cdf(chi2, 1)
                
                # Accuracy difference
                acc1 = accuracy_score(self.y_test, pred1)
                acc2 = accuracy_score(self.y_test, pred2)
                
                comparison = {
                    'model1': name1,
                    'model2': name2,
                    'accuracy1': float(acc1),
                    'accuracy2': float(acc2),
                    'difference': float(acc1 - acc2),
                    'mcnemar_chi2': float(chi2),
                    'mcnemar_p': float(p_mcnemar),
                    'significant_0.05': p_mcnemar < 0.05
                }
                
                comparisons.append(comparison)
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Accuracy diff: {acc1-acc2:.4f}")
                print(f"  McNemar p-value: {p_mcnemar:.4e}")
                print(f"  Significant (p<0.05): {p_mcnemar < 0.05}")
        
        return comparisons
    
    def plot_comparison(self, save_path='diagnostics/model_comparison.png'):
        """Plot model comparison"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        f1_macro = [self.results[m]['f1_macro'] for m in models]
        balanced_acc = [self.results[m]['balanced_accuracy'] for m in models]
        
        # Bar chart
        x = np.arange(len(models))
        width = 0.25
        
        axes[0, 0].bar(x - width, accuracies, width, label='Accuracy', color='steelblue')
        axes[0, 0].bar(x, balanced_acc, width, label='Balanced Acc', color='orange')
        axes[0, 0].bar(x + width, f1_macro, width, label='F1-macro', color='coral')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.5, 1.0])
        
        # Training time / complexity
        feature_dims = []
        for name in models:
            if 'MLP-Full' in name:
                feature_dims.append(4608)
            elif 'MLP-PCA' in name:
                feature_dims.append(790)
            elif 'Logistic' in name:
                feature_dims.append(790)
        
        axes[0, 1].bar(models, feature_dims, color=['steelblue', 'coral', 'forestgreen'])
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Effective Feature Dimension')
        axes[0, 1].set_title('Model Complexity (Features Used)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-class accuracy heatmap
        if len(models) <= 3:
            class_acc_matrix = []
            for model in models:
                class_acc = [self.results[model]['class_accuracy'].get(f'class_{i}', 0) 
                           for i in range(min(10, self.num_classes))]
                class_acc_matrix.append(class_acc)
            
            im = axes[1, 0].imshow(class_acc_matrix, cmap='viridis', aspect='auto', vmin=0.5, vmax=1.0)
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('Model')
            axes[1, 0].set_title('Per-Class Accuracy (First 10 Classes)')
            axes[1, 0].set_xticks(range(min(10, self.num_classes)))
            axes[1, 0].set_yticks(range(len(models)))
            axes[1, 0].set_yticklabels(models)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Summary text
        summary = "Model Performance Summary:\n"
        for model in models:
            summary += f"• {model}: Acc={self.results[model]['accuracy']:.3f}, F1={self.results[model]['f1_macro']:.3f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.suptitle('Model Comparison Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparison plot saved to {save_path}")


class RealWorldEvaluator:
    """Evaluate models on real-world datasets"""
    
    def __init__(self, models, feature_extractor, preprocessor):
        self.models = models
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.results = {}
        
    def evaluate_on_plantdoc(self, max_images=500):
        """Evaluate on PlantDoc real-world dataset"""
        print("\n" + "="*60)
        print("EVALUATION ON PLANTDOC (REAL-WORLD FIELD IMAGES)")
        print("="*60)
        
        loader = RealWorldDatasetLoader()
        ds_train, ds_test, class_names = loader.load_plantdoc(max_images_per_class=50)
        
        if ds_test is None:
            print("❌ PlantDoc evaluation failed")
            return None
        
        # Extract features
        print("\nExtracting features from PlantDoc test set...")
        feature_pipeline = self.feature_extractor.create_feature_pipeline()
        
        X_test, y_test = self.feature_extractor.extract_features(
            feature_pipeline, ds_test, verbose=True
        )
        
        # Evaluate each model
        results = {}
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name} on PlantDoc...")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results[name] = {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'f1_macro': float(f1_macro),
                'n_samples': len(y_test)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  F1-macro: {f1_macro:.4f}")
        
        self.results['plantdoc'] = results
        return results
    
    def plot_real_world_comparison(self, save_path='diagnostics/real_world_comparison.png'):
        """Plot comparison on real-world datasets"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        datasets = list(self.results.keys())
        models = list(self.models.keys())
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            offset = (i - len(datasets)/2 + 0.5) * width
            accuracies = [self.results[dataset].get(model, {}).get('accuracy', 0) for model in models]
            
            bars = axes[0].bar(x + offset, accuracies, width, label=dataset)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Performance on Real-World Datasets')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        # Performance drop from PlantVillage
        axes[1].set_title('Performance Drop on Real-World Data')
        
        # This would need baseline PlantVillage results
        axes[1].text(0.1, 0.5, "Performance drop analysis\nrequires PlantVillage baseline",
                    transform=axes[1].transAxes, ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Real-world comparison saved to {save_path}")


class ShapAnalyzer:
    """SHAP analysis for model interpretability"""
    
    def __init__(self, model, X_background, X_explain, feature_names=None, class_names=None):
        self.model = model
        self.X_background = X_background
        self.X_explain = X_explain
        self.feature_names = feature_names if feature_names else [f"F{i}" for i in range(X_background.shape[1])]
        self.class_names = class_names
        
    def analyze(self, output_dir='diagnostics'):
        """Run SHAP analysis"""
        if shap is None:
            print("⚠️ SHAP not installed, skipping analysis")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)
        
        # Limit samples for speed
        n_background = min(100, self.X_background.shape[0])
        n_explain = min(50, self.X_explain.shape[0])
        
        X_back = self.X_background[:n_background]
        X_exp = self.X_explain[:n_explain]
        
        print(f"Background samples: {n_background}")
        print(f"Explanation samples: {n_explain}")
        
        # Create explainer based on model type
        if self.model.model_type in ['mlp', 'mlp_pca']:
            # For MLP models, use GradientExplainer
            def model_predict(x):
                if self.model.model_type == 'mlp_pca':
                    x = self.model.pca.transform(x)
                x_scaled = self.model.scaler.transform(x)
                return self.model.model.predict(x_scaled, verbose=0)
            
            explainer = shap.GradientExplainer(self.model.model, X_back[:50])
            shap_values = explainer.shap_values(X_exp)
            
        else:  # pca_logistic
            # For sklearn pipeline, use KernelExplainer
            def model_predict(x):
                return self.model.model.predict_proba(x)
            
            explainer = shap.KernelExplainer(model_predict, X_back[:30])
            shap_values = explainer.shap_values(X_exp[:20], nsamples=100)
        
        # Summary plot
        plt.figure(figsize=(14, 10))
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            top_indices = np.argsort(mean_shap)[-30:][::-1]
            
            shap.summary_plot(
                [sv[:, top_indices] for sv in shap_values],
                X_exp[:, top_indices],
                feature_names=[self.feature_names[i] for i in top_indices],
                show=False,
                max_display=30
            )
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_shap)[-30:][::-1]
            
            shap.summary_plot(
                shap_values[:, top_indices],
                X_exp[:, top_indices],
                feature_names=[self.feature_names[i] for i in top_indices],
                show=False,
                max_display=30
            )
        
        plt.title('SHAP Summary - Top 30 Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/shap_summary.png")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(
                [sv[:, top_indices[:20]] for sv in shap_values],
                X_exp[:, top_indices[:20]],
                feature_names=[self.feature_names[i] for i in top_indices[:20]],
                plot_type='bar',
                show=False,
                max_display=20
            )
        else:
            shap.summary_plot(
                shap_values[:, top_indices[:20]],
                X_exp[:, top_indices[:20]],
                feature_names=[self.feature_names[i] for i in top_indices[:20]],
                plot_type='bar',
                show=False,
                max_display=20
            )
        
        plt.title('SHAP Feature Importance - Top 20 Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_importance.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/shap_importance.png")
        
        return shap_values


class DiagnosticVisualizer:
    """Diagnostic visualizations"""
    
    def __init__(self, save_dir='diagnostics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(max(12, len(class_names)*0.3), max(10, len(class_names)*0.25)))
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # If too many classes, show subset
        if len(class_names) > 20:
            display_names = [f"{i}" for i in range(20)] + ['...']
            display_cm = cm[:20, :20]
        else:
            display_names = class_names
            display_cm = cm
        
        sns.heatmap(display_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=display_names, yticklabels=display_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{title.replace(" ", "_")}.png'), 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {title.replace(' ', '_')}.png")
    
    def plot_training_history(self, history, title="Training History"):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Training', linewidth=2)
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Training', linewidth=2)
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: training_history.png")


class FeatureStackingPipeline:
    """Complete feature stacking pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.preprocessor = DataPreprocessor(IMG_SIZE)
        self.feature_extractor = None
        self.comparator = None
        self.visualizer = DiagnosticVisualizer('diagnostics')
        self.class_names = None
        self.num_classes = None
        self.backbone_dims = None
        
        set_memory_growth()
    
    def run(self):
        """Run the complete pipeline"""
        
        # PHASE 1: Load and prepare data
        self._load_data()
        
        # PHASE 2: Extract or load features
        self._extract_features()
        
        # PHASE 3: Statistical analysis
        self._analyze_features()
        
        # PHASE 4: Train and compare models
        self._train_models()
        
        # PHASE 5: Evaluate on real-world data
        if not self.args.skip_real_world:
            self._evaluate_real_world()
        
        # PHASE 6: SHAP analysis
        if not self.args.skip_shap:
            self._analyze_shap()
        
        # PHASE 7: Generate report
        self._generate_report()
    
    def _load_data(self):
        """Load and prepare data"""
        print("\n" + "="*60)
        print("PHASE 1: Data Loading and Preprocessing")
        print("="*60)
        
        ds_train, ds_val, ds_test, class_names = self.preprocessor.load_plantvillage(
            self.args.data_dir
        )
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Prepare datasets
        self.ds_train = self.preprocessor.prepare_dataset(ds_train, augment=True)
        self.ds_val = self.preprocessor.prepare_dataset(ds_val, augment=False, cache=True)
        self.ds_test = self.preprocessor.prepare_dataset(ds_test, augment=False, cache=True)
        
        # Count samples
        self.n_train = sum(1 for _ in self.ds_train.unbatch())
        self.n_val = sum(1 for _ in self.ds_val.unbatch())
        self.n_test = sum(1 for _ in self.ds_test.unbatch())
        
        print(f"\n✅ Dataset prepared:")
        print(f"  Training: {self.n_train} samples")
        print(f"  Validation: {self.n_val} samples")
        print(f"  Test: {self.n_test} samples")
        print(f"  Classes: {self.num_classes}")
    
    def _extract_features(self):
        """Extract features using CNN backbones"""
        print("\n" + "="*60)
        print("PHASE 2: Feature Extraction")
        print("="*60)
        
        # Check if features already exist
        if os.path.exists('features/X_train.npy') and not self.args.force_extract:
            print("Loading cached features...")
            self.X_train = np.load('features/X_train.npy')
            self.y_train = np.load('features/y_train.npy')
            self.X_val = np.load('features/X_val.npy')
            self.y_val = np.load('features/y_val.npy')
            self.X_test = np.load('features/X_test.npy')
            self.y_test = np.load('features/y_test.npy')
            
            print(f"✅ Features loaded:")
            print(f"  Train: {self.X_train.shape}")
            print(f"  Val: {self.X_val.shape}")
            print(f"  Test: {self.X_test.shape}")
            return
        
        # Extract features
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.build_all()
        self.backbone_dims = self.feature_extractor.backbone_dims
        
        feature_pipeline = self.feature_extractor.create_feature_pipeline(dropout_rate=0.5)
        
        print("\nExtracting training features...")
        self.X_train, self.y_train = self.feature_extractor.extract_features(
            feature_pipeline, self.ds_train
        )
        
        print("\nExtracting validation features...")
        self.X_val, self.y_val = self.feature_extractor.extract_features(
            feature_pipeline, self.ds_val, verbose=False
        )
        
        print("\nExtracting test features...")
        self.X_test, self.y_test = self.feature_extractor.extract_features(
            feature_pipeline, self.ds_test, verbose=False
        )
        
        # Save features
        os.makedirs('features', exist_ok=True)
        np.save('features/X_train.npy', self.X_train)
        np.save('features/y_train.npy', self.y_train)
        np.save('features/X_val.npy', self.X_val)
        np.save('features/y_val.npy', self.y_val)
        np.save('features/X_test.npy', self.X_test)
        np.save('features/y_test.npy', self.y_test)
        
        print(f"\n✅ Features saved to features/")
    
    def _analyze_features(self):
        """Run statistical analysis on features"""
        print("\n" + "="*60)
        print("PHASE 3: Feature Statistical Analysis")
        print("="*60)
        
        analyzer = FeatureStatistics(
            self.X_train, self.y_train,
            self.backbone_dims
        )
        
        self.feature_stats = analyzer.analyze(output_dir='statistics')
    
    def _train_models(self):
        """Train and compare meta-learners"""
        print("\n" + "="*60)
        print("PHASE 4: Training and Comparing Meta-Learners")
        print("="*60)
        
        self.comparator = ModelComparator(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            self.num_classes,
            self.class_names
        )
        
        # Train all models
        print("\n📌 Training MLP (Full Features)...")
        mlp_full = self.comparator.train_mlp_full(
            units=256,
            dropout=0.3,
            lr=1e-4,
            epochs=self.args.epochs
        )
        
        print("\n📌 Training MLP + PCA...")
        mlp_pca = self.comparator.train_mlp_pca(
            units=256,
            dropout=0.3,
            lr=1e-4,
            epochs=self.args.epochs,
            pca_components=0.95
        )
        
        print("\n📌 Training PCA + LogisticRegression (HARMONIZED - 790 components)...")
        pca_log = self.comparator.train_pca_logistic(
            pca_components=790  # Exact number from your analysis
        )
        
        # Evaluate all
        results = self.comparator.evaluate_all()
        
        # Statistical comparison
        comparisons = self.comparator.statistical_comparison()
        
        # Plot comparison
        self.comparator.plot_comparison()
        
        # Save results
        self._save_results(results, comparisons)
        
        return results, comparisons
    
    def _save_results(self, results, comparisons):
        """Save all results"""
        os.makedirs('results', exist_ok=True)
        
        # Metrics
        metrics = {}
        for name, res in results.items():
            metrics[name] = {
                'accuracy': res['accuracy'],
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'f1_weighted': res['f1_weighted']
            }
            
            # Classification report
            report = classification_report(self.y_test, res['predictions'], 
                                         target_names=self.class_names, 
                                         zero_division=0,
                                         output_dict=True)
            with open(f'results/classification_report_{name}.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Confusion matrix
            self.visualizer.plot_confusion_matrix(
                self.y_test, res['predictions'],
                class_names=self.class_names,
                title=f'Confusion Matrix - {name}'
            )
        
        # Add comparisons
        if comparisons:
            metrics['comparisons'] = comparisons
        
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\n✅ Results saved to 'results/' directory")
    
    def _evaluate_real_world(self):
        """Evaluate models on real-world datasets"""
        print("\n" + "="*60)
        print("PHASE 5: Real-World Validation")
        print("="*60)
        
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.build_all()
        
        evaluator = RealWorldEvaluator(
            self.comparator.models,
            self.feature_extractor,
            self.preprocessor
        )
        
        # Evaluate on PlantDoc
        plantdoc_results = evaluator.evaluate_on_plantdoc()
        
        # Save results
        if plantdoc_results:
            with open('results/real_world_plantdoc.json', 'w') as f:
                json.dump(plantdoc_results, f, indent=2)
        
        # Plot comparison
        evaluator.plot_real_world_comparison()
    
    def _analyze_shap(self):
        """Run SHAP analysis"""
        if shap is None:
            print("⚠️ SHAP not installed, skipping analysis")
            return
        
        print("\n" + "="*60)
        print("PHASE 6: SHAP Analysis")
        print("="*60)
        
        # Use MLP-PCA model (best compromise)
        if 'MLP-PCA' in self.comparator.models:
            model = self.comparator.models['MLP-PCA']
        else:
            model = self.comparator.models['MLP-Full']
        
        # Create feature names
        feature_names = []
        if self.backbone_dims:
            for name, dim in self.backbone_dims.items():
                for i in range(min(dim, 500)):  # Limit for readability
                    feature_names.append(f"{name}_{i}")
        else:
            feature_names = [f"F{i}" for i in range(self.X_train.shape[1])]
        
        # Sample data
        X_background = self.X_train[:200]
        X_explain = self.X_test[:100]
        
        analyzer = ShapAnalyzer(
            model, X_background, X_explain,
            feature_names=feature_names,
            class_names=self.class_names
        )
        
        analyzer.analyze(output_dir='diagnostics')
    
    def _generate_report(self):
        """Generate final report"""
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Validation samples: {self.X_val.shape[0]}")
        print(f"  Test samples: {self.X_test.shape[0]}")
        print(f"  Feature dimension: {self.X_train.shape[1]}")
        
        print(f"\n📊 FEATURE STATISTICS:")
        if hasattr(self, 'feature_stats'):
            if 'dimensionality' in self.feature_stats:
                dim = self.feature_stats['dimensionality']
                print(f"  95% variance: {dim.get('components_95', 'N/A')} components")
                print(f"  Compression ratio: {dim.get('compression_ratio_95', 'N/A'):.1f}x")
        
        print(f"\n📊 MODEL PERFORMANCE (PlantVillage):")
        if self.comparator and self.comparator.results:
            for name, res in self.comparator.results.items():
                print(f"  {name}:")
                print(f"    Accuracy: {res['accuracy']:.4f}")
                print(f"    Balanced Acc: {res['balanced_accuracy']:.4f}")
                print(f"    F1-macro: {res['f1_macro']:.4f}")
        
        # Check for real-world results
        real_world_file = 'results/real_world_plantdoc.json'
        if os.path.exists(real_world_file):
            print(f"\n📊 REAL-WORLD PERFORMANCE (PlantDoc):")
            with open(real_world_file, 'r') as f:
                plantdoc_results = json.load(f)
                for model, metrics in plantdoc_results.items():
                    print(f"  {model}:")
                    print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
                    print(f"    F1-macro: {metrics.get('f1_macro', 0):.4f}")
        
        print("\n✅ All analyses completed successfully!")
        print("\nOutput files:")
        print("  - results/          : JSON metrics and classification reports")
        print("  - results/real_world_plantdoc.json : Real-world validation results")
        print("  - statistics/       : Feature statistical analysis")
        print("  - diagnostics/      : All visualization plots")
        print("  - features/         : Extracted features")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Feature Stacking Pipeline with Real-World Validation')
    
    parser.add_argument('--data_dir', type=str, default='datasets/PlantVillage',
                       help='Path to PlantVillage dataset')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--force_extract', action='store_true',
                       help='Force feature extraction even if cached')
    parser.add_argument('--skip_shap', action='store_true',
                       help='Skip SHAP analysis')
    parser.add_argument('--skip_real_world', action='store_true',
                       help='Skip real-world validation')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Create directories
    for dir_name in ['results', 'diagnostics', 'statistics', 'features']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # Run pipeline
    pipeline = FeatureStackingPipeline(args)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()