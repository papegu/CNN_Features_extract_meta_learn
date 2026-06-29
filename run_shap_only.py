"""
run_shap_only.py - CORRIGÉ V2

Script pour relancer uniquement l'analyse SHAP après avoir sauvegardé les résultats.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
SEED = 42
FIGURE_DPI = 300
plt.rc('font', family='Arial', size=10)

# Import SHAP
try:
    import shap
    print("✅ SHAP imported successfully")
except ImportError:
    shap = None
    print("❌ SHAP not installed. Install with: pip install shap")
    exit(1)

# Configuration TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')


class MetaLearner:
    """Simplified Meta-learner class for SHAP analysis"""
    
    def __init__(self, input_dim, num_classes, model_type='mlp', **kwargs):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def build_mlp_full(self):
        """Build MLP-Full model"""
        from tensorflow.keras import layers, models, optimizers
        
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        
        # Hidden layer 1
        x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Hidden layer 2
        x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_mlp_pca(self, n_components=814):
        """Build MLP-PCA model"""
        from tensorflow.keras import layers, models, optimizers
        
        inputs = layers.Input(shape=(n_components,))
        x = layers.BatchNormalization()(inputs)
        
        # Hidden layer 1
        x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Hidden layer 2
        x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def predict(self, X, proba=False):
        """Make predictions"""
        if self.model_type == 'mlp_full':
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict(X_scaled, verbose=0)
            return probs if proba else np.argmax(probs, axis=1)
        elif self.model_type == 'mlp_pca':
            X_pca = self.pca.transform(X)
            X_scaled = self.scaler.transform(X_pca)
            probs = self.model.predict(X_scaled, verbose=0)
            return probs if proba else np.argmax(probs, axis=1)
        else:
            return None


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
        if self.model.model_type == 'mlp_pca':
            # For MLP-PCA model
            def model_predict(x):
                x_pca = self.model.pca.transform(x)
                x_scaled = self.model.scaler.transform(x_pca)
                return self.model.model.predict(x_scaled, verbose=0)
            
            # Use a subset for background
            background_subset = X_back[:50]
            explainer = shap.GradientExplainer(self.model.model, background_subset)
            shap_values = explainer.shap_values(X_exp)
            
        else:  # mlp_full
            # For MLP-Full model
            def model_predict(x):
                x_scaled = self.model.scaler.transform(x)
                return self.model.model.predict(x_scaled, verbose=0)
            
            # Use a subset for background
            background_subset = X_back[:50]
            explainer = shap.GradientExplainer(self.model.model, background_subset)
            shap_values = explainer.shap_values(X_exp)
        
        # Convertir top_indices en liste d'entiers Python de façon robuste
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            # Obtenir les indices et les convertir en entiers
            top_indices = np.argsort(mean_shap)[-30:][::-1]
            top_indices = top_indices.flatten().tolist()  # Conversion robuste
            top_indices = [int(i) for i in top_indices]   # Conversion en int Python
            
            # Summary plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(
                [sv[:, top_indices] for sv in shap_values],
                X_exp[:, top_indices],
                feature_names=[self.feature_names[i] for i in top_indices],
                show=False,
                max_display=30
            )
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            # Obtenir les indices et les convertir en entiers
            top_indices = np.argsort(mean_shap)[-30:][::-1]
            top_indices = top_indices.flatten().tolist()  # Conversion robuste
            top_indices = [int(i) for i in top_indices]   # Conversion en int Python
            
            # Summary plot
            plt.figure(figsize=(14, 10))
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
        
        # Bar plot - Top 20 features
        top_20_indices = top_indices[:20]
        
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(
                [sv[:, top_20_indices] for sv in shap_values],
                X_exp[:, top_20_indices],
                feature_names=[self.feature_names[i] for i in top_20_indices],
                plot_type='bar',
                show=False,
                max_display=20
            )
        else:
            shap.summary_plot(
                shap_values[:, top_20_indices],
                X_exp[:, top_20_indices],
                feature_names=[self.feature_names[i] for i in top_20_indices],
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

class SaliencyAnalyzer:
    """Gradient-based saliency analysis for MLP models"""
    
    def __init__(self, model, X_explain, feature_names=None):
        self.model = model
        self.X_explain = X_explain
        self.feature_names = feature_names if feature_names else [f"F{i}" for i in range(X_explain.shape[1])]
    
    def compute_saliency(self, X):
        """Compute gradient-based saliency"""
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tf)
            
            if self.model.model_type == 'mlp_full':
                X_scaled = self.model.scaler.transform(X)
            else:
                X_pca = self.model.pca.transform(X)
                X_scaled = self.model.scaler.transform(X_pca)
            
            X_scaled_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            preds = self.model.model(X_scaled_tf)
            
            # Prendre la classe prédite
            pred_class = tf.argmax(preds, axis=1)
            selected = tf.reduce_sum(
                preds * tf.one_hot(pred_class, depth=preds.shape[1]),
                axis=1
            )
        
        grads = tape.gradient(selected, X_tf)
        saliency = tf.abs(grads).numpy()
        
        return saliency
    
    def analyze(self, output_dir='diagnostics'):
        """Run saliency analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("SALIENCY ANALYSIS")
        print("="*60)
        
        # Limiter pour vitesse
        X_exp = self.X_explain[:50]
        
        saliency = self.compute_saliency(X_exp)
        
        # Importance moyenne
        importance = np.mean(saliency, axis=0)
        
        # Top features
        top_indices = np.argsort(importance)[-30:][::-1]
        
        # Plot importance
        plt.figure(figsize=(12, 8))
        plt.barh(
            range(20),
            importance[top_indices[:20]][::-1]
        )
        plt.yticks(
            range(20),
            [self.feature_names[i] for i in top_indices[:20]][::-1]
        )
        plt.title('Saliency Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'saliency_importance.png'), dpi=300)
        plt.close()
        
        print(f"✅ Saved: {output_dir}/saliency_importance.png")
        
        return saliency
def load_features():
    """Load cached features"""
    print("\n" + "="*60)
    print("Loading cached features...")
    print("="*60)
    
    X_train = np.load('features/X_train.npy')
    y_train = np.load('features/y_train.npy')
    X_val = np.load('features/X_val.npy')
    y_val = np.load('features/y_val.npy')
    X_test = np.load('features/X_test.npy')
    y_test = np.load('features/y_test.npy')
    
    print(f"✅ Features loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Charger les noms de classes depuis le rapport
    try:
        with open('results/classification_report_MLP-Full.json', 'r') as f:
            report = json.load(f)
            class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    except:
        class_names = [f"Class_{i}" for i in range(38)]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def create_feature_names(backbone_dims=None):
    """Create feature names for SHAP plots"""
    if backbone_dims is None:
        backbone_dims = {
            'ResNet50': 2048,
            'EfficientNetB0': 1280,
            'MobileNetV2': 1280
        }
    
    feature_names = []
    for name, dim in backbone_dims.items():
        for i in range(min(dim, 500)):  # Limit for readability
            feature_names.append(f"{name}_{i}")
    
    # Compléter avec des noms génériques si nécessaire
    total_features = sum(min(dim, 500) for dim in backbone_dims.values())
    if total_features < 4608:
        for i in range(total_features, 4608):
            feature_names.append(f"F{i}")
    
    return feature_names


def main():
    """Main function for SHAP-only analysis"""
    
    print("\n" + "="*60)
    print("SHAP-ONLY ANALYSIS")
    print("="*60)
    print("This script will run SHAP analysis on the trained models.")
    
    # Vérifier que SHAP est installé
    if shap is None:
        print("\n❌ SHAP is not installed. Please install it with:")
        print("pip install shap")
        return
    
    # Vérifier que les features existent
    if not os.path.exists('features/X_train.npy'):
        print("\n❌ Features not found. Please run the full pipeline first.")
        return
    
    # Charger les features
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features()
    num_classes = len(np.unique(y_train))
    
    print(f"\n📊 Dataset summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Créer les noms de features
    feature_names = create_feature_names()
    print(f"  Created {len(feature_names)} feature names")
    
    # Options pour l'analyse
    print("\n" + "="*60)
    print("Choose model for SHAP analysis:")
    print("="*60)
    print("1. MLP-Full (all 4608 features)")
    print("2. MLP-PCA (814 features, 95% variance)")
    print("3. Both models")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    models_to_analyze = []
    if choice == '1':
        models_to_analyze = ['MLP-Full']
    elif choice == '2':
        models_to_analyze = ['MLP-PCA']
    elif choice == '3':
        models_to_analyze = ['MLP-Full', 'MLP-PCA']
    else:
        print("Invalid choice. Running both models.")
        models_to_analyze = ['MLP-Full', 'MLP-PCA']
    
    # Analyse SHAP
    for model_name in models_to_analyze:
        print(f"\n" + "="*60)
        print(f"SHAP Analysis for {model_name}")
        print("="*60)
        
        # Créer le modèle
        if model_name == 'MLP-Full':
            meta_learner = MetaLearner(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                model_type='mlp_full'
            )
            
            # Entraîner le scaler
            print("  Fitting scaler...")
            meta_learner.scaler.fit(X_train)
            
            # Créer le modèle
            print("  Building model...")
            meta_learner.build_mlp_full()
            
            print(f"  ✅ {model_name} model created")
            
            # Entraînement rapide
            print("  Quick training (5 epochs)...")
            X_train_scaled = meta_learner.scaler.transform(X_train)
            X_val_scaled = meta_learner.scaler.transform(X_val)
            
            history = meta_learner.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=5,
                batch_size=32,
                verbose=0
            )
            print(f"  Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
            
        else:  # MLP-PCA
            meta_learner = MetaLearner(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                model_type='mlp_pca'
            )
            
            # Appliquer PCA
            print("  Fitting PCA...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95, random_state=SEED)
            X_train_pca = pca.fit_transform(X_train)
            
            meta_learner.pca = pca
            print(f"  🔄 PCA: {X_train.shape[1]} → {X_train_pca.shape[1]} features")
            print(f"  📈 Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
            
            # Entraîner le scaler
            print("  Fitting scaler...")
            meta_learner.scaler.fit(X_train_pca)
            
            # Créer le modèle
            print("  Building model...")
            meta_learner.build_mlp_pca(n_components=X_train_pca.shape[1])
            
            print(f"  ✅ {model_name} model created")
            
            # Entraînement rapide
            print("  Quick training (5 epochs)...")
            X_train_scaled = meta_learner.scaler.transform(X_train_pca)
            X_val_pca = pca.transform(X_val)
            X_val_scaled = meta_learner.scaler.transform(X_val_pca)
            
            history = meta_learner.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=5,
                batch_size=32,
                verbose=0
            )
            print(f"  Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        # Vérifier que le modèle fonctionne
        print("\n  Testing model prediction...")
        test_pred = meta_learner.predict(X_test[:10])
        print(f"  ✅ Model working, predictions shape: {test_pred.shape}")
        
        # Sample data for SHAP
        X_background = X_train[:200]
        X_explain = X_test[:100]
        
        # Créer et exécuter l'analyseur SHAP
        analyzer = ShapAnalyzer(
            meta_learner, 
            X_background, 
            X_explain,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Sauvegarder dans un dossier spécifique
        output_dir = f'diagnostics/shap_{model_name.lower().replace("-", "_")}'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            shap_values = analyzer.analyze(output_dir=output_dir)
            print(f"\n✅ SHAP analysis completed for {model_name}")
            print(f"   Figures saved to: {output_dir}/")
        except Exception as e:
            print(f"\n❌ Error during SHAP analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    if 'MLP-Full' in models_to_analyze:
        print("  - diagnostics/shap_mlp_full/shap_summary.png")
        print("  - diagnostics/shap_mlp_full/shap_importance.png")
    if 'MLP-PCA' in models_to_analyze:
        print("  - diagnostics/shap_mlp_pca/shap_summary.png")
        print("  - diagnostics/shap_mlp_pca/shap_importance.png")

        # SALIENCY ANALYSIS
try:
    saliency_analyzer = SaliencyAnalyzer(
        meta_learner,
        X_explain,
        feature_names=feature_names
    )
    
    saliency = saliency_analyzer.analyze(output_dir=output_dir)
    
    print(f"\n✅ Saliency analysis completed for {model_name}")
    
except Exception as e:
    print(f"\n❌ Error during saliency analysis: {e}")
if __name__ == '__main__':
    main()