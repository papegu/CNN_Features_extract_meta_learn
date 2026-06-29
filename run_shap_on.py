"""
run_shap_only.py - CORRIGÉ V3 avec export CSV complet
"""

import os
import numpy as np
import pandas as pd
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
        from tensorflow.keras import layers, models, optimizers, regularizers
        
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.BatchNormalization()(inputs)
        
        # Hidden layer 1
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Hidden layer 2
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4))(x)
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
        from tensorflow.keras import layers, models, optimizers, regularizers
        
        inputs = layers.Input(shape=(n_components,))
        x = layers.BatchNormalization()(inputs)
        
        # Hidden layer 1
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        # Hidden layer 2
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4))(x)
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
    """SHAP analysis with CSV export"""
    
    def __init__(self, model, X_background, X_explain, feature_names=None, class_names=None):
        self.model = model
        self.X_background = X_background
        self.X_explain = X_explain
        self.feature_names = feature_names if feature_names else [f"F{i}" for i in range(X_background.shape[1])]
        self.class_names = class_names
    
    def get_backbone_from_index(self, idx):
        """Determine backbone based on index range"""
        if idx < 2048:
            return 'ResNet50'
        elif idx < 2048 + 1280:
            return 'EfficientNetB0'
        elif idx < 2048 + 1280 + 1280:
            return 'MobileNetV2'
        else:
            return 'Unknown'
    
    def save_shap_values_to_csv(self, shap_values, output_dir):
        """Save all SHAP values to CSV with proper indexing"""
        
        if isinstance(shap_values, list):
            # Pour classification multi-classe, prendre la moyenne sur toutes les classes
            n_features = shap_values[0].shape[1]
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            n_features = shap_values.shape[1]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Créer un DataFrame avec toutes les features
        df_all = pd.DataFrame({
            'Feature_Index': np.arange(n_features),
            'Feature_Name': self.feature_names[:n_features],
            'Mean_Abs_SHAP': mean_abs_shap,
            'Backbone': [self.get_backbone_from_index(i) for i in range(n_features)]
        })
        
        # Trier par importance
        df_all = df_all.sort_values('Mean_Abs_SHAP', ascending=False)
        
        # Sauvegarder toutes les features
        df_all.to_csv(os.path.join(output_dir, 'all_shap_values.csv'), index=False)
        print(f"✅ Saved all SHAP values: {output_dir}/all_shap_values.csv")
        
        # Sauvegarder top 100
        df_all.head(100).to_csv(os.path.join(output_dir, 'top100_shap_values.csv'), index=False)
        print(f"✅ Saved top 100: {output_dir}/top100_shap_values.csv")
        
        # Statistiques par backbone
        print("\n📊 SHAP Statistics by Backbone (all features):")
        backbone_stats = df_all.groupby('Backbone').agg({
            'Mean_Abs_SHAP': ['count', 'sum', 'mean']
        }).round(6)
        print(backbone_stats)
        
        # Sauvegarder les stats
        backbone_stats.to_csv(os.path.join(output_dir, 'backbone_statistics.csv'))
        
        return df_all
    
    def plot_top_features(self, df_all, output_dir, n_top=20):
        """Plot top N features with clear labels"""
        
        # Prendre les top N
        df_top = df_all.head(n_top).copy()
        df_top = df_top.sort_values('Mean_Abs_SHAP', ascending=True)
        
        # Créer des noms courts pour l'affichage
        short_names = []
        for idx, backbone in zip(df_top['Feature_Index'], df_top['Backbone']):
            if backbone == 'ResNet50':
                short_names.append(f"R{idx}")
            elif backbone == 'EfficientNetB0':
                short_names.append(f"E{idx - 2048}")
            elif backbone == 'MobileNetV2':
                short_names.append(f"M{idx - 3328}")
            else:
                short_names.append(f"F{idx}")
        
        # Couleurs par backbone
        colors = {
            'ResNet50': '#D55E00',      # Orange
            'EfficientNetB0': '#0072B2', # Bleu
            'MobileNetV2': '#009E73',    # Vert
            'Unknown': '#999999'         # Gris
        }
        bar_colors = [colors[b] for b in df_top['Backbone']]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Barres horizontales
        y_pos = np.arange(len(df_top))
        bars = ax.barh(y_pos, df_top['Mean_Abs_SHAP'], 
                      color=bar_colors,
                      height=0.7,
                      edgecolor='black',
                      linewidth=1,
                      alpha=0.9)
        
        # Personnalisation
        ax.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold')
        ax.set_ylabel('Features', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {n_top} Most Important Features by Backbone', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Labels Y avec noms courts et backbones
        y_labels = [f"{name} ({backbone})" for name, backbone in zip(short_names, df_top['Backbone'])]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10)
        
        # Ajouter les valeurs sur les barres
        max_val = df_top['Mean_Abs_SHAP'].max()
        for i, (bar, val) in enumerate(zip(bars, df_top['Mean_Abs_SHAP'])):
            ax.text(val + 0.02 * max_val, bar.get_y() + bar.get_height()/2, 
                   f'{val:.6f}', va='center', fontsize=9, fontweight='bold')
        
        # Légende
        legend_elements = []
        for backbone, color in colors.items():
            if backbone in df_top['Backbone'].values:
                count = sum(df_top['Backbone'] == backbone)
                legend_elements.append(
                    plt.Rectangle((0,0),1,1, facecolor=color, 
                                 label=f'{backbone} ({count} features)')
                )
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Grille
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top{n_top}_features.png'), 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved plot: {output_dir}/top{n_top}_features.png")
        
        return df_top
    
    def analyze(self, output_dir='diagnostics'):
        """Run SHAP analysis with CSV export"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("SHAP ANALYSIS WITH CSV EXPORT")
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
            def model_predict(x):
                x_pca = self.model.pca.transform(x)
                x_scaled = self.model.scaler.transform(x_pca)
                return self.model.model.predict(x_scaled, verbose=0)
            
            background_subset = X_back[:50]
            explainer = shap.GradientExplainer(self.model.model, background_subset)
            shap_values = explainer.shap_values(X_exp)
            
        else:  # mlp_full
            def model_predict(x):
                x_scaled = self.model.scaler.transform(x)
                return self.model.model.predict(x_scaled, verbose=0)
            
            background_subset = X_back[:50]
            explainer = shap.GradientExplainer(self.model.model, background_subset)
            shap_values = explainer.shap_values(X_exp)
        
        # Sauvegarder toutes les valeurs SHAP dans CSV
        df_all = self.save_shap_values_to_csv(shap_values, output_dir)
        
        # Plot top 20 et top 50
        df_top20 = self.plot_top_features(df_all, output_dir, n_top=20)
        df_top50 = self.plot_top_features(df_all, output_dir, n_top=50)
        
        return shap_values, df_all, df_top20, df_top50


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
    
    # Charger les noms de classes
    try:
        with open('results/classification_report_MLP-Full.json', 'r') as f:
            report = json.load(f)
            class_names = [k for k in report.keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg']]
    except:
        class_names = [f"Class_{i}" for i in range(38)]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


def create_feature_names():
    """Create feature names for SHAP plots"""
    backbone_dims = {
        'ResNet50': 2048,
        'EfficientNetB0': 1280,
        'MobileNetV2': 1280
    }
    
    feature_names = []
    for name, dim in backbone_dims.items():
        for i in range(dim):
            feature_names.append(f"{name}_{i}")
    
    return feature_names


def main():
    """Main function for SHAP-only analysis"""
    
    print("\n" + "="*60)
    print("SHAP-ONLY ANALYSIS WITH CSV EXPORT")
    print("="*60)
    
    # Vérifier SHAP
    if shap is None:
        print("\n❌ SHAP is not installed.")
        return
    
    # Vérifier les features
    if not os.path.exists('features/X_train.npy'):
        print("\n❌ Features not found.")
        return
    
    # Charger les features
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features()
    num_classes = len(np.unique(y_train))
    
    print(f"\n📊 Dataset summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    
    # Créer les noms de features
    feature_names = create_feature_names()
    print(f"  Created {len(feature_names)} feature names")
    
    # Menu choix modèle
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
        print("Invalid choice. Running MLP-Full only.")
        models_to_analyze = ['MLP-Full']
    
    # Analyse SHAP pour chaque modèle
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
            
            print("  Fitting scaler...")
            meta_learner.scaler.fit(X_train)
            
            print("  Building model...")
            meta_learner.build_mlp_full()
            
            print(f"  ✅ {model_name} model created")
            
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
            
            print("  Fitting PCA...")
            pca = PCA(n_components=0.95, random_state=SEED)
            X_train_pca = pca.fit_transform(X_train)
            
            meta_learner.pca = pca
            print(f"  🔄 PCA: {X_train.shape[1]} → {X_train_pca.shape[1]} features")
            print(f"  📈 Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
            
            print("  Fitting scaler...")
            meta_learner.scaler.fit(X_train_pca)
            
            print("  Building model...")
            meta_learner.build_mlp_pca(n_components=X_train_pca.shape[1])
            
            print(f"  ✅ {model_name} model created")
            
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
        
        # Vérifier le modèle
        print("\n  Testing model prediction...")
        test_pred = meta_learner.predict(X_test[:10])
        print(f"  ✅ Model working, predictions shape: {test_pred.shape}")
        
        # Sample data for SHAP
        X_background = X_train[:200]
        X_explain = X_test[:100]
        
        # Créer l'analyseur SHAP
        analyzer = ShapAnalyzer(
            meta_learner, 
            X_background, 
            X_explain,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Output directory
        output_dir = f'diagnostics/shap_{model_name.lower().replace("-", "_")}_export'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            shap_values, df_all, df_top20, df_top50 = analyzer.analyze(output_dir=output_dir)
            print(f"\n✅ SHAP analysis completed for {model_name}")
            print(f"   Files saved to: {output_dir}/")
            
        except Exception as e:
            print(f"\n❌ Error during SHAP analysis: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    for model in models_to_analyze:
        model_dir = f'diagnostics/shap_{model.lower().replace("-", "_")}_export'
        print(f"  {model}:")
        print(f"    - {model_dir}/all_shap_values.csv (all {X_train.shape[1]} features)")
        print(f"    - {model_dir}/top100_shap_values.csv (top 100)")
        print(f"    - {model_dir}/backbone_statistics.csv")
        print(f"    - {model_dir}/top20_features.png")
        print(f"    - {model_dir}/top50_features.png")


if __name__ == '__main__':
    main()