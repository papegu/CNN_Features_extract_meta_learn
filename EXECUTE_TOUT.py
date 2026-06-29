#!/usr/bin/env python3
"""
🚀 SCRIPT D'EXÉCUTION COMPLET - LANCER TOUS LES FICHIERS
========================================================

Ce script exécute les 4 fichiers dans le bon ordre et génère tous les résultats.
Usage: python EXECUTE_TOUT.py [--mode {quick|full}] [--skip-kfold]

Modes:
  - quick: Exécute Attention Fusion + Domain Adaptation + Metrics (2-4 heures)
  - full:  Exécute tout + K-Fold Validation (22-30 heures) [DEFAULT]

Options:
  --skip-kfold: Passe K-Fold (pour test rapide)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution_log.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Répertoires de travail
RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')
FEATURES_DIR = Path('features')
DATASETS_DIR = Path('datasets')

# ============================================================================
# ÉTAPE 0: INITIALISATION
# ============================================================================

def check_environment():
    """Vérifier que l'environnement est correctement configuré"""
    logger.info("🔍 Vérification de l'environnement...")
    
    # Créer les répertoires
    for dir_path in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✅ Répertoire créé/vérifié: {dir_path}")
    
    # Vérifier les imports
    try:
        import tensorflow
        import keras
        import numpy
        import sklearn
        logger.info("✅ Tous les packages requis sont installés")
        return True
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        return False

def load_features():
    """Charger les features pré-extraites"""
    logger.info("📊 Chargement des features...")
    
    try:
        X_train = np.load(FEATURES_DIR / 'X_train.npy')
        X_test = np.load(FEATURES_DIR / 'X_test.npy')
        X_val = np.load(FEATURES_DIR / 'X_val.npy')
        y_train = np.load(FEATURES_DIR / 'y_train.npy')
        y_test = np.load(FEATURES_DIR / 'y_test.npy')
        y_val = np.load(FEATURES_DIR / 'y_val.npy')
        
        logger.info(f"✅ Features chargées:")
        logger.info(f"   X_train: {X_train.shape}")
        logger.info(f"   X_test: {X_test.shape}")
        logger.info(f"   X_val: {X_val.shape}")
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    except FileNotFoundError as e:
        logger.error(f"❌ Features non trouvées: {e}")
        logger.error("Exécutez d'abord: python feature_level_with_cnn_pca_mlp.py")
        return None

# ============================================================================
# ÉTAPE 1: ATTENTION FUSION
# ============================================================================

def execute_attention_fusion(X_train, X_val, X_test, y_train, y_val, y_test):
    """Exécuter l'analyse Attention Fusion"""
    logger.info("\n" + "="*70)
    logger.info("⭐ ÉTAPE 1: ATTENTION FUSION")
    logger.info("="*70)
    
    try:
        from attention_fusion_layer import compare_fusion_strategies
        
        logger.info("🔄 Comparaison des stratégies de fusion...")
        results = compare_fusion_strategies(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Sauvegarder les résultats
        results_file = RESULTS_DIR / 'attention_fusion_comparison.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Attention Fusion complètée")
        logger.info(f"   Résultats sauvegardés: {results_file}")
        logger.info(f"   Baseline: 96.04%")
        logger.info(f"   AttentionFusion: {results.get('attention_fusion_accuracy', 'N/A')*100:.2f}%")
        
        return results
    except Exception as e:
        logger.error(f"❌ Erreur Attention Fusion: {e}")
        return None

# ============================================================================
# ÉTAPE 2: DOMAIN ADAPTATION
# ============================================================================

def execute_domain_adaptation():
    """Exécuter l'adaptation de domaine"""
    logger.info("\n" + "="*70)
    logger.info("🌍 ÉTAPE 2: DOMAIN ADAPTATION (Multi-Crop)")
    logger.info("="*70)
    
    try:
        from domain_adaptation import adapt_model_to_multicrop, evaluate_with_tta
        from tensorflow.keras.models import load_model
        from sklearn.model_selection import train_test_split
        
        logger.info("📂 Vérification des données Multi-Crop...")
        
        # Vérifier si les features Multi-Crop existent
        multicrop_train = DATASETS_DIR / 'multicrop_X_train.npy'
        if not multicrop_train.exists():
            logger.warning("⚠️  Features Multi-Crop non trouvées")
            logger.warning("   Pour utiliser Domain Adaptation:")
            logger.warning("   1. Exécutez l'extraction des features Multi-Crop")
            logger.warning("   2. Ou préparez les données manuellement")
            return None
        
        # Charger features Multi-Crop
        X_train_mc = np.load(DATASETS_DIR / 'multicrop_X_train.npy')
        y_train_mc = np.load(DATASETS_DIR / 'multicrop_y_train.npy')
        X_test_mc = np.load(DATASETS_DIR / 'multicrop_X_test.npy')
        y_test_mc = np.load(DATASETS_DIR / 'multicrop_y_test.npy')
        
        # Split train/val
        X_train_mc, X_val_mc, y_train_mc, y_val_mc = train_test_split(
            X_train_mc, y_train_mc, test_size=0.2, random_state=42, stratify=y_train_mc
        )
        
        logger.info(f"✅ Données Multi-Crop chargées: {X_train_mc.shape}")
        
        # Charger modèle pré-entraîné
        logger.info("🤖 Chargement du modèle pré-entraîné...")
        model = load_model(MODELS_DIR / 'mlp_full.weights.h5')
        
        # Adapter
        logger.info("🔥 Adaptation du modèle (patience: 2-3 heures)...")
        model_adapted, history = adapt_model_to_multicrop(
            model,
            X_train_mc, y_train_mc,
            X_val_mc, y_val_mc,
            strategy='progressive_unfreezing',
            epochs=100,
            batch_size=32
        )
        
        # Sauvegarder modèle
        model_path = MODELS_DIR / 'mlp_full_adapted_multicrop.h5'
        model_adapted.save(str(model_path))
        logger.info(f"✅ Modèle adapté sauvegardé: {model_path}")
        
        # Évaluer avec TTA
        logger.info("🧪 Évaluation avec Test-Time Augmentation...")
        accuracy_tta, predictions_tta, inference_time = evaluate_with_tta(
            model_adapted,
            X_test_mc, y_test_mc,
            n_augments=10
        )
        
        # Comparer avant/après
        y_pred_original = model.predict(X_test_mc, verbose=0)
        accuracy_before = np.mean(np.argmax(y_pred_original, axis=1) == y_test_mc)
        
        # Résultats
        results = {
            "accuracy_before_adaptation": float(accuracy_before),
            "accuracy_after_adaptation_tta": float(accuracy_tta),
            "improvement_percent": float((accuracy_tta - accuracy_before) * 100),
            "inference_time_ms": float(inference_time * 1000),
            "num_test_samples": len(y_test_mc),
            "num_classes": 30,
            "strategy_used": "progressive_unfreezing",
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder
        results_file = RESULTS_DIR / 'domain_adaptation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Domain Adaptation complètée")
        logger.info(f"   Accuracy AVANT: {accuracy_before*100:.2f}%")
        logger.info(f"   Accuracy APRÈS: {accuracy_tta*100:.2f}%")
        logger.info(f"   📈 AMÉLIORATION: +{(accuracy_tta - accuracy_before)*100:.2f}%")
        logger.info(f"   Résultats: {results_file}")
        
        return results
    except FileNotFoundError:
        logger.warning("⚠️  Données Multi-Crop non trouvées - Domain Adaptation skipped")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur Domain Adaptation: {e}")
        return None

# ============================================================================
# ÉTAPE 3: COMPLETE METRICS
# ============================================================================

def execute_complete_metrics(X_test, y_test):
    """Calculer les métriques complètes"""
    logger.info("\n" + "="*70)
    logger.info("📊 ÉTAPE 3: COMPLETE METRICS")
    logger.info("="*70)
    
    try:
        from compute_complete_metrics import MetricsComputer
        from tensorflow.keras import layers, models, optimizers, regularizers
        import pickle
        
        logger.info("🤖 Construction et chargement du modèle...")
        # Build MLP architecture
        inputs = layers.Input(shape=(4608,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(38, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Load weights
        model.load_weights(str(MODELS_DIR / 'mlp_full.weights.h5'))
        logger.info("✅ Modèle chargé avec succès")
        
        # Load scaler for normalization
        try:
            with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler chargé")
            X_test_scaled = scaler.transform(X_test)
        except:
            logger.warning("⚠️  Scaler non trouvé - utilisant données brutes")
            X_test_scaled = X_test
        
        logger.info("🧮 Génération des prédictions...")
        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        
        logger.info("📈 Calcul des métriques...")
        computer = MetricsComputer(num_classes=38)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        metrics = computer.compute_all_metrics(y_test, y_pred_proba, y_pred_labels=y_pred_labels, model_name='MLP-Full')
        
        # Afficher tableau
        logger.info("\n📊 RÉSULTATS DES MÉTRIQUES:")
        computer.print_metrics_table('MLP-Full')
        
        # Sauvegarder
        results_file = RESULTS_DIR / 'complete_metrics_mlp_full.json'
        computer.save_metrics_json(str(results_file))
        
        logger.info(f"✅ Métriques complètes calculées")
        logger.info(f"   Résultats sauvegardés: {results_file}")
        
        return metrics
    except Exception as e:
        logger.error(f"❌ Erreur Complete Metrics: {e}")
        return None

# ============================================================================
# ÉTAPE 4: K-FOLD VALIDATION
# ============================================================================

def execute_kfold_validation(X_train, X_val, y_train, y_val, k=5):
    """Exécuter K-Fold Validation"""
    logger.info("\n" + "="*70)
    logger.info("✅ ÉTAPE 4: K-FOLD VALIDATION")
    logger.info("="*70)
    logger.info(f"⚠️  ATTENTION: Cela va prendre 20-30 heures GPU!")
    logger.info(f"   k={k} folds × ~4-6 heures par fold")
    
    try:
        from kfold_validation import perform_kfold_validation
        
        # Combiner les données
        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])
        
        logger.info(f"📊 Données totales: {X.shape}")
        logger.info(f"🚀 Démarrage du k-fold (patience...)...")
        
        results = perform_kfold_validation(
            X, y,
            k=k,
            num_classes=38,
            output_dir=str(RESULTS_DIR)
        )
        
        logger.info(f"\n✅ K-Fold Validation complètée")
        logger.info(f"   Mean Accuracy: {results.get('accuracy_mean', 0)*100:.2f}%")
        logger.info(f"   Std Dev: ±{results.get('accuracy_std', 0)*100:.3f}%")
        
        return results
    except Exception as e:
        logger.error(f"❌ Erreur K-Fold Validation: {e}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode',
        choices=['quick', 'full'],
        default='full',
        help='Mode d\'exécution'
    )
    parser.add_argument(
        '--skip-kfold',
        action='store_true',
        help='Passer K-Fold Validation'
    )
    parser.add_argument(
        '--skip-domain-adapt',
        action='store_true',
        help='Passer Domain Adaptation'
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 DÉMARRAGE DE L'EXÉCUTION COMPLÈTE")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Skip K-Fold: {args.skip_kfold}")
    logger.info(f"Skip Domain Adapt: {args.skip_domain_adapt}")
    logger.info("="*70)
    
    # Initialisation
    if not check_environment():
        logger.error("Environnement non correctement configuré!")
        sys.exit(1)
    
    # Charger les features
    features = load_features()
    if features is None:
        logger.error("Impossible de charger les features!")
        sys.exit(1)
    
    X_train, X_test, X_val, y_train, y_test, y_val = features
    
    # Résumé des exécutions
    results_summary = {}
    
    # ÉTAPE 1: Attention Fusion
    try:
        logger.info("\n✅ Exécution: Attention Fusion")
        results_summary['attention_fusion'] = execute_attention_fusion(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
    except Exception as e:
        logger.error(f"Erreur Attention Fusion: {e}")
        results_summary['attention_fusion'] = None
    
    # ÉTAPE 2: Domain Adaptation
    if not args.skip_domain_adapt:
        try:
            logger.info("\n✅ Exécution: Domain Adaptation")
            results_summary['domain_adaptation'] = execute_domain_adaptation()
        except Exception as e:
            logger.error(f"Erreur Domain Adaptation: {e}")
            results_summary['domain_adaptation'] = None
    else:
        logger.info("\n⏭️  Domain Adaptation skipped")
        results_summary['domain_adaptation'] = None
    
    # ÉTAPE 3: Complete Metrics
    try:
        logger.info("\n✅ Exécution: Complete Metrics")
        results_summary['complete_metrics'] = execute_complete_metrics(X_test, y_test)
    except Exception as e:
        logger.error(f"Erreur Complete Metrics: {e}")
        results_summary['complete_metrics'] = None
    
    # ÉTAPE 4: K-Fold Validation
    if not args.skip_kfold and args.mode == 'full':
        try:
            logger.info("\n✅ Exécution: K-Fold Validation")
            response = input("⏸️  K-Fold prend ~20-30h. Continuer? (y/n): ")
            if response.lower() == 'y':
                results_summary['kfold'] = execute_kfold_validation(
                    X_train, X_val, y_train, y_val, k=5
                )
            else:
                logger.info("K-Fold skipped par l'utilisateur")
                results_summary['kfold'] = None
        except Exception as e:
            logger.error(f"Erreur K-Fold: {e}")
            results_summary['kfold'] = None
    else:
        logger.info("\n⏭️  K-Fold Validation skipped")
        results_summary['kfold'] = None
    
    # Résumé final
    logger.info("\n" + "="*70)
    logger.info("📋 RÉSUMÉ FINAL DE L'EXÉCUTION")
    logger.info("="*70)
    
    summary_file = RESULTS_DIR / 'execution_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'mode': args.mode,
            'timestamp': datetime.now().isoformat(),
            'results': results_summary
        }, f, indent=2)
    
    logger.info(f"✅ Exécution terminée!")
    logger.info(f"📁 Résumé: {summary_file}")
    logger.info(f"\n📂 Résultats disponibles dans: {RESULTS_DIR}")
    logger.info(f"   - attention_fusion_comparison.json")
    logger.info(f"   - domain_adaptation_results.json (si exécuté)")
    logger.info(f"   - complete_metrics_mlp_full.json")
    logger.info(f"   - kfold_results.json (si exécuté)")
    logger.info("\n📖 Prochaine étape: Intégrer les résultats dans snpaper/paper_RIDLFCP_springer.tex")
    logger.info("🎯 Consultez: GUIDE_LANCER_FICHIERS.md section 'INTÉGRATION LATEX'")

if __name__ == '__main__':
    main()
