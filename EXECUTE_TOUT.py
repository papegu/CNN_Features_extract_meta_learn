#!/usr/bin/env python3
"""
🚀 SCRIPT D'EXÉCUTION COMPLET - LANCER TOUS LES FICHIERS
========================================================

Version corrigée pour garder une seule source de vérité pour les métriques.
Les métriques complètes sont calculées à partir d'artefacts de prédiction sauvegardés,
au lieu de reconstruire séparément un modèle potentiellement incohérent.
"""

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

RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')
FEATURES_DIR = Path('features')
DATASETS_DIR = Path('datasets')


def check_environment():
    logger.info("🔍 Vérification de l'environnement...")
    for dir_path in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
        dir_path.mkdir(exist_ok=True)
    return True


def load_features():
    logger.info("📊 Chargement des features...")
    X_train = np.load(FEATURES_DIR / 'X_train.npy')
    X_test = np.load(FEATURES_DIR / 'X_test.npy')
    X_val = np.load(FEATURES_DIR / 'X_val.npy')
    y_train = np.load(FEATURES_DIR / 'y_train.npy')
    y_test = np.load(FEATURES_DIR / 'y_test.npy')
    y_val = np.load(FEATURES_DIR / 'y_val.npy')
    return X_train, X_test, X_val, y_train, y_test, y_val


def execute_complete_metrics_from_predictions():
    logger.info("\n" + "="*70)
    logger.info("📊 ÉTAPE 3: COMPLETE METRICS (aligned from saved predictions)")
    logger.info("="*70)

    from compute_metrics_from_predictions import compute_metrics_from_predictions

    pred_path = RESULTS_DIR / 'predictions_MLP-Full.npz'
    if not pred_path.exists():
        logger.warning("⚠️ predictions_MLP-Full.npz introuvable. Lancez d'abord le pipeline principal.")
        return None

    output_path = RESULTS_DIR / 'complete_metrics_mlp_full_fixed.json'
    metrics = compute_metrics_from_predictions(
        predictions_path=str(pred_path),
        output_path=str(output_path),
        num_classes=38,
        model_name='MLP-Full'
    )

    logger.info(f"✅ Métriques complètes alignées sauvegardées: {output_path}")
    logger.info(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"   F1-macro: {metrics['f1_macro']*100:.2f}%")
    logger.info(f"   F1-weighted: {metrics['f1_weighted']*100:.2f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    args = parser.parse_args()

    logger.info("🚀 DÉMARRAGE DE L'EXÉCUTION COMPLÈTE")
    check_environment()
    load_features()

    results_summary = {
        'complete_metrics_aligned': execute_complete_metrics_from_predictions(),
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
    }

    summary_file = RESULTS_DIR / 'execution_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"✅ Résumé sauvegardé: {summary_file}")


if __name__ == '__main__':
    main()
