"""
compute_complete_metrics.py

COMPREHENSIVE METRICS: Precision, Recall, F1, ROC-AUC, Balanced Accuracy
Addresses Reviewer 2 criticism: "L'exactitude, la précision, le rappel, le score F1 
et l'aire sous la courbe ROC ne sont pas indiqués"

Computes and reports all metrics for complete model evaluation and comparison.
"""

import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, hamming_loss
)
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


class MetricsComputer:
    """Compute comprehensive evaluation metrics"""
    
    def __init__(self, num_classes=38):
        self.num_classes = num_classes
        self.metrics = {}
    
    def compute_all_metrics(self, y_true, y_pred_proba, y_pred_labels=None, 
                           class_names=None, model_name='Model'):
        """
        Compute all relevant metrics
        
        Args:
            y_true: True labels (one-hot encoded or class indices)
            y_pred_proba: Prediction probabilities (batch_size x num_classes)
            y_pred_labels: Predicted class labels (if not provided, computed from proba)
            class_names: Class names for per-class metrics
            model_name: Name of model for reporting
            
        Returns:
            Dictionary with all metrics
        """
        
        # Ensure proper format
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true.flatten()
        
        if y_pred_labels is None:
            y_pred_labels = np.argmax(y_pred_proba, axis=1)
        
        metrics = {}
        
        # ========== BASIC ACCURACY ==========
        metrics['accuracy'] = accuracy_score(y_true_labels, y_pred_labels)
        
        # ========== PRECISION (3 variants) ==========
        metrics['precision_macro'] = precision_score(
            y_true_labels, y_pred_labels, 
            average='macro', 
            zero_division=0
        )
        metrics['precision_micro'] = precision_score(
            y_true_labels, y_pred_labels, 
            average='micro', 
            zero_division=0
        )
        metrics['precision_weighted'] = precision_score(
            y_true_labels, y_pred_labels, 
            average='weighted', 
            zero_division=0
        )
        
        # ========== RECALL (3 variants) ==========
        metrics['recall_macro'] = recall_score(
            y_true_labels, y_pred_labels, 
            average='macro', 
            zero_division=0
        )
        metrics['recall_micro'] = recall_score(
            y_true_labels, y_pred_labels, 
            average='micro', 
            zero_division=0
        )
        metrics['recall_weighted'] = recall_score(
            y_true_labels, y_pred_labels, 
            average='weighted', 
            zero_division=0
        )
        
        # ========== F1-SCORE (3 variants) ==========
        metrics['f1_macro'] = f1_score(
            y_true_labels, y_pred_labels, 
            average='macro', 
            zero_division=0
        )
        metrics['f1_micro'] = f1_score(
            y_true_labels, y_pred_labels, 
            average='micro', 
            zero_division=0
        )
        metrics['f1_weighted'] = f1_score(
            y_true_labels, y_pred_labels, 
            average='weighted', 
            zero_division=0
        )
        
        # ========== BALANCED ACCURACY ==========
        # Important for imbalanced datasets
        metrics['balanced_accuracy'] = balanced_accuracy_score(
            y_true_labels, y_pred_labels
        )
        
        # ========== ROC-AUC SCORES ==========
        try:
            # Convert to one-hot for ROC-AUC computation
            y_true_onehot = tf.keras.utils.to_categorical(y_true_labels, self.num_classes)
            
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true_onehot,
                y_pred_proba,
                average='macro',
                multi_class='ovr'
            )
            metrics['roc_auc_weighted'] = roc_auc_score(
                y_true_onehot,
                y_pred_proba,
                average='weighted',
                multi_class='ovr'
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not compute ROC-AUC: {e}")
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_weighted'] = None
        
        # ========== CONFUSION MATRIX ==========
        metrics['confusion_matrix'] = confusion_matrix(
            y_true_labels, y_pred_labels
        ).tolist()
        
        # ========== HAMMING LOSS ==========
        # Fraction of labels that are incorrectly predicted
        metrics['hamming_loss'] = hamming_loss(y_true_labels, y_pred_labels)
        
        # ========== PER-CLASS METRICS ==========
        precision_per_class = precision_score(
            y_true_labels, y_pred_labels, 
            average=None, 
            zero_division=0
        )
        recall_per_class = recall_score(
            y_true_labels, y_pred_labels, 
            average=None, 
            zero_division=0
        )
        f1_per_class = f1_score(
            y_true_labels, y_pred_labels, 
            average=None, 
            zero_division=0
        )
        support_per_class = np.bincount(y_true_labels, minlength=self.num_classes)
        
        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        }
        
        # ========== CLASSIFICATION REPORT ==========
        metrics['classification_report'] = classification_report(
            y_true_labels, y_pred_labels,
            output_dict=True,
            zero_division=0
        )
        
        # ========== ADDITIONAL STATISTICS ==========
        # Worst and best performing classes
        f1_scores = metrics['per_class']['f1']
        metrics['best_class'] = {
            'class_idx': int(np.argmax(f1_scores)),
            'f1_score': float(np.max(f1_scores))
        }
        metrics['worst_class'] = {
            'class_idx': int(np.argmin(f1_scores)),
            'f1_score': float(np.min(f1_scores))
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def print_metrics_table(self, model_name='Model', include_per_class=False):
        """
        Print formatted metrics table
        """
        if model_name not in self.metrics:
            print(f"Error: Model '{model_name}' not found in computed metrics")
            return
        
        metrics = self.metrics[model_name]
        
        print("\n" + "="*80)
        print(f"COMPREHENSIVE METRICS: {model_name}")
        print("="*80)
        
        # Main metrics table
        metrics_display = [
            ("Accuracy", metrics['accuracy']),
            ("Balanced Accuracy", metrics['balanced_accuracy']),
            ("Hamming Loss", metrics['hamming_loss']),
            ("", None),
            ("Precision (Macro)", metrics['precision_macro']),
            ("Precision (Weighted)", metrics['precision_weighted']),
            ("", None),
            ("Recall (Macro)", metrics['recall_macro']),
            ("Recall (Weighted)", metrics['recall_weighted']),
            ("", None),
            ("F1-Score (Macro)", metrics['f1_macro']),
            ("F1-Score (Weighted)", metrics['f1_weighted']),
        ]
        
        if metrics['roc_auc_macro'] is not None:
            metrics_display.extend([
                ("", None),
                ("ROC-AUC (Macro)", metrics['roc_auc_macro']),
                ("ROC-AUC (Weighted)", metrics['roc_auc_weighted']),
            ])
        
        print(f"\n{'Metric':<30} {'Score':<15}")
        print("-"*45)
        
        for metric_name, score in metrics_display:
            if score is None:
                print()
            else:
                print(f"{metric_name:<30} {score:<15.4f}")
        
        # Best/Worst classes
        print("\n" + "-"*45)
        print(f"Best Class:  #{metrics['best_class']['class_idx']:2d} "
              f"(F1: {metrics['best_class']['f1_score']:.4f})")
        print(f"Worst Class: #{metrics['worst_class']['class_idx']:2d} "
              f"(F1: {metrics['worst_class']['f1_score']:.4f})")
        
        print("="*80 + "\n")
        
        # Per-class metrics if requested
        if include_per_class:
            self.print_per_class_metrics(model_name)
    
    def print_per_class_metrics(self, model_name='Model'):
        """Print detailed per-class metrics"""
        if model_name not in self.metrics:
            print(f"Error: Model '{model_name}' not found")
            return
        
        metrics = self.metrics[model_name]
        per_class = metrics['per_class']
        
        print("\n" + "="*80)
        print(f"PER-CLASS METRICS: {model_name}")
        print("="*80)
        print(f"\n{'Class':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
        print("-"*80)
        
        for i in range(len(per_class['f1'])):
            print(f"{i:<6} {per_class['precision'][i]:<12.4f} "
                  f"{per_class['recall'][i]:<12.4f} {per_class['f1'][i]:<12.4f} "
                  f"{per_class['support'][i]:<8}")
        
        print("="*80 + "\n")
    
    def compare_models(self, model_names=None):
        """
        Compare multiple models side-by-side
        """
        if model_names is None:
            model_names = list(self.metrics.keys())
        
        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)
        
        # Prepare comparison table
        comparison_data = {
            'Accuracy': [],
            'Balanced Acc': [],
            'Precision (W)': [],
            'Recall (W)': [],
            'F1-Score (W)': [],
            'ROC-AUC (M)': [],
        }
        
        for model_name in model_names:
            if model_name not in self.metrics:
                continue
            
            m = self.metrics[model_name]
            comparison_data['Accuracy'].append(m['accuracy'])
            comparison_data['Balanced Acc'].append(m['balanced_accuracy'])
            comparison_data['Precision (W)'].append(m['precision_weighted'])
            comparison_data['Recall (W)'].append(m['recall_weighted'])
            comparison_data['F1-Score (W)'].append(m['f1_weighted'])
            comparison_data['ROC-AUC (M)'].append(
                m['roc_auc_macro'] if m['roc_auc_macro'] is not None else 0.0
            )
        
        # Print comparison table
        print(f"\n{'Model':<20}", end='')
        for metric in comparison_data.keys():
            print(f" | {metric:<12}", end='')
        print("\n" + "-"*100)
        
        for i, model_name in enumerate(model_names):
            print(f"{model_name:<20}", end='')
            for metric in comparison_data.keys():
                value = comparison_data[metric][i]
                print(f" | {value:<12.4f}", end='')
            print()
        
        print("="*100 + "\n")
        
        # Best model identification
        best_model_idx = np.argmax(comparison_data['F1-Score (W)'])
        best_model_name = model_names[best_model_idx]
        
        print(f"🏆 BEST MODEL: {best_model_name} "
              f"(F1-Score: {comparison_data['F1-Score (W)'][best_model_idx]:.4f})")
        print()
    
    def save_metrics_json(self, output_path='results/complete_metrics.json'):
        """Save all metrics to JSON file"""
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(self.metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"✅ Metrics saved to: {output_path}")


# ============================================================================
# QUICK START USAGE
# ============================================================================

def example_usage():
    """Example of how to use MetricsComputer"""
    
    # Create random dummy data for demonstration
    num_samples = 1000
    num_classes = 38
    
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred_proba = np.random.dirichlet(np.ones(num_classes), num_samples)
    
    # Compute metrics
    computer = MetricsComputer(num_classes=num_classes)
    metrics_mlp_full = computer.compute_all_metrics(
        y_true, y_pred_proba, model_name='MLP-Full'
    )
    
    # Print results
    computer.print_metrics_table('MLP-Full')
    
    # Compare multiple models (if you have them)
    computer.print_metrics_table('MLP-Full', include_per_class=False)
    
    # Save to file
    computer.save_metrics_json()


if __name__ == '__main__':
    print("✅ Complete Metrics module loaded successfully")
    print("\nUsage:")
    print("  from compute_complete_metrics import MetricsComputer")
    print("  computer = MetricsComputer(num_classes=38)")
    print("  metrics = computer.compute_all_metrics(y_true, y_pred_proba)")
    print("  computer.print_metrics_table('ModelName')")
    print("  computer.save_metrics_json()")
