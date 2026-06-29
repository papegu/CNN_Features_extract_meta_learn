"""
kfold_validation.py

K-FOLD CROSS-VALIDATION: Robust validation strategy
Addresses Reviewer 1 criticism: "Validation trop simple (80-10-10), k-fold cross-validation nécessaire"

Implements stratified k-fold cross-validation for robust model evaluation and stability assessment.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 200


class KFoldValidator:
    """Perform stratified k-fold cross-validation"""
    
    def __init__(self, k=5, random_state=SEED):
        self.k = k
        self.random_state = random_state
        self.fold_results = []
        self.overall_results = {}
    
    def validate(self, X, y, model_builder_fn, num_classes=38):
        """
        Perform k-fold cross-validation
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Labels (n_samples x num_classes) one-hot or (n_samples,) class indices
            model_builder_fn: Function that returns a new model instance
            num_classes: Number of classes
            
        Returns:
            Dictionary with fold-wise and overall results
        """
        
        # Ensure labels are class indices
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(
            n_splits=self.k, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        print("\n" + "="*80)
        print(f"K-FOLD CROSS-VALIDATION (k={self.k})")
        print("="*80)
        
        fold_accuracies = []
        fold_f1_scores = []
        fold_balanced_accs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_labels), 1):
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx}/{self.k}")
            print(f"{'='*80}")
            print(f"Training samples: {len(train_idx)}")
            print(f"Testing samples:  {len(test_idx)}")
            
            # Split data
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y_labels[train_idx], y_labels[test_idx]
            
            # Further split training into train/val (80/20 of training data)
            n_train = len(X_train_fold)
            n_val = int(0.2 * n_train)
            n_train = n_train - n_val
            
            indices = np.arange(len(X_train_fold))
            np.random.shuffle(indices)
            
            train_split_idx = indices[:n_train]
            val_split_idx = indices[n_train:]
            
            X_train_f = X_train_fold[train_split_idx]
            X_val_f = X_train_fold[val_split_idx]
            y_train_f = y_train_fold[train_split_idx]
            y_val_f = y_train_fold[val_split_idx]
            
            # Normalize features (fit on train only!)
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train_f)
            X_val_f = scaler.transform(X_val_f)
            X_test_f = scaler.transform(X_test_fold)
            
            # Convert labels to one-hot
            y_train_f_onehot = tf.keras.utils.to_categorical(y_train_f, num_classes)
            y_val_f_onehot = tf.keras.utils.to_categorical(y_val_f, num_classes)
            y_test_f_onehot = tf.keras.utils.to_categorical(y_test_fold, num_classes)
            
            # Build and train model
            model = model_builder_fn()
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
            
            print(f"Training...")
            history = model.fit(
                X_train_f, y_train_f_onehot,
                validation_data=(X_val_f, y_val_f_onehot),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluate on test set
            y_pred_proba = model.predict(X_test_f, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Compute metrics
            fold_acc = accuracy_score(y_test_fold, y_pred)
            fold_f1 = f1_score(y_test_fold, y_pred, average='weighted', zero_division=0)
            fold_balanced = balanced_accuracy_score(y_test_fold, y_pred)
            
            fold_accuracies.append(fold_acc)
            fold_f1_scores.append(fold_f1)
            fold_balanced_accs.append(fold_balanced)
            
            fold_result = {
                'fold': fold_idx,
                'accuracy': float(fold_acc),
                'f1_weighted': float(fold_f1),
                'balanced_accuracy': float(fold_balanced),
                'epochs_trained': len(history.history['loss']),
                'train_final_loss': float(history.history['loss'][-1]),
                'val_final_loss': float(history.history['val_loss'][-1]),
            }
            
            self.fold_results.append(fold_result)
            
            print(f"\nFold {fold_idx} Results:")
            print(f"  Accuracy:           {fold_acc:.4f}")
            print(f"  F1-Score (Weighted): {fold_f1:.4f}")
            print(f"  Balanced Accuracy:  {fold_balanced:.4f}")
            print(f"  Epochs Trained:     {len(history.history['loss'])}")
        
        # Compute overall statistics
        self.overall_results = {
            'k': self.k,
            'accuracy': {
                'mean': float(np.mean(fold_accuracies)),
                'std': float(np.std(fold_accuracies)),
                'min': float(np.min(fold_accuracies)),
                'max': float(np.max(fold_accuracies)),
            },
            'f1_weighted': {
                'mean': float(np.mean(fold_f1_scores)),
                'std': float(np.std(fold_f1_scores)),
                'min': float(np.min(fold_f1_scores)),
                'max': float(np.max(fold_f1_scores)),
            },
            'balanced_accuracy': {
                'mean': float(np.mean(fold_balanced_accs)),
                'std': float(np.std(fold_balanced_accs)),
                'min': float(np.min(fold_balanced_accs)),
                'max': float(np.max(fold_balanced_accs)),
            },
        }
        
        self._print_summary(fold_accuracies, fold_f1_scores, fold_balanced_accs)
        
        return self.fold_results, self.overall_results
    
    def _print_summary(self, accuracies, f1_scores, balanced_accs):
        """Print k-fold summary"""
        
        print("\n" + "="*80)
        print(f"K-FOLD CROSS-VALIDATION SUMMARY (k={self.k})")
        print("="*80)
        
        print(f"\n{'Metric':<25} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        
        # Accuracy
        print(f"{'Accuracy':<25} {np.mean(accuracies):<12.4f} "
              f"{np.std(accuracies):<12.4f} {np.min(accuracies):<12.4f} "
              f"{np.max(accuracies):<12.4f}")
        
        # F1-Score
        print(f"{'F1-Score (Weighted)':<25} {np.mean(f1_scores):<12.4f} "
              f"{np.std(f1_scores):<12.4f} {np.min(f1_scores):<12.4f} "
              f"{np.max(f1_scores):<12.4f}")
        
        # Balanced Accuracy
        print(f"{'Balanced Accuracy':<25} {np.mean(balanced_accs):<12.4f} "
              f"{np.std(balanced_accs):<12.4f} {np.min(balanced_accs):<12.4f} "
              f"{np.max(balanced_accs):<12.4f}")
        
        print("="*80 + "\n")
        
        # Interpretation
        print("INTERPRETATION:")
        print(f"  ✅ Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  ✅ Model Stability: Std Dev = {np.std(accuracies):.4f} (lower is better)")
        
        if np.std(accuracies) < 0.01:
            print("     → Excellent stability (very consistent across folds)")
        elif np.std(accuracies) < 0.02:
            print("     → Good stability (minor variance)")
        else:
            print("     → Variable stability (consider more folds or data balance)")
        
        print()
    
    def save_results(self, output_path='results/kfold_results.json'):
        """Save k-fold results to JSON"""
        
        results = {
            'k': self.k,
            'fold_results': self.fold_results,
            'overall_results': self.overall_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ K-fold results saved to: {output_path}")


def build_standard_mlp_full(input_dim=4608, num_classes=38):
    """
    Build standard MLP-Full model for k-fold validation
    Must match the training configuration exactly
    """
    
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.BatchNormalization()(inputs)
    
    # Hidden layer 1
    x = layers.Dense(256, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # Hidden layer 2
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def perform_kfold_validation(X, y, k=5, num_classes=38, output_dir='results/'):
    """
    High-level function to perform complete k-fold validation
    
    Args:
        X: Feature matrix
        y: Labels
        k: Number of folds
        num_classes: Number of classes
        output_dir: Directory to save results
        
    Returns:
        Validator object with results
    """
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize validator
    validator = KFoldValidator(k=k)
    
    # Perform validation
    fold_results, overall_results = validator.validate(
        X, y,
        model_builder_fn=build_standard_mlp_full,
        num_classes=num_classes
    )
    
    # Save results
    validator.save_results(os.path.join(output_dir, 'kfold_results.json'))
    
    return validator


# ============================================================================
# QUICK START USAGE
# ============================================================================

if __name__ == '__main__':
    print("✅ K-Fold Validation module loaded successfully")
    print("\nUsage:")
    print("  from kfold_validation import KFoldValidator, perform_kfold_validation")
    print("")
    print("  # Option 1: Using high-level function")
    print("  validator = perform_kfold_validation(X, y, k=5)")
    print("")
    print("  # Option 2: Manual control")
    print("  validator = KFoldValidator(k=5)")
    print("  fold_results, overall = validator.validate(X, y, model_builder_fn)")
    print("  validator.save_results()")
