"""
LightGBM Re-Ranking Model Training
Train a classification model to distinguish good candidates from bad ones
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for training by splitting features and labels.
    
    Args:
        df: DataFrame with features and target
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, y_train, y_val, feature_names
    """
    print("=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    
    # Identify feature columns (exclude metadata and target)
    metadata_cols = ['prefix', 'candidate', 'target']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Metadata columns: {metadata_cols}")
    
    # Extract features and labels
    X = df[feature_cols]
    y = df['target']
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Positive samples: {(y == 1).sum():,} ({100 * (y == 1).mean():.2f}%)")
    print(f"Negative samples: {(y == 0).sum():,} ({100 * (y == 0).mean():.2f}%)")
    
    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.any():
        print(f"\n⚠ Warning: Missing values detected:")
        print(missing_counts[missing_counts > 0])
        print("Filling missing values with 0...")
        X = X.fillna(0)
    
    # Split data - stratify to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\n{'Split':<15} | {'Samples':<10} | {'Positive':<10} | {'Negative':<10}")
    print("-" * 80)
    print(f"{'Training':<15} | {len(X_train):<10,} | {(y_train == 1).sum():<10,} | {(y_train == 0).sum():<10,}")
    print(f"{'Validation':<15} | {len(X_val):<10,} | {(y_val == 1).sum():<10,} | {(y_val == 0).sum():<10,}")
    
    return X_train, X_val, y_train, y_val, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Train a LightGBM classification model for re-ranking.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature column names
    
    Returns:
        Trained LightGBM model
    """
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM MODEL")
    print("=" * 80)
    
    # Define LightGBM parameters
    params = {
        # Task configuration
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        
        # Learning parameters
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,  # No limit
        
        # Regularization
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        
        # Other
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key:<25} : {value}")
    
    # Initialize model
    print("\nInitializing LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=500,  # Will use early stopping
        **params
    )
    
    # Train with early stopping
    print("\nTraining model with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'valid'],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"\n✓ Training completed!")
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Best score: {model.best_score_['valid']['auc']:.4f}")
    
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, feature_names):
    """
    Comprehensive evaluation of the trained model.
    
    Args:
        model: Trained LightGBM model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = {}
    
    # Make predictions
    print("\nGenerating predictions...")
    
    # Probability predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Binary predictions (threshold = 0.5)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # ========================================================================
    # 1. AUC Scores
    # ========================================================================
    
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print("\n" + "-" * 80)
    print("AUC SCORES (PRIMARY METRIC)")
    print("-" * 80)
    print(f"Training AUC:   {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Difference:     {abs(train_auc - val_auc):.4f}")
    
    results['train_auc'] = train_auc
    results['val_auc'] = val_auc
    
    # ========================================================================
    # 2. Classification Metrics
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("CLASSIFICATION METRICS")
    print("-" * 80)
    
    # Training metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Validation metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"\n{'Metric':<15} | {'Training':<12} | {'Validation':<12}")
    print("-" * 80)
    print(f"{'Accuracy':<15} | {train_acc:<12.4f} | {val_acc:<12.4f}")
    print(f"{'Precision':<15} | {train_prec:<12.4f} | {val_prec:<12.4f}")
    print(f"{'Recall':<15} | {train_rec:<12.4f} | {val_rec:<12.4f}")
    print(f"{'F1-Score':<15} | {train_f1:<12.4f} | {val_f1:<12.4f}")
    
    results.update({
        'train_accuracy': train_acc, 'val_accuracy': val_acc,
        'train_precision': train_prec, 'val_precision': val_prec,
        'train_recall': train_rec, 'val_recall': val_rec,
        'train_f1': train_f1, 'val_f1': val_f1
    })
    
    # ========================================================================
    # 3. Confusion Matrix
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX (VALIDATION)")
    print("-" * 80)
    
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\n                Predicted")
    print(f"              Neg     Pos")
    print(f"Actual  Neg   {cm[0][0]:<7,} {cm[0][1]:<7,}")
    print(f"        Pos   {cm[1][0]:<7,} {cm[1][1]:<7,}")
    
    # ========================================================================
    # 4. Classification Report
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("DETAILED CLASSIFICATION REPORT (VALIDATION)")
    print("-" * 80)
    print("\n" + classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))
    
    # ========================================================================
    # 5. Feature Importance
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("TOP 20 FEATURE IMPORTANCES")
    print("-" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'Rank':<6} | {'Feature':<40} | {'Importance':<12}")
    print("-" * 80)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{feature_importance.index.get_loc(idx) + 1:<6} | {row['feature']:<40} | {row['importance']:<12.2f}")
    
    results['feature_importance'] = feature_importance
    
    # ========================================================================
    # 6. Prediction Distribution Analysis
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("PREDICTION PROBABILITY DISTRIBUTION")
    print("-" * 80)
    
    print("\nValidation Set:")
    print(f"  Positive class (target=1):")
    print(f"    Mean probability: {y_val_pred_proba[y_val == 1].mean():.4f}")
    print(f"    Median probability: {np.median(y_val_pred_proba[y_val == 1]):.4f}")
    print(f"    Std probability: {y_val_pred_proba[y_val == 1].std():.4f}")
    
    print(f"  Negative class (target=0):")
    print(f"    Mean probability: {y_val_pred_proba[y_val == 0].mean():.4f}")
    print(f"    Median probability: {np.median(y_val_pred_proba[y_val == 0]):.4f}")
    print(f"    Std probability: {y_val_pred_proba[y_val == 0].std():.4f}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(model, X_val, y_val, y_val_pred_proba, feature_importance_df):
    """
    Create and save evaluation visualizations.
    
    Args:
        model: Trained model
        X_val, y_val: Validation data
        y_val_pred_proba: Predicted probabilities
        feature_importance_df: Feature importance DataFrame
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ========================================================================
    # 1. ROC Curve
    # ========================================================================
    
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
    auc = roc_auc_score(y_val, y_val_pred_proba)
    
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 0].set_title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. Feature Importance (Top 15)
    # ========================================================================
    
    top_features = feature_importance_df.head(15)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='steelblue')
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'], fontsize=9)
    axes[0, 1].set_xlabel('Importance', fontsize=12)
    axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # ========================================================================
    # 3. Prediction Distribution
    # ========================================================================
    
    axes[1, 0].hist(y_val_pred_proba[y_val == 1], bins=50, alpha=0.6, 
                    label='Positive (target=1)', color='green', edgecolor='black')
    axes[1, 0].hist(y_val_pred_proba[y_val == 0], bins=50, alpha=0.6, 
                    label='Negative (target=0)', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Predicted Probability', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='upper center', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
    
    # ========================================================================
    # 4. Confusion Matrix Heatmap
    # ========================================================================
    
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1, 1],
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    axes[1, 1].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 1].set_ylabel('True Label', fontsize=12)
    axes[1, 1].set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    viz_file = 'model_evaluation.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {viz_file}")
    
    plt.close()


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, feature_names, results, model_path='reranking_model.pkl'):
    """
    Save trained model and metadata to disk.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        results: Dictionary of evaluation metrics
        model_path: Path to save the model
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'LightGBM',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'best_iteration': model.best_iteration_,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in results.items() if k != 'feature_importance'}
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Model metadata saved to: {metadata_path}")
    
    # Save feature importance
    feature_importance_df = results['feature_importance']
    importance_path = model_path.replace('.pkl', '_feature_importance.csv')
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to: {importance_path}")
    
    # Save LightGBM native format (optional, for better compatibility)
    lgb_model_path = model_path.replace('.pkl', '.txt')
    model.booster_.save_model(lgb_model_path)
    print(f"✓ LightGBM native model saved to: {lgb_model_path}")
    
    return metadata


def load_model(model_path='reranking_model.pkl'):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from: {model_path}")
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM RE-RANKING MODEL TRAINING")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 80)
    
    data_file = 'reranking_training_data.parquet'
    print(f"\nLoading data from: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"✓ Data loaded: {df.shape}")
    
    # ========================================================================
    # STEP 2: PREPARE DATA
    # ========================================================================
    
    X_train, X_val, y_train, y_val, feature_names = prepare_data(
        df, 
        test_size=0.2, 
        random_state=42
    )
    
    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    
    model = train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names)
    
    # ========================================================================
    # STEP 4: EVALUATE MODEL
    # ========================================================================
    
    results = evaluate_model(model, X_train, y_train, X_val, y_val, feature_names)
    
    # ========================================================================
    # STEP 5: CREATE VISUALIZATIONS
    # ========================================================================
    
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    create_visualizations(model, X_val, y_val, y_val_pred_proba, results['feature_importance'])
    
    # ========================================================================
    # STEP 6: SAVE MODEL
    # ========================================================================
    
    metadata = save_model(model, feature_names, results, model_path='reranking_model.pkl')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} | {'Value':<15}")
    print("-" * 80)
    print(f"{'Validation AUC':<30} | {results['val_auc']:<15.4f}")
    print(f"{'Validation Accuracy':<30} | {results['val_accuracy']:<15.4f}")
    print(f"{'Validation F1-Score':<30} | {results['val_f1']:<15.4f}")
    print(f"{'Best Iteration':<30} | {model.best_iteration_:<15}")
    print(f"{'Number of Features':<30} | {len(feature_names):<15}")
    
    print("\nOutput Files:")
    print("  • reranking_model.pkl - Trained model (pickle)")
    print("  • reranking_model.txt - LightGBM native format")
    print("  • reranking_model_metadata.json - Model metadata")
    print("  • reranking_model_feature_importance.csv - Feature importances")
    print("  • model_evaluation.png - Evaluation visualizations")
    
    print("\nNext Steps:")
    print("  1. Review model evaluation metrics and visualizations")
    print("  2. If AUC is good (>0.80), proceed to inference")
    print("  3. Use the saved model for re-ranking candidates in production")
