import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_and_prep_data(filepath):
    """Loads engineered matrices and separates features from target proxy."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing training matrix file asset at: {filepath}")
    
    df = pd.read_csv(filepath)
    # Isolate targets and features, discarding administrative string IDs
    y = df['is_high_risk'].astype(int)
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    return X, y

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Computes standard Basel II banking model verification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics

def run_model_training_sequence():
    print("Initiating Task 5 Credit Risk Training & Tracking Sequence...")
    
    # 1. Establish MLflow experiment project space
    mlflow.set_experiment("Bati_Fintech_Credit_Risk_Scorecard")
    
    # 2. Ingest the engineered artifacts produced in Task 4
    data_path = 'data/processed/model_ready_features.csv'
    X, y = load_and_prep_data(data_path)
    
    # 3. Create reproducible Training and Testing splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Dataset split completed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ==========================================
    # RUN 1: BASELINE LOGISTIC REGRESSION
    # ==========================================
    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        print("\nTraining Baseline Logistic Regression Model...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        
        # Predict outcomes and extraction distribution probabilities
        lr_preds = lr_model.predict(X_test)
        lr_probs = lr_model.predict_proba(X_test)[:, 1]
        
        lr_metrics = evaluate_model_performance(y_test, lr_preds, lr_probs)
        
        # Log Hyperparameters to MLflow tracking panel
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C_regularization", 1.0)
        
        # Log performance metrics
        for metric_name, val in lr_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", val)
            print(f" LR Test {metric_name.capitalize()}: {val:.4f}")
            
        # Log actual model binaries as an artifact
        mlflow.sklearn.log_model(lr_model, "baseline_lr_scorecard")

    # ==========================================
    # RUN 2: RANDOM FOREST WITH HYPERPARAMETER TUNING
    # ==========================================
    with mlflow.start_run(run_name="Tuned_Random_Forest"):
        print("\nTraining Tuned Random Forest Model via Random Search...")
        rf_base = RandomForestClassifier(random_state=42)
        
        # Define hyperparameter grid distribution parameters
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        # Execute hyperparameter tuning optimization search cross-validation loop
        rf_search = RandomizedSearchCV(
            estimator=rf_base, param_distributions=param_dist,
            n_iter=5, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1
        )
        rf_search.fit(X_train, y_train)
        
        best_rf_model = rf_search.best_estimator_
        rf_preds = best_rf_model.predict(X_test)
        rf_probs = best_rf_model.predict_proba(X_test)[:, 1]
        
        rf_metrics = evaluate_model_performance(y_test, rf_preds, rf_probs)
        
        # Log optimized parameters directly from search results
        mlflow.log_param("model_type", "RandomForest")
        for param, value in rf_search.best_params_.items():
            mlflow.log_param(f"opt_{param}", value)
            
        # Log evaluation metrics
        for metric_name, val in rf_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", val)
            print(f" RF Test {metric_name.capitalize()}: {val:.4f}")
            
        # Save optimized model artifact and register it in the official MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=best_rf_model,
            artifact_path="tuned_rf_scorecard",
            registered_model_name="Bati_Fintech_Production_RF_Model"
        )
        print("\nSuccess! Best random forest candidate registered securely in MLflow Model Registry.")

if __name__ == "__main__":
    run_model_training_sequence()