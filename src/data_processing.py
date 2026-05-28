import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xverse.transformers import MonotonicBinning, WoETransformer

# =====================================================================
# CUSTOM CUSTOMER AGGREGATION LAYER
# =====================================================================
class CustomerRFMAggregator(BaseEstimator, TransformerMixin):
    """
    Transforms individual raw log data rows into compressed, 
    customer-level aggregated profiles to isolate borrower behaviors.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            max_date = df['TransactionStartTime'].max()
            df['RecencyDays'] = (max_date - df['TransactionStartTime']).dt.days
        else:
            df['RecencyDays'] = 0
            
        if 'Amount' in df.columns:
            df['AbsoluteAmount'] = df['Amount'].abs()
        else:
            df['AbsoluteAmount'] = 0

        agg_dict = {
            'AbsoluteAmount': ['sum', 'mean', 'std'],
            'CustomerId': 'count',
            'RecencyDays': 'min'
        }
        
        cust_profile = df.groupby('CustomerId').agg(agg_dict)
        cust_profile.columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Recency'
        ]
        cust_profile = cust_profile.reset_index()
        cust_profile['Std_Amount'] = cust_profile['Std_Amount'].fillna(0)
        
        return cust_profile

# =====================================================================
# FINAL COMPLIANT BASEL II PIPELINE CREATION
# =====================================================================
def create_xverse_risk_pipeline():
    """
    Constructs an end-to-end processing execution block that leverages xverse
    to automatically execute monotonic binning and WoE transformations.
    """
    pipeline = Pipeline(steps=[
        ('rfm_aggregator', CustomerRFMAggregator()),
        ('imputer', SimpleImputer(strategy='median')),
        # MonotonicBinning automatically determines optimal mathematical splits for numerical columns
        ('binning', MonotonicBinning()),
        # WoETransformer automatically calculates weights based on supervised target outcomes
        ('woe_transform', WoETransformer()),
        ('scaler', StandardScaler())
    ])
    return pipeline

# =====================================================================
# EXECUTION WORKFLOW ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("Executing Task 3: Processing data via xverse WoE automation...")
    
    # Mock data simulation for local validation
    np.random.seed(42)
    mock_raw_data = pd.DataFrame({
        'CustomerId': [f'Cust_{i}' for i in np.random.randint(1, 50, 500)],
        'Amount': np.random.uniform(-1000, 10000, 500),
        'TransactionStartTime': pd.date_range(start='2026-01-01', periods=500, freq='h')
    })
    
    # Generate the target vector at the unique customer level 
    # (0 = Low Risk Borrower, 1 = High Risk Default Default)
    unique_customers = mock_raw_data['CustomerId'].unique()
    mock_targets = np.random.choice([0, 1], size=len(unique_customers), p=[0.92, 0.08])
    target_series = pd.Series(mock_targets, index=unique_customers)
    
    # Build the production setup
    xverse_pipeline = create_xverse_risk_pipeline()
    
    # Extract features to align indices cleanly for the supervised target
    features_step1 = xverse_pipeline.steps[0][1].transform(mock_raw_data)
    y_train = features_step1['CustomerId'].map(target_series).fillna(0).astype(int)
    
    # Drop non-predictive identifiers before passing to xverse matrices
    X_train = features_step1.drop(columns=['CustomerId'])
    
    # Fit the structural binning and compute final WoE values
    print("Fitting Monotonic Binning boundaries and extracting IV parameters...")
    xverse_pipeline.steps[1][1].fit(X_train) # Fit imputer
    
    # Slice pipeline to run binning and WoE transformations manually to showcase extraction
    X_imputed = xverse_pipeline.steps[1][1].transform(X_train)
    
    # Run xverse fit sequences
    clf_binning = xverse_pipeline.steps[2][1].fit(X_imputed, y_train)
    X_binned = clf_binning.transform(X_imputed)
    
    clf_woe = xverse_pipeline.steps[3][1].fit(X_binned, y_train)
    X_woe = clf_woe.transform(X_binned)
    
    # Display calculated Information Values (IV) for Information Quality verification
    print("\n=== FEATURE SELECTION SCREENING (INFORMATION VALUE) ===")
    print(clf_woe.iv_df)
    
    # Final output scaling lock
    final_output = xverse_pipeline.steps[4][1].fit_transform(X_woe)
    
    # Output file lock verification
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    pd.DataFrame(final_output).to_csv(f'{processed_dir}/model_ready_features.csv', index=False)
    print(f"\nPipeline execution successful! Production artifacts locked into data/processed/ folder.")