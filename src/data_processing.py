import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# =====================================================================
# PHASE 1 & 2: CUSTOM FEATURE EXTRACTION & AGGREGATION
# =====================================================================

class CompleteFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Combines log-level parsing, temporal feature extraction, and 
    customer-level RFM aggregations to prevent matrix shape mismatch.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # 1. Parse Temporal Fields at Log Level
        if 'TransactionStartTime' in df.columns:
            timestamps = pd.to_datetime(df['TransactionStartTime'])
            df['Tx_Hour'] = timestamps.dt.hour
            df['Tx_Day'] = timestamps.dt.day
            df['Tx_Month'] = timestamps.dt.month
            df['Tx_Year'] = timestamps.dt.year
            max_date = timestamps.max()
            df['RecencyDays'] = (max_date - timestamps).dt.days
        else:
            df['Tx_Hour'], df['Tx_Day'], df['Tx_Month'], df['Tx_Year'], df['RecencyDays'] = 12, 15, 6, 2026, 0

        # Handle negative amount bounds via absolute credit exposure tracking
        if 'Amount' in df.columns:
            df['AbsoluteAmount'] = df['Amount'].abs()
        else:
            df['AbsoluteAmount'] = 0

        # 2. Execute Aggregation Grouping per Customer
        agg_dict = {
            'AbsoluteAmount': ['sum', 'mean', 'std'],
            'RecencyDays': 'min',
            'Tx_Hour': 'mean',
            'Tx_Day': 'mean',
            'ProviderId': lambda x: x.mode()[0] if not x.empty else 'Unknown',
            'ProductCategory': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }
        
        cust_profile = df.groupby('CustomerId').agg(agg_dict)
        cust_profile.columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Recency', 
            'Avg_Tx_Hour', 'Avg_Tx_Day', 'Primary_Provider', 'Primary_Category'
        ]
        cust_profile = cust_profile.reset_index()
        cust_profile['Std_Amount'] = cust_profile['Std_Amount'].fillna(0)
        
        return cust_profile

# =====================================================================
# PHASE 3: BASEL II WEIGHT OF EVIDENCE (WoE) ENCODER
# =====================================================================

class BaselWoEEncoder(BaseEstimator, TransformerMixin):
    """
    Applies custom Weight of Evidence (WoE) transformation to binned/categorical
    features to strictly ensure Basel II compliance and auditability.
    """
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.woe_maps = {}
        
    def fit(self, X, y):
        df = X.copy()
        df['Target'] = y  # Expects binary target: 0 for Good, 1 for Bad
        
        total_goods = (df['Target'] == 0).sum()
        total_bads = (df['Target'] == 1).sum()
        
        # Avoid division-by-zero errors in empty edge datasets
        total_goods = total_goods if total_goods > 0 else 1
        total_bads = total_bads if total_bads > 0 else 1

        for col in self.categorical_cols:
            self.woe_maps[col] = {}
            # Group goods and bads per unique category value
            stats = df.groupby(col)['Target'].agg(
                goods=lambda x: (x == 0).sum(),
                bads=lambda x: (x == 1).sum()
            )
            
            for val, row in stats.iterrows():
                # Apply Laplace smoothing to stabilize zero counts
                g_pct = (row['goods'] + 0.5) / total_goods
                b_pct = (row['bads'] + 0.5) / total_bads
                
                # Compute WoE equation
                self.woe_maps[col][val] = np.log(g_pct / b_pct)
                
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.categorical_cols:
            # Replace raw strings/bins with continuous auditable WoE metrics
            default_woe = 0.0
            df[f'{col}_WoE'] = df[col].map(self.woe_maps[col]).fillna(default_woe)
            df = df.drop(columns=[col])
        return df

# =====================================================================
# PIPELINE GENERATION FUNCTION
# =====================================================================

def create_production_pipeline():
    """
    Constructs the final, end-to-end production-ready feature processing pipeline.
    """
    pipeline = Pipeline(steps=[
        ('engineer', CompleteFeatureEngineer()),
        ('woe_encoder', BaselWoEEncoder(categorical_cols=['Primary_Provider', 'Primary_Category'])),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

# =====================================================================
# EXECUTION ENTRYPOINT (AUTOMATING THE DATA OUTPUT LOCK)
# =====================================================================

if __name__ == "__main__":
    print("Initializing Task 3: Feature Engineering Pipeline Automation...")
    
    # Path configuration setup
    raw_data_path = 'data/raw/data.csv'  # Adjust to match your filename
    processed_dir = 'data/processed'
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # For development verification: Mock data if actual file is missing
    try:
        raw_df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Raw data file not found at {raw_data_path}. Simulating template pipeline sequence...")
        # Simulating dummy Xente structured data matching your EDA observations
        raw_df = pd.DataFrame({
            'CustomerId': [f'Cust_{i}' for i in np.random.randint(1, 100, 1000)],
            'Amount': np.random.uniform(-500, 5000, 1000),
            'TransactionStartTime': pd.date_range(start='2026-01-01', periods=1000, freq='h'),
            'ProviderId': [f'ProviderId_{i}' for i in np.random.choice([4, 6, 1, 2], 1000)],
            'ProductCategory': [np.random.choice(['financial_services', 'airtime', 'utility_bill']) for _ in range(1000)]
        })

    # Generate the default RFM target proxy metrics directly
    print("Formulating RFM default proxy classification parameters...")
    # Pre-calculate a placeholder target using your future target classification strategy
    # High volume spending + low recency = Good (0), High recency + low activity = Bad (1)
    # We create a dummy column here to feed into the fit method for supervised WoE
    np.random.seed(42)
    mock_target_per_customer = np.random.choice([0, 1], size=raw_df['CustomerId'].nunique(), p=[0.95, 0.05])
    cust_id_map = {cid: target for cid, target in zip(raw_df['CustomerId'].unique(), mock_target_per_customer)}
    
    # Extract structural engineering framework
    pipeline_obj = create_production_pipeline()
    
    # Isolate customer level step dependencies to pass the target vector safely
    transformed_features = pipeline_obj.steps[0][1].transform(raw_df)
    y_vector = transformed_features['CustomerId'].map(cust_id_map).fillna(0).astype(int)
    
    # Fit downstream parameters (WoE scales, standardizations)
    print("Fitting pipeline transformations and extracting predictive Basel II signals...")
    pipeline_obj.steps[1][1].fit(transformed_features, y_vector)
    
    final_engineered_matrix = pipeline_obj.transform(raw_df)
    
    # Save the transformed output file artifact matrix
    output_path = os.path.join(processed_dir, 'model_ready_features.csv')
    pd.DataFrame(final_engineered_matrix).to_csv(output_path, index=False)
    print(f"Task 3 pipeline complete! Clean model-ready matrices saved securely to: {output_path}")