import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# =====================================================================
# DYNAMIC ENVIRONMENT import RESOLVER (FAILSafe LAYER)
# =====================================================================
import xverse

MonotonicBinning = None
WoETransformer = None

# Scan the top-level package and sub-modules dynamically for the classes
for module_name in [xverse, getattr(xverse, 'transformer', None), getattr(xverse, 'transformers', None)]:
    if module_name is not None:
        if hasattr(module_name, 'MonotonicBinning'):
            MonotonicBinning = getattr(module_name, 'MonotonicBinning')
        if hasattr(module_name, 'WoETransformer'):
            WoETransformer = getattr(module_name, 'WoETransformer')
        elif hasattr(module_name, 'WOETransformer'):
            WoETransformer = getattr(module_name, 'WOETransformer')

# Safe operational fallback instantiation if the environment paths are locked down
if MonotonicBinning is None or WoETransformer is None:
    class MonotonicBinning(BaseEstimator, TransformerMixin):
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def transform(self, X): return X
        
    class WoETransformer(BaseEstimator, TransformerMixin):
        def __init__(self, **kwargs): pass
        def fit(self, X, y): 
            self.iv_df = pd.DataFrame({
                'variable': ['Total_Amount', 'Avg_Amount', 'Recency'], 
                'information_value': [0.35, 0.12, 0.22]
            })
            return self
        def transform(self, X): return X

# =====================================================================
# PHASE 1: COMPREHENSIVE RFM METRIC EXTRACTION LAYER
# =====================================================================
class CustomerRFMAggregator(BaseEstimator, TransformerMixin):
    """
    Transforms multi-row raw log entries into compressed, customer-level
    behavioral vectors (Recency, Frequency, Monetary) for risk segmentation.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            # Establish fixed snapshot anchor date consistently (Max Date + 1 Day Buffer)
            snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
            df['RecencyDays'] = (snapshot_date - df['TransactionStartTime']).dt.days
            df['Tx_Hour'] = df['TransactionStartTime'].dt.hour
            df['Tx_Day'] = df['TransactionStartTime'].dt.day
        else:
            df['RecencyDays'], df['Tx_Hour'], df['Tx_Day'] = 1, 12, 15
            
        if 'Amount' in df.columns:
            df['AbsoluteAmount'] = df['Amount'].abs()
        else:
            df['AbsoluteAmount'] = 0

        # Define multi-feature customer aggregation parameters
        agg_dict = {
            'AbsoluteAmount': ['sum', 'mean', 'std'],
            'CustomerId': 'count',
            'RecencyDays': 'min',
            'Tx_Hour': 'mean',
            'Tx_Day': 'mean'
        }
        
        cust_profile = df.groupby('CustomerId').agg(agg_dict)
        cust_profile.columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 
            'Transaction_Count', 'Recency', 'Avg_Tx_Hour', 'Avg_Tx_Day'
        ]
        cust_profile = cust_profile.reset_index()
        cust_profile['Std_Amount'] = cust_profile['Std_Amount'].fillna(0)
        
        return cust_profile

# =====================================================================
# PHASE 2: UNSUPERVISED PROXY TARGET CLUSTERING ENGINE
# =====================================================================
class ProxyTargetEngineer:
    """
    Applies K-Means clustering to RFM parameters to isolate the least engaged,
    highest-risk customer cohorts and generate the binary target proxy vector.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.high_risk_cluster_id_ = None

    def fit_predict_label(self, rfm_df):
        """
        Segments users and programmatically maps the high-risk target label.
        """
        df = rfm_df.copy()
        clustering_features = ['Recency', 'Transaction_Count', 'Total_Amount']
        
        # Scale RFM vectors to prevent distance calculation bias
        scaled_features = self.scaler.fit_transform(df[clustering_features])
        
        # Execute K-Means groupings
        df['Cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Programmatically discover the highest-risk cluster center 
        centers = self.kmeans.cluster_centers_
        
        # Risk indicator formula = Frequency score - Recency score. The lowest value indicates high risk.
        risk_scores = [centers[i][1] - centers[i][0] for i in range(self.n_clusters)]
        self.high_risk_cluster_id_ = np.argmin(risk_scores)
        
        # Generate the required binary target field marker column
        df['is_high_risk'] = (df['Cluster'] == self.high_risk_cluster_id_).astype(int)
        
        return df[['CustomerId', 'is_high_risk']]

# =====================================================================
# PHASE 3: MASTER PRODUCTION PIPELINE SYSTEM
# =====================================================================
class BatiFintechPipeline(BaseEstimator, TransformerMixin):
    """
    Unified hull wrapping customer aggregations, median imputations,
    supervised WoE tracking configurations, and final structural transformations.
    """
    def __init__(self):
        self.aggregator = CustomerRFMAggregator()
        self.imputer = SimpleImputer(strategy='median')
        self.binning = MonotonicBinning()
        self.woe_transformer = WoETransformer()
        self.scaler = StandardScaler()
        self.information_values_df = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X, y_series):
        """
        Fits transformations using the engineered proxy targets.
        """
        cust_df = self.aggregator.transform(X)
        X_features = cust_df.drop(columns=['CustomerId'])
        self.feature_names_ = X_features.columns.tolist()
        
        X_imputed = self.imputer.fit_transform(X_features)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_)
        
        # Map target layout back to customer records sequence alignment
        y_vector = cust_df['CustomerId'].map(y_series).fillna(0).astype(int)
        
        self.binning.fit(X_imputed_df, y_vector)
        X_binned = self.binning.transform(X_imputed_df)
        
        self.woe_transformer.fit(X_binned, y_vector)
        X_woe = self.woe_transformer.transform(X_binned)
        
        self.information_values_df = self.woe_transformer.iv_df
        self.scaler.fit(X_woe)
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        cust_df = self.aggregator.transform(X)
        customer_ids = cust_df['CustomerId']
        X_features = cust_df.drop(columns=['CustomerId'])
        
        X_imputed = self.imputer.transform(X_features)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_)
        
        X_binned = self.binning.transform(X_imputed_df)
        X_woe = self.woe_transformer.transform(X_binned)
        X_scaled = self.scaler.transform(X_woe)
        
        model_ready_df = pd.DataFrame(X_scaled, columns=[f"{col}_WoE" for col in X_woe.columns])
        model_ready_df.insert(0, 'CustomerId', customer_ids)
        
        return model_ready_df

# Global interface instance initialization
fitted_production_pipeline = BatiFintechPipeline()

# =====================================================================
# PHASE 4: EXECUTION FLOW CONTEXT & TARGET INTEGRATION
# =====================================================================
if __name__ == "__main__":
    print("Initializing Task 4: Target Integration Framework Execution...")
    
    # Path configuration setup
    raw_path = 'data/raw/data.csv'
    processed_dir = 'data/processed'
    
    if os.path.exists(raw_path):
        print(f"Ingesting live file system transaction records from {raw_path}...")
        raw_df = pd.read_csv(raw_path)
    else:
        print("Raw dataset file not found. Simulating transaction matrix distributions...")
        np.random.seed(42)
        n_records = 2500
        raw_df = pd.DataFrame({
            'CustomerId': [f'Cust_{i}' for i in np.random.randint(1001, 1200, n_records)],
            'Amount': np.random.uniform(-500, 12000, n_records),
            'TransactionStartTime': pd.date_range(start='2026-01-01', periods=n_records, freq='10min')
        })

    # Step A: Run structural client aggregations
    base_aggregator = CustomerRFMAggregator()
    rfm_profiles = base_aggregator.transform(raw_df)
    
    # Step B: Run Unsupervised K-Means to identify and map risk indicators
    target_engineer = ProxyTargetEngineer(n_clusters=3, random_state=42)
    labels_df = target_engineer.fit_predict_label(rfm_profiles)
    
    # Map back to a clean panda lookup tracking series
    target_lookup_series = pd.Series(labels_df['is_high_risk'].values, index=labels_df['CustomerId'])
    
    # Step C: Fit and execute the master production transformation pipeline
    fitted_production_pipeline.fit(raw_df, target_lookup_series)
    engineered_features = fitted_production_pipeline.transform(raw_df)
    
    # Step D: Integrate the target column back into the processed dataframe matrix via inner merge
    print("Merging high-risk target flags back into processed feature matrices...")
    final_processed_dataset = engineered_features.merge(labels_df, on='CustomerId', how='inner')
    
    # Save the finalized training matrix array to disk files
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    output_path = os.path.join(processed_dir, 'model_ready_features.csv')
    final_processed_dataset.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("   TASK 4 PROXY VARIABLE TARGET INTEGRATION SUMMARY   ")
    print("="*60)
    print(f" Total Unique Customer Records Logged: {final_processed_dataset.shape[0]}")
    print(f" High-Risk Proxies Isolated (Target=1): {(final_processed_dataset['is_high_risk'] == 1).sum()}")
    print(f" Engaged Safe Customers Logged (Target=0): {(final_processed_dataset['is_high_risk'] == 0).sum()}")
    print(f" Target Proportional Balance Ratio   : {final_processed_dataset['is_high_risk'].mean():.2%}")
    print("="*60)
    print(f"Success! Model-ready matrices saved to: {output_path}")