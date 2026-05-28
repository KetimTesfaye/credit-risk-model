import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xverse.transformers import MonotonicBinning, WoETransformer

# =====================================================================
# 1. LOG-LEVEL EXTRACTION & PROFILE AGGREGATION LAYER
# =====================================================================
class CustomerRFMAggregator(BaseEstimator, TransformerMixin):
    """
    Transforms multi-row raw log entries into compressed, customer-level
    behavioral vectors (Recency, Frequency, Monetary) to match risk targets.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Parse temporal configurations at the log level
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            max_date = df['TransactionStartTime'].max()
            df['RecencyDays'] = (max_date - df['TransactionStartTime']).dt.days
            
            # Extract basic log-level cyclical time features before grouping
            df['Tx_Hour'] = df['TransactionStartTime'].dt.hour
            df['Tx_Day'] = df['TransactionStartTime'].dt.day
        else:
            df['RecencyDays'], df['Tx_Hour'], df['Tx_Day'] = 0, 12, 15
            
        # Manage negative bounds by treating amounts as absolute credit exposure
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
        
        # Protect against single-transaction users generating NaN standard deviations
        cust_profile['Std_Amount'] = cust_profile['Std_Amount'].fillna(0)
        
        return cust_profile

# =====================================================================
# 2. HULL INTERFACE FOR ENTIRE FITTED PIPELINE ENCAPSULATION
# =====================================================================
class BatiFintechPipeline(BaseEstimator, TransformerMixin):
    """
    Unified hull wrapping the customer aggregation layer, median imputations,
    xverse supervised WoE modeling, and final scaling to export a single 
    fitted pipeline instance that yields model-ready DataFrames.
    """
    def __init__(self):
        # Internalize sequence components natively
        self.aggregator = CustomerRFMAggregator()
        self.imputer = SimpleImputer(strategy='median')
        self.binning = MonotonicBinning()
        self.woe_transformer = WoETransformer()
        self.scaler = StandardScaler()
        
        # Storage parameters for banking credit audits
        self.information_values_df = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X, y_dict):
        """
        Fits all feature processing steps sequentially.
        
        Parameters:
        X (DataFrame): Raw multi-row transaction data logs.
        y_dict (dict): A mapping dictionary where keys are unique CustomerIds
                       and values are binary targets (0 for Good, 1 for Bad).
        """
        # Step A: Transform raw logs to compressed customer summaries
        cust_df = self.aggregator.transform(X)
        
        # Step B: Map the supervised customer targets accurately to the compressed matrix rows
        customer_ids = cust_df['CustomerId']
        y_vector = customer_ids.map(y_dict).fillna(0).astype(int)
        
        # Isolate training features by removing non-predictive administrative string IDs
        X_features = cust_df.drop(columns=['CustomerId'])
        self.feature_names_ = X_features.columns.tolist()
        
        # Step C: Fit numeric baseline imputation weights
        X_imputed = self.imputer.fit_transform(X_features)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_)
        
        # Step D: Fit monotonic binning boundaries via supervised target alignment
        self.binning.fit(X_imputed_df, y_vector)
        X_binned = self.binning.transform(X_imputed_df)
        
        # Step E: Fit Basel II compliant Weight of Evidence matrices
        self.woe_transformer.fit(X_binned, y_vector)
        X_woe = self.woe_transformer.transform(X_binned)
        
        # Capture internal operational audit table for scorecard transparency
        self.information_values_df = self.woe_transformer.iv_df
        
        # Step F: Fit final standard scaling parameters
        self.scaler.fit(X_woe)
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Applies pre-calculated processing thresholds to produce a clean,
        scaled, fully engineered DataFrame artifact ready for predictive models.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted via structured target dictionary before transformation.")
            
        cust_df = self.aggregator.transform(X)
        customer_ids = cust_df['CustomerId']
        X_features = cust_df.drop(columns=['CustomerId'])
        
        X_imputed = self.imputer.transform(X_features)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_)
        
        X_binned = self.binning.transform(X_imputed_df)
        X_woe = self.woe_transformer.transform(X_binned)
        
        X_scaled = self.scaler.transform(X_woe)
        
        # Re-assemble scaled metrics into a clean DataFrame retaining proper target indexing strings
        model_ready_df = pd.DataFrame(X_scaled, columns=[f"{col}_WoE" for col in X_woe.columns])
        model_ready_df.insert(0, 'CustomerId', customer_ids)
        
        return model_ready_df

# =====================================================================
# 3. GLOBAL VARIABLE EXPORT DELIVERABLE
# =====================================================================
# Defining the globally accessible fitted pipeline object instance required by deliverables
fitted_production_pipeline = BatiFintechPipeline()

# =====================================================================
# 4. EXECUTION VERIFICATION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("Initiating automated pipeline execution verification sequence...")
    
    # 1. Simulating an unlabelled raw Xente transaction stream matching your EDA distributions
    np.random.seed(42)
    n_logs = 1500
    mock_raw_logs = pd.DataFrame({
        'CustomerId': [f'Cust_{i}' for i in np.random.randint(1001, 1150, n_logs)],
        'Amount': np.random.uniform(-1500, 15000, n_logs),
        'TransactionStartTime': pd.date_range(start='2026-01-01', periods=n_logs, freq='15min')
    })
    
    # 2. Formulating the target proxy dictionary mapping for training target supervision
    # (0 = Compliant Payer, 1 = Delinquent/Default Proxy)
    unique_borrowers = mock_raw_logs['CustomerId'].unique()
    mock_target_outcomes = np.random.choice([0, 1], size=len(unique_borrowers), p=[0.93, 0.07])
    production_target_map = dict(zip(unique_borrowers, mock_target_outcomes))
    
    print(f"Ingested {len(mock_raw_logs)} raw transaction records across {len(unique_borrowers)} unique customers.")
    print("Fitting global pipeline constraints onto training dependencies...")
    
    # Fit the standalone master object globally
    fitted_production_pipeline.fit(mock_raw_logs, production_target_map)
    
    # Extract the clean operational output artifact
    print("Transforming incoming datasets to model-ready continuous score tracks...")
    model_ready_output = fitted_production_pipeline.transform(mock_raw_logs)
    
    # Display information metrics for banking auditing verification
    print("\n" + "="*55)
    print("   BASEL II RISK SELECTION LOG (INFORMATION VALUE)   ")
    print("="*55)
    print(fitted_production_pipeline.information_values_df.to_string(index=False))
    print("="*55)
    
    # Save the transformed output matrices to your persistent disk configuration
    processed_directory = 'data/processed'
    if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)
        
    output_filepath = os.path.join(processed_directory, 'model_ready_features.csv')
    model_ready_output.to_csv(output_filepath, index=False)
    
    print(f"\nExecution successful! Sample output head shape: {model_ready_output.shape}")
    print(f"Artifact locked securely within production volume file path: {output_filepath}")