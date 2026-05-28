import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class CustomerRFMAggregator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to aggregate individual transaction logs into 
    customer-level behavioral profiles (Recency, Frequency, Monetary).
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Ensure timestamp is parsed properly
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            # Anchor to the maximum date in the dataset to calculate recency
            max_date = df['TransactionStartTime'].max()
            df['RecencyDays'] = (max_date - df['TransactionStartTime']).dt.days
        else:
            df['RecencyDays'] = 0
            
        # Handle negative bounds by treating amounts as absolute credit exposure
        if 'Amount' in df.columns:
            df['AbsoluteAmount'] = df['Amount'].abs()
        else:
            df['AbsoluteAmount'] = 0

        # Define aggregation logic per unique customer
        agg_dict = {
            'AbsoluteAmount': ['sum', 'mean', 'std'],
            'CustomerId': 'count',
            'RecencyDays': 'min'
        }
        
        # Aggregate up to the customer level
        cust_profile = df.groupby('CustomerId').agg(agg_dict)
        
        # Flatten multi-index column headers
        cust_profile.columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Recency'
        ]
        cust_profile = cust_profile.reset_index()
        
        # Fill standard deviation for single-transaction users with 0
        cust_profile['Std_Amount'] = cust_profile['Std_Amount'].fillna(0)
        
        return cust_profile

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract cyclical time characteristics from raw logs.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        if 'TransactionStartTime' in df.columns:
            timestamps = pd.to_datetime(df['TransactionStartTime'])
            df['Tx_Hour'] = timestamps.dt.hour
            df['Tx_Day'] = timestamps.dt.day
            df['Tx_Month'] = timestamps.dt.month
            df['Tx_Year'] = timestamps.dt.year
        else:
            df['Tx_Hour'] = 12
            df['Tx_Day'] = 15
            df['Tx_Month'] = 6
            df['Tx_Year'] = 2026
        return df

def build_production_pipeline(categorical_features, numerical_features):
    """
    Constructs a repeatable scikit-learn Pipeline object chaining data imputation,
    one-hot encoding, and standard scaling.
    """
    # Numerical execution track
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical execution track
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine parallel execution matrices via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ])
        
    # Final sequential workflow encapsulation
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline