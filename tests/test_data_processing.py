import pytest
import pandas as pd
import numpy as np
from src.data_processing import CustomerRFMAggregator, ProxyTargetEngineer

def test_customer_rfm_aggregator_column_output():
    """Verifies that the customer log aggregation layer returns all required baseline columns."""
    # 1. Setup mock multi-row transaction dataset logs
    mock_logs = pd.DataFrame({
        'CustomerId': ['Cust_X', 'Cust_X', 'Cust_Y'],
        'Amount': [250.0, -50.0, 1500.0],
        'TransactionStartTime': pd.to_datetime(['2026-01-01 10:00:00', '2026-01-02 11:00:00', '2026-01-03 12:00:00'])
    })
    
    aggregator = CustomerRFMAggregator()
    processed_df = aggregator.transform(mock_logs)
    
    # 2. Define the exact architectural columns expected by the downstream pipeline
    expected_columns = [
        'CustomerId', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 
        'Transaction_Count', 'Recency', 'Avg_Tx_Hour', 'Avg_Tx_Day'
    ]
    
    # 3. Validation assertions
    assert isinstance(processed_df, pd.DataFrame), "Output must be a standard pandas DataFrame object."
    assert processed_df.shape[0] == 2, "Multi-row logs did not compress down correctly to unique customer vectors."
    for col in expected_columns:
        assert col in processed_df.columns, f"Required column payload structural asset missing: {col}"

def test_proxy_target_engineer_binary_bounds():
    """Ensures that the unsupervised clustering engine strictly yields uniform binary assignments (0 or 1)."""
    # Setup mock pre-aggregated RFM database profiles
    mock_rfm = pd.DataFrame({
        'CustomerId': [f'Cust_{i}' for i in range(10)],
        'Recency': [1, 2, 28, 3, 30, 2, 1, 29, 4, 2],
        'Transaction_Count': [15, 12, 1, 14, 1, 11, 19, 2, 10, 13],
        'Total_Amount': [5000, 4200, 50, 6100, 20, 3900, 8900, 110, 2200, 4100]
    })
    
    engineer = ProxyTargetEngineer(n_clusters=3, random_state=42)
    labels_df = engineer.fit_predict_label(mock_rfm)
    
    # Validation assertions
    assert 'is_high_risk' in labels_df.columns, "The engineered binary target column 'is_high_risk' was not generated."
    unique_labels = labels_df['is_high_risk'].unique()
    
    # Ensure that entries fall within the required binary bounds
    for label in unique_labels:
        assert label in [0, 1], f"Mathematical assignment leaking outside binary constraints: {label}"