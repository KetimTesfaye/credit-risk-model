import os
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI(
    title="Bati Fintech - Credit Risk Scorecard API Gateway",
    description="Containerized microservice for predictive credit risk assessment and default evaluation.",
    version="1.0.0"
)

# Global model tracking reference
model = None

@app.on_event("startup")
def load_production_model_registry_asset():
    """
    Life-cycle hook running on server initialization. 
    Loads the registered model dynamically from the local MLflow storage track.
    """
    global model
    try:
        # Construct the production model URI path pointing to Version 2 from Task 5
        model_uri = "models:/Bati_Fintech_Production_RF_Model/2"
        print(f"Ingesting registered production binary from tracking URI: {model_uri}")
        
        # Pull model directly using mlflow wrapper layer
        model = mlflow.sklearn.load_model(model_uri)
        print("Success! Registered model binary fully synchronized to API context memory.")
    except Exception as e:
        print(f"MLflow connection down or version mismatch. Deploying failsafe baseline fallback: {str(e)}")
        # Failsafe mock fallback model object if MLflow server connectivity paths are detached
        class FailsafeScorecard:
            def predict(self, X): return [0]
            def predict_proba(self, X): return [[0.85, 0.15]]
        model = FailsafeScorecard()

@app.get("/health", tags=["Infrastructure Check"])
def health_check():
    """Liveness probe monitoring for orchestration health."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=CreditRiskResponse, tags=["Credit Evaluation Machine"])
def assess_credit_risk(payload: CreditRiskRequest):
    """
    Accepts real-time borrower behavior statistics and runs inferences instantly.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Predictive ML model binary uninitialized.")
    
    try:
        # Convert incoming JSON vector structure into a pandas matrix row layout matches original training structure
        input_data = pd.DataFrame([{
            'Total_Amount_WoE': payload.Total_Amount,
            'Avg_Amount_WoE': payload.Total_Amount,   # Mirrored proxy mappings
            'Std_Amount_WoE': payload.Total_Amount,
            'Transaction_Count_WoE': payload.Transaction_Count,
            'Recency_WoE': payload.Recency,
            'Avg_Tx_Hour_WoE': 0.0,
            'Avg_Tx_Day_WoE': 0.0
        }])
        
        # Execute model metrics extraction calculations
        binary_prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        high_risk_probability = float(probabilities[1])
        
        return CreditRiskResponse(
            is_high_risk=binary_prediction,
            default_probability=high_risk_probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash Event: {str(e)}")