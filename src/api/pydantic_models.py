from pydantic import BaseModel, Field

class CreditRiskRequest(BaseModel):
    """
    Strict validation schema for incoming loan borrower evaluation data.
    Maps cleanly to the feature space expected by the model.
    """
    Recency: float = Field(..., description="Standardized Recency Days (WoE or scaled vector)")
    Transaction_Count: float = Field(..., description="Standardized transactional frequency count")
    Total_Amount: float = Field(..., description="Standardized cumulative absolute throughput volume")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Recency": -0.521,
                "Transaction_Count": 1.420,
                "Total_Amount": 0.854
            }
        }

class CreditRiskResponse(BaseModel):
    """
    Output payload format returned to downstream core banking apps.
    """
    is_high_risk: int = Field(..., description="Binary risk marker proxy indicator (1=High Risk, 0=Safe)")
    default_probability: float = Field(..., description="Model confidence score bounding risk percentage")