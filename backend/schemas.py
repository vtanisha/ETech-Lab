from pydantic import BaseModel
from typing import List, Optional

class SHAPValueBase(BaseModel):
    feature_name: str
    contribution: float

class CustomerBase(BaseModel):
    name: str
    age: int
    income: float
    loan_amount: float
    employment_type: str
    credit_score: float
    risk_tier: str
    default_probability: float

class Customer(CustomerBase):
    id: int
    
    class Config:
        from_attributes = True

class CustomerDetail(Customer):
    shap_values: List[SHAPValueBase] = []
    
    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    customer_id: int
    income: Optional[float] = None
    loan_amount: Optional[float] = None

class ChatRequest(BaseModel):
    customer_id: int
    query: str
