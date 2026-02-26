import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import database, models, schemas

app = FastAPI(title="CrediRisk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database layout
models.Base.metadata.create_all(bind=database.engine)

@app.get("/customers", response_model=list[schemas.Customer])
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    customers = db.query(models.Customer).offset(skip).limit(limit).all()
    return customers

@app.get("/customers/{customer_id}", response_model=schemas.CustomerDetail)
def read_customer(customer_id: int, db: Session = Depends(database.get_db)):
    customer = db.query(models.Customer).filter(models.Customer.id == customer_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.post("/predict/whatif")
def predict_whatif(request: schemas.PredictionRequest, db: Session = Depends(database.get_db)):
    customer = db.query(models.Customer).filter(models.Customer.id == request.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Extract live feature modifications from UI
    current_income = request.income if request.income else customer.income
    current_loan = request.loan_amount if request.loan_amount else customer.loan_amount
    
    from inference import run_live_inference
    import math

    # Pass actual numerical parameters to trained PyTorch model
    prob, shap_vals = run_live_inference(
        income=current_income,
        loan_amount=current_loan,
        age=customer.age,
        credit_score=customer.credit_score
    )
    
    return {
        "new_default_probability": max(0.0, min(1.0, prob)),
        "delta": prob - customer.default_probability,
        "mock_shap_values": [
            {"feature_name": "Income", "contribution": float(shap_vals[0])},
            {"feature_name": "Loan Amount", "contribution": float(shap_vals[1])},
            {"feature_name": "Age", "contribution": float(shap_vals[2])},
            {"feature_name": "Credit Score", "contribution": float(shap_vals[3])}
        ]
    }

@app.post("/chat")
def chat(request: schemas.ChatRequest, db: Session = Depends(database.get_db)):
    customer = db.query(models.Customer).filter(models.Customer.id == request.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to mock if API key is not provided yet
        return {"reply": f"Based on my analysis of customer {customer.name}, the requested parameter shifts will significantly alter their risk profile. Their income level offers some mitigation, but increasing the loan amount will raise the likelihood of default."}
    
    # Here we would do:
    # client = openai.OpenAI(api_key=api_key)
    # response = client.chat.completions.create(...)
    # return {"reply": response.choices[0].message.content}
    
    # But since it's a real API call and we need it to work in the prototype if the key is provided:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a risk analyst explainability engine. Explain the user's risk based on their factors."},
                {"role": "user", "content": f"Customer: {customer.name}, Income: {customer.income}, Loan: {customer.loan_amount}. Query: {request.query}"}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
         return {"reply": f"[OpenAI Error: {str(e)}] Fallback: This customer's base default probability is {customer.default_probability*100}%."}
