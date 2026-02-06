import pandas as pd
import os
from database import SessionLocal, engine
from models import Base, Customer, SHAPValue

# Create tables
Base.metadata.create_all(bind=engine)

def seed_database():
    db = SessionLocal()
    try:
        if db.query(Customer).count() > 0:
            print("Database already seeded.")
            return

        mock_customers = [
            {
                "name": "Alex Johnson",
                "age": 34,
                "income": 75000,
                "loan_amount": 250000,
                "employment_type": "Salaried",
                "credit_score": 720,
                "risk_tier": "Low",
                "default_probability": 0.05,
            },
            {
                "name": "Sarah Lee",
                "age": 28,
                "income": 45000,
                "loan_amount": 180000,
                "employment_type": "Self-Employed",
                "credit_score": 650,
                "risk_tier": "Medium",
                "default_probability": 0.25,
            },
            {
                "name": "Mike Davis",
                "age": 45,
                "income": 30000,
                "loan_amount": 350000,
                "employment_type": "Unemployed",
                "credit_score": 550,
                "risk_tier": "High",
                "default_probability": 0.82,
            }
        ]
        
        for data in mock_customers:
            cust = Customer(**data)
            db.add(cust)
            db.commit()
            db.refresh(cust)
            
            # Add mock SHAP values
            shap1 = SHAPValue(customer_id=cust.id, feature_name="Income", contribution=-0.05 if cust.income > 50000 else 0.1)
            shap2 = SHAPValue(customer_id=cust.id, feature_name="Loan Amount", contribution=0.15 if cust.loan_amount > 200000 else -0.05)
            db.add(shap1)
            db.add(shap2)
        
        db.commit()
        print("Mock database populated successfully!")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
