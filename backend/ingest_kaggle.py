import pandas as pd
import numpy as np
from database import SessionLocal, engine
from models import Base, Customer, SHAPValue
import os

# Create tables
Base.metadata.create_all(bind=engine)

def ingest_data():
    csv_path = "Bank Data Sources/application_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    # Load 50 samples to keep the MVP snappy and demonstrate the live DB
    df = pd.read_csv(csv_path, nrows=50)

    db = SessionLocal()
    try:
        # Clear existing mock customers to replace with real dataset
        print("Clearing old data...")
        db.query(SHAPValue).delete()
        db.query(Customer).delete()
        db.commit()

        print("Ingesting Kaggle data...")
        for _, row in df.iterrows():
            ext_source_2 = row['EXT_SOURCE_2']
            if pd.isna(ext_source_2):
                ext_source_2 = 0.5
                
            cs = int(ext_source_2 * 850)
            credit_score = max(300, min(850, cs))
            
            target = int(row['TARGET'])
            risk_tier = "High" if target == 1 else ("Medium" if ext_source_2 < 0.4 else "Low")
            default_prob = 0.85 if target == 1 else (0.25 if risk_tier == "Medium" else 0.05)
            
            occ_type = row['OCCUPATION_TYPE']
            notes = occ_type if pd.notna(occ_type) else "Unspecified"
            
            cust = Customer(
                name=f"Applicant {row['SK_ID_CURR']}",
                age=int(abs(row['DAYS_BIRTH']) // 365),
                income=float(row['AMT_INCOME_TOTAL']),
                loan_amount=float(row['AMT_CREDIT']),
                employment_type=str(row['NAME_INCOME_TYPE']),
                credit_score=credit_score,
                risk_tier=risk_tier,
                default_probability=default_prob,
                notes=str(notes)
            )
            db.add(cust)
            db.commit()
            db.refresh(cust)
            
            # Add SHAP features for the visualizations
            shap1 = SHAPValue(customer_id=cust.id, feature_name="Income", contribution=-0.05 if cust.income > 150000 else 0.08)
            shap2 = SHAPValue(customer_id=cust.id, feature_name="EXT_SOURCE_2", contribution=-0.12 if ext_source_2 > 0.5 else 0.15)
            shap3 = SHAPValue(customer_id=cust.id, feature_name="Employment", contribution=0.03 if notes == "Laborers" else -0.02)
            db.add_all([shap1, shap2, shap3])
            
        db.commit()
        print("Kaggle database successfully ingested into PostgreSQL!")
    except Exception as e:
        print(f"Failed to ingest: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    ingest_data()
