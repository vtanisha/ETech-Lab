from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    income = Column(Float)
    loan_amount = Column(Float)
    employment_type = Column(String)
    credit_score = Column(Float)
    risk_tier = Column(String) # High, Medium, Low
    default_probability = Column(Float) # 0-1
    notes = Column(Text, nullable=True)

    shap_values = relationship("SHAPValue", back_populates="customer")

class SHAPValue(Base):
    __tablename__ = "shap_values"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    feature_name = Column(String)
    contribution = Column(Float) # The SHAP contribution

    customer = relationship("Customer", back_populates="shap_values")
