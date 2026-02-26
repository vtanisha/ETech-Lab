import torch
import joblib
import numpy as np
import shap
import warnings
from train_model import TabularDeepModel

warnings.filterwarnings('ignore')

model = None
scaler = None
explainer = None
background_tensor = None

def load_ml_system():
    global model, scaler, explainer, background_tensor
    if model is not None:
        return
        
    print("Loading PyTorch ML System into FastAPI memory...")
    try:
        model = TabularDeepModel(4, 1)
        model.load_state_dict(torch.load("models/model.pt"))
        model.eval()
        scaler = joblib.load("models/scaler.pkl")
        
        # Synthetic background for SHAP (zero mean normally)
        background = np.zeros((100, 4), dtype=np.float32)
        background_tensor = torch.tensor(background)
        # try/except SHAP deep explainer which can crash on newer pythons
        try:
            explainer = shap.DeepExplainer(model, background_tensor)
        except Exception:
            explainer = None
    except Exception as e:
        print(f"Error loading models: {e}")

def run_live_inference(income, loan_amount, age, credit_score):
    global model, scaler, explainer
    if model is None:
        load_ml_system()
        
    if model is None:
        # Fallback if model still not loaded
        return 0.5, [0, 0, 0, 0]
        
    # Features derived exactly as train_model.py
    # ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'EXT_SOURCE_2']
    ext_source_2 = credit_score / 850.0
    
    features = np.array([[income, loan_amount, age, ext_source_2]], dtype=np.float32)
    scaled_feats = scaler.transform(features)
    tensor_feats = torch.tensor(scaled_feats, dtype=torch.float32)
    
    with torch.no_grad():
        prob = model(tensor_feats).item()
        
    shap_vals = [0.0, 0.0, 0.0, 0.0]
    if explainer is not None:
        try:
            sv = explainer.shap_values(tensor_feats)
            if isinstance(sv, list):
                sv = sv[0]
            shap_vals = sv[0].tolist()
        except:
            pass # fallback to zeros if shap fails computationally on native mac arm64 
    
    # If shap failed, do a simple perturbation to compute directional importance
    if sum(abs(v) for v in shap_vals) < 0.0001:
        # manual gradient approximation
        tensor_feats.requires_grad_(True)
        out = model(tensor_feats)
        out.backward()
        grads = tensor_feats.grad[0].numpy()
        shap_vals = (grads * scaled_feats[0]).tolist()
        
    return prob, shap_vals
