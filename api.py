from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# CORS cho live server 5500
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("best_bank_model_fixed.pkl")

# ==== Input schema (đúng các field dashboard gửi) ====
class PredictInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

def to_features(d: PredictInput):
    """
    ⚠️ Bạn phải map input -> đúng format model đã train.
    Nếu model của bạn là pipeline xử lý categorical luôn,
    thì chỉ cần đưa DataFrame đúng cột.
    """
    import pandas as pd
    return pd.DataFrame([d.model_dump()])

@app.post("/predict")
def predict(inp: PredictInput):
    X = to_features(inp)

    # predict + proba
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        proba = float(pred)

    return {"pred": pred, "proba": proba}
