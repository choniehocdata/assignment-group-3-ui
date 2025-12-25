import streamlit as st
import joblib
from pathlib import Path
import threading
import time

# ===== Load model =====
MODEL_PATH = "best_bank_model_fixed.pkl"
model = joblib.load(MODEL_PATH)

# ===== Streamlit config =====
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;} /* optional */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===== Start FastAPI in background (1 lần) =====
def start_api():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import pandas as pd
    import uvicorn

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL_PATH}

    @app.post("/predict")
    def predict(payload: dict):
        # Map đúng cột như lúc train
        input_data = pd.DataFrame([{
            "age": int(payload["age"]),
            "job": payload["job"],
            "marital": payload["marital"],
            "education": payload["education"],
            "default": payload["default"],
            "balance": float(payload["balance"]),
            "housing": payload["housing"],
            "loan": payload["loan"],
            "contact": payload["contact"],
            "day": int(payload["day"]),
            "month": payload["month"],
            "duration": int(payload["duration"]),
            "campaign": int(payload["campaign"]),
            "pdays": int(payload["pdays"]),
            "previous": int(payload["previous"]),
            "poutcome": payload["poutcome"],
        }])

        pred = int(model.predict(input_data)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_data)[0][1])  # xác suất class=1

        return {"pred": pred, "proba": proba}

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

# chạy api 1 lần / session
if "api_started" not in st.session_state:
    st.session_state.api_started = True
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    time.sleep(0.3)  # đợi API nhích lên chút

# ===== Render HTML dashboard =====
html_file = Path("dashboard.html")
if html_file.exists():
    html_content = html_file.read_text(encoding="utf-8")
    st.components.v1.html(html_content, height=2000, scrolling=True)
else:
    st.error("Không tìm thấy dashboard.html")
