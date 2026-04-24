"""
Cardiac Mortality Prediction — FastAPI Backend
Run with: uvicorn app:app --reload
Then open: http://localhost:8000
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import numpy as np
import os

# ── Load model & metadata ──────────────────────────────────────────────────
if not os.path.exists("model.pkl"):
    raise RuntimeError("model.pkl not found — run train.py first!")

pipeline     = joblib.load("model.pkl")
feature_info = json.load(open("feature_info.json"))
FEATURES     = feature_info["features"]

app = FastAPI(title="Cardiac Mortality Risk Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ── ALL fields Optional — FastAPI will NEVER return 422 for missing fields ──
class PatientData(BaseModel):
    patient_age:                Optional[float] = None
    gender_id:                  Optional[int]   = 1
    BMI:                        Optional[float] = None
    diabetes:                   Optional[int]   = 0
    hypertension:               Optional[int]   = 0
    Dyslipidemia:               Optional[int]   = 0
    dyslipidemia:               Optional[int]   = 0
    Active_tobacco_use:         Optional[int]   = 0
    f_history_cad:              Optional[int]   = 0
    Cerebovascular_disease:     Optional[int]   = 0
    chronic_lung_disease:       Optional[int]   = 0
    dialysis:                   Optional[int]   = 0
    ejection_fraction:          Optional[float] = None
    NYHA_class:                 Optional[int]   = 2
    congestive_heart_failure_A: Optional[int]   = 0
    cardiogenic_shock:          Optional[int]   = 0
    resuscitation:              Optional[int]   = 0
    myocardial_infarction:      Optional[int]   = 0
    arrhythmia:                 Optional[int]   = 0
    AFibFlutter:                Optional[int]   = 0
    pulmonary_artery_hypertension: Optional[int] = 0
    Coronaries_diseased:        Optional[int]   = 2
    left_main_disease:          Optional[int]   = 0
    Aortic_regurgitation:       Optional[int]   = 0
    Mitral_regurgitation:       Optional[int]   = 0
    last_hematocrit:            Optional[float] = None
    last_cretenine_preop:       Optional[float] = None
    BPsystolic:                 Optional[float] = None
    perfusion_time_min:         Optional[float] = None
    cross_clamp_time_min:       Optional[float] = None
    IABP:                       Optional[int]   = 0
    intraop_blood_products:     Optional[int]   = 0
    Total_bypasses_grafted:     Optional[int]   = 3
    initial_hours_ventilated:   Optional[float] = None
    initial_icu_hours:          Optional[float] = None
    drainage_at_12_hours:       Optional[float] = None
    drainage_at_24_hours:       Optional[float] = None
    post_op_creatinine:         Optional[float] = None
    reintubated_hospital_stay:  Optional[int]   = None

    model_config = {"populate_by_name": True, "extra": "allow"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        data = patient.model_dump()

        # Merge dyslipidemia casings
        if data.get("Dyslipidemia") is not None:
            data["dyslipidemia"] = data["Dyslipidemia"]

        # Build feature vector — missing → NaN (pipeline imputer handles it)
        row = []
        for feat in FEATURES:
            val = data.get(feat)
            row.append(np.nan if val is None else float(val))

        X    = np.array(row, dtype=float).reshape(1, -1)
        prob = float(pipeline.predict_proba(X)[0, 1])

        if prob < 0.05:
            level, color = "Low", "green"
            interp = "Mortality risk is low. Standard post-operative monitoring recommended."
        elif prob < 0.15:
            level, color = "Moderate", "yellow"
            interp = "Moderate risk. Enhanced monitoring and early intervention protocols advised."
        elif prob < 0.35:
            level, color = "High", "orange"
            interp = "High risk. ICU-level care, cardiology review, and family counselling recommended."
        else:
            level, color = "Critical", "red"
            interp = "Critical risk. Multidisciplinary review required. Consider risk-benefit of surgery."

        selector    = pipeline.named_steps["selector"]
        model       = pipeline.named_steps["model"]
        sel_feats   = np.array(FEATURES)[selector.get_support()]
        importances = model.feature_importances_
        descriptions = feature_info.get("feature_descriptions", {})

        top_risk = []
        for i in np.argsort(importances)[::-1][:5]:
            feat = sel_feats[i]
            top_risk.append({
                "feature":    descriptions.get(feat, feat),
                "value":      data.get(feat),
                "importance": round(float(importances[i]), 3)
            })

        return {
            "risk_probability":  round(prob, 4),
            "risk_percent":      f"{prob*100:.1f}%",
            "risk_level":        level,
            "risk_color":        color,
            "interpretation":    interp,
            "model_auc":         feature_info["model_auc"],
            "top_risk_features": top_risk,
        }

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500,
                            content={"detail": str(e), "trace": traceback.format_exc()})


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "model_auc": feature_info["model_auc"]}

@app.get("/model-info")
def model_info():
    return {
        "features_used": len(FEATURES),
        "model_auc":     feature_info["model_auc"],
        "cv_auc_mean":   feature_info["cv_auc_mean"],
        "cv_auc_std":    feature_info["cv_auc_std"],
    }

