import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "models" / "churn_model.joblib")
model = bundle["model"]
threshold = float(bundle["threshold"])
meta = json.loads((BASE / "models" / "meta.json").read_text())
columns = list(model.feature_names_in_)

app = FastAPI(title="ChurnSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_origin_regex=r"https://.*\.(vercel\.app|onrender\.com)",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    """Tells the frontend which inputs to render."""
    return {"columns": columns, "num": meta["num"], "cat": meta["cat"],
            "threshold": threshold}


@app.post("/predict")
def predict(payload: dict):
    missing = [c for c in columns if c not in payload]
    if missing:
        raise HTTPException(422, f"Missing fields: {missing}")

    row = pd.DataFrame([{c: payload[c] for c in columns}])

    for c in meta["num"]:
        row[c] = pd.to_numeric(row[c], errors="coerce")
    if row[list(meta["num"])].isna().any().any():
        raise HTTPException(422, "Numeric fields must be valid numbers")

    for c, options in meta["cat"].items():
        if str(row.at[0, c]) not in options:
            raise HTTPException(422, f"Invalid value for {c}")

    score = float(model.predict_proba(row)[0, 1])
    return {"score": score, "threshold": threshold, "churn": score >= threshold}