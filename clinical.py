"""Transparent RULE-BASED reference score (partial NEWS2: RR, SpO2, HR, temperature).
NOT a trained ML model. Logistic Regression / XGBoost baselines need real, authorised data (see train_clinical.py)."""
from typing import Optional
from pydantic import BaseModel, Field
class ClinicalInput(BaseModel):
    age: int = Field(ge=0, le=120)
    heart_rate: Optional[float] = None
    resp_rate: Optional[float] = None
    temperature: Optional[float] = None
    spo2: Optional[float] = None
    wbc: Optional[float] = None
    creatinine: Optional[float] = None
    lactate: Optional[float] = None
def _band(v, bands):
    for lo, hi, p in bands:
        if lo <= v <= hi: return p
    return 0
RR = [(0,8,3),(9,11,1),(12,20,0),(21,24,2),(25,999,3)]
SP = [(0,91,3),(92,93,2),(94,95,1),(96,100,0)]
HR = [(0,40,3),(41,50,1),(51,90,0),(91,110,1),(111,130,2),(131,999,3)]
TP = [(0,35.0,3),(35.01,36.0,1),(36.01,38.0,0),(38.01,39.0,1),(39.01,50,2)]
def assess(x: ClinicalInput) -> dict:
    parts, missing = {}, []
    for name, v, b in [("resp_rate", x.resp_rate, RR), ("spo2", x.spo2, SP), ("heart_rate", x.heart_rate, HR), ("temperature", x.temperature, TP)]:
        if v is None: missing.append(name)
        else: parts[name] = _band(v, b)
    score = sum(parts.values())
    cat = "High" if score >= 7 or (3 in parts.values() and score >= 5) else "Medium" if score >= 5 else "Low-Medium" if score >= 1 else "Low"
    flags = []
    for n, lo, hi in [("wbc", 4, 11), ("creatinine", 0.5, 1.3), ("lactate", 0, 2)]:
        v = getattr(x, n)
        if v is not None and not lo <= v <= hi: flags.append(f"{n}={v} outside reference {lo}-{hi}")
    return {"method": "rule-based partial NEWS2 (not ML)", "score": score, "priority": cat, "components": parts, "missing": missing, "lab_flags": flags}
