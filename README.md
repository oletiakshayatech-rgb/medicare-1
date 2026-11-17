# MedScope AI — research prototype (NOT a diagnostic system)
Run: `pip install -r requirements.txt` → `streamlit run app.py` · API: `uvicorn backend.main:app` · Tests: `pytest`
Honest status: UI, SQLite, API, quality checks, fusion logic, audit, reports, patient-level splitting are implemented.
Clinical scoring is a rule-based partial NEWS2, not ML. DenseNet-121 inference, Grad-CAM, SHAP/XGBoost/MONAI/Captum
need real checkpoints/authorised data (CheXpert, MIMIC-CXR, MIMIC-IV, CheXpert Plus) and are untrained/untested here. No metrics are reported.
