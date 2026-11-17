from fastapi import FastAPI
from pydantic import BaseModel
from medscope import core, clinical, imaging, datasets
core.init(); app = FastAPI(title="MedScope AI (research prototype)")
class P(BaseModel): age: int; sex: str; history: str = ""
@app.get("/health")
def health(): return {"status": "ok", "disclaimer": "Research prototype; not a diagnostic system."}
@app.post("/patients")
def patients(p: P): return {"pid": core.add_patient(p.age, p.sex, p.history)}
@app.get("/patients")
def list_patients(): return core.rows("select * from patients")
@app.post("/assess/{pid}")
def assess(pid: str, x: clinical.ClinicalInput):
    r = clinical.assess(x); return {"case_id": core.add_case(pid, r, r["priority"], r["score"]), **r}
@app.get("/cases")
def cases(): return core.rows("select * from cases")
@app.post("/cases/{cid}/review")
def review(cid: int, status: str): core.set_status(cid, status); return {"ok": True}
@app.get("/audit")
def audit(): return core.rows("select * from audit order by id desc")
@app.get("/models")
def models(): return {"image": imaging.model_available(), "datasets": {k: datasets.status(k) for k in datasets.REGISTRY}}
