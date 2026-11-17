import os, tempfile, pandas as pd, pytest
os.environ["MEDSCOPE_DB"] = tempfile.mktemp(suffix=".db")
from medscope import core, clinical, fusion, datasets, imaging
def test_clinical_scoring():
    r = clinical.assess(clinical.ClinicalInput(age=60, resp_rate=26, spo2=90, heart_rate=120, temperature=39.5))
    assert r["priority"] == "High" and r["missing"] == []
def test_missing_flagged(): assert clinical.assess(clinical.ClinicalInput(age=30, spo2=97))["missing"]
def test_fusion_no_fake_image():
    f = fusion.late_fusion(None, clinical.assess(clinical.ClinicalInput(age=50, spo2=97)))
    assert "Image model unavailable" in f["warnings"][0]
def test_split_no_leakage():
    df = pd.DataFrame({"subject_id": [i // 3 for i in range(300)], "x": range(300)}); p = datasets.patient_split(df)
    assert not (set(p[0].subject_id) & set(p[2].subject_id))
    with pytest.raises(ValueError): datasets.assert_no_leakage({1}, {1})
def test_db_audit():
    core.init(); pid = core.add_patient(40, "F"); cid = core.add_case(pid, {}, "Low", 0); core.set_status(cid, "Reviewed")
    assert any(a["action"] == "clinician_decision" for a in core.rows("select * from audit"))
def test_quality_rejects_garbage(): assert not imaging.validate(b"junk")["ok"]
