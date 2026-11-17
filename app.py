import json, pandas as pd, streamlit as st
from medscope import core, clinical, imaging, fusion, datasets
core.init()
st.set_page_config("MedScope AI", "🩺", layout="wide")
st.markdown("""<style>
[data-testid=stSidebar]{background:#0B2545}[data-testid=stSidebar] *{color:#fff!important}
h1,h2,h3{color:#0B2545}.card{background:#fff;border-left:5px solid #0FA3B1;padding:16px;border-radius:10px;box-shadow:0 2px 8px #0b254518}
.warn{background:#FFF4E5;border-left:5px solid #E8A33D;padding:10px;border-radius:8px}</style>""", unsafe_allow_html=True)
S = st.session_state
PAGES = ["Overview", "Patient Management", "Image Analysis", "Clinical Assessment", "Multimodal Assessment", "Case Prioritization", "Explainable AI", "Reports", "Audit History", "Model Registry"]
st.sidebar.title("🩺 MedScope AI"); page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption("Research prototype · clinical decision support only · not a diagnosis · demo data is SYNTHETIC")
st.markdown('<div class="warn">⚠️ Research prototype. Outputs are not confirmed diagnoses. All patients shown are <b>synthetic</b>.</div>', unsafe_allow_html=True)
pts, cases = core.rows("select * from patients"), core.rows("select * from cases")
def pick(label="Patient"):
    if not pts: st.info("Register a patient first."); st.stop()
    return st.selectbox(label, [p["pid"] for p in pts])
st.title(page)
if page == "Overview":
    c = st.columns(4); ok, why = imaging.model_available()
    c[0].metric("Total cases", len(cases)); c[1].metric("Pending reviews", sum(x["status"] == "Pending review" for x in cases))
    c[2].metric("High priority", sum(x["priority"] == "High" for x in cases)); c[3].metric("Image model", "Available" if ok else "Unavailable")
    if cases: st.bar_chart(pd.DataFrame(cases).priority.value_counts())
    else: st.info("No cases yet. Add synthetic patients and run a clinical assessment.")
    if st.button("Seed 3 synthetic demo patients"):
        for a, s in [(54, "M"), (71, "F"), (38, "F")]: core.add_patient(a, s, "SYNTHETIC demo record")
        st.rerun()
elif page == "Patient Management":
    with st.form("p"):
        a = st.number_input("Age", 0, 120, 50); s = st.selectbox("Sex", ["F", "M", "Other"]); h = st.text_area("History (fictional only; no real identifiers)")
        if st.form_submit_button("Register synthetic patient"): st.success("Created " + core.add_patient(a, s, h)); st.rerun()
    st.dataframe(pd.DataFrame(pts), use_container_width=True)
    if pts:
        pid = pick("Case history"); st.dataframe(pd.DataFrame([x for x in cases if x["pid"] == pid]), use_container_width=True)
elif page == "Image Analysis":
    f = st.file_uploader("Chest X-ray (PNG/JPG)", ["png", "jpg", "jpeg"])
    if f:
        data = f.getvalue(); S["img"] = data; st.image(data, width=360); q = imaging.validate(data); S["q"] = q
        (st.success if q["ok"] else st.error)("Quality check: " + ("passed" if q["ok"] else "; ".join(q["issues"]))); st.json(q)
        probs = imaging.predict(data); S["probs"] = probs
        if probs: st.bar_chart(pd.Series(probs)); st.caption("Model scores, not diagnoses. Validation status: see Model Registry.")
        else: st.warning("No image model available: " + imaging.model_available()[1] + ". No findings are shown. Nothing is simulated.")
elif page == "Clinical Assessment":
    pid = pick(); c = st.columns(4)
    v = dict(age=c[0].number_input("Age", 0, 120, 50), heart_rate=c[1].number_input("Heart rate", 0.0, 300.0, 80.0), resp_rate=c[2].number_input("Resp rate", 0.0, 80.0, 16.0), temperature=c[3].number_input("Temp °C", 25.0, 45.0, 37.0))
    c = st.columns(4); v["spo2"] = c[0].number_input("SpO₂ %", 0.0, 100.0, 97.0); v["wbc"] = c[1].number_input("WBC", 0.0, 100.0, 7.0); v["creatinine"] = c[2].number_input("Creatinine", 0.0, 20.0, 1.0); v["lactate"] = c[3].number_input("Lactate", 0.0, 30.0, 1.0)
    if st.button("Run assessment"):
        r = clinical.assess(clinical.ClinicalInput(**v)); S["clin"] = r; S["pid"] = pid; st.json(r); st.bar_chart(pd.Series(r["components"]))
        st.caption("Rule-based reference score (partial NEWS2) — NOT a trained ML model.")
        S["case"] = core.add_case(pid, r, r["priority"], r["score"])
elif page == "Multimodal Assessment":
    f = fusion.late_fusion(S.get("probs"), S.get("clin")); st.metric("Fusion status", f["status"]); st.metric("Provisional priority", f["priority"])
    for w in f["warnings"]: st.warning(w)
    st.caption("Fusion rule: higher-priority modality wins. Thresholds are unvalidated placeholders.")
    if S.get("clin"): st.json(S["clin"])
elif page == "Case Prioritization":
    if not cases: st.info("No cases.")
    for x in sorted(cases, key=lambda x: -x["score"]):
        c = st.columns([1, 1, 1, 2, 2]); c[0].write(f"#{x['id']}"); c[1].write(x["pid"]); c[2].write(x["priority"]); c[3].write(x["status"])
        new = c[4].selectbox("Decision", ["Pending review", "Reviewed – agree", "Reviewed – disagree", "Escalated"], key=f"s{x['id']}", index=0, label_visibility="collapsed")
        if new != x["status"] and c[4].button("Save", key=f"b{x['id']}"): core.set_status(x["id"], new); st.rerun()
elif page == "Explainable AI":
    ok, why = imaging.model_available()
    st.subheader("Grad-CAM"); st.info("Available." if ok and S.get("img") else f"Unavailable: requires a real trained DenseNet-121 checkpoint and an uploaded image ({why}).")
    if ok and S.get("img"):
        import numpy as np; from PIL import Image
        i = st.selectbox("Finding", range(5), format_func=lambda k: imaging.FINDINGS[k]); cam = imaging.gradcam(S["img"], i)
        base = np.asarray(Image.open(__import__("io").BytesIO(S["img"])).convert("L").resize((224, 224)), float) / 255
        st.image(np.stack([cam, base * .6, 1 - cam], -1), caption="Heatmap (explains model attention, not pathology)")
    st.subheader("SHAP"); st.info("Unavailable: requires a trained clinical ML model. The current clinical score is rule-based, so its per-feature points are shown instead.")
    if S.get("clin"): st.bar_chart(pd.Series(S["clin"]["components"]))
elif page == "Reports":
    if not cases: st.info("No cases.")
    else:
        cid = st.selectbox("Case", [x["id"] for x in cases]); x = next(c for c in cases if c["id"] == cid); s = json.loads(x["summary"])
        rep = f"""# PRELIMINARY ASSESSMENT (RESEARCH PROTOTYPE – NOT A DIAGNOSIS)\nCase {x['id']} | Patient {x['pid']} (synthetic) | {x['created']}\n\nPriority: {x['priority']} (score {x['score']})\nMethod: {s.get('method')}\nMissing inputs: {', '.join(s.get('missing', [])) or 'none'}\nLab flags: {'; '.join(s.get('lab_flags', [])) or 'none'}\nImage model output: {'not included (model unavailable)' if not S.get('probs') else 'included'}\nClinician review status: {x['status']}\n\nThis output supports, and does not replace, clinical judgement."""
        st.code(rep); st.download_button("Download report", rep, f"report_case_{cid}.md"); core.log("clinician", "report_viewed", f"case {cid}")
elif page == "Audit History": st.dataframe(pd.DataFrame(core.rows("select * from audit order by id desc")), use_container_width=True)
elif page == "Model Registry":
    ok, why = imaging.model_available()
    st.dataframe(pd.DataFrame([{"Model": "DenseNet-121 (5 findings)", "Version": "–", "Available": ok, "Detail": why, "Validation": "Not validated"},
        {"Model": "Clinical Logistic Regression / XGBoost", "Version": "–", "Available": False, "Detail": "not trained (needs authorised data)", "Validation": "Not validated"},
        {"Model": "Rule-based partial NEWS2", "Version": "1.0", "Available": True, "Detail": "deterministic rules", "Validation": "Not validated in this project"}]), use_container_width=True)
    st.subheader("Datasets (access-gated, never auto-downloaded)")
    st.dataframe(pd.DataFrame([{"Dataset": k, "Requirement": v["access"], "Status": datasets.status(k)} for k, v in datasets.REGISTRY.items()]), use_container_width=True)
