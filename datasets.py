"""Dataset access gates + patient-level splitting. Nothing is ever downloaded automatically."""
import os
REGISTRY = {
 "CheXpert": {"access": "Stanford AIMI research agreement", "env": "CHEXPERT_DIR"},
 "MIMIC-CXR": {"access": "PhysioNet credentialed + CITI training + DUA", "env": "MIMIC_CXR_DIR"},
 "MIMIC-IV": {"access": "PhysioNet credentialed + CITI training + DUA", "env": "MIMIC_IV_DIR"},
 "CheXpert Plus": {"access": "Stanford AIMI research agreement", "env": "CHEXPERT_PLUS_DIR"},
}
def status(name):
    d = os.environ.get(REGISTRY[name]["env"]); return "configured" if d and os.path.isdir(d) else "not configured (access required)"
def assert_no_leakage(*sets):
    for i, s in enumerate(sets):
        for t in sets[i + 1:]:
            if s & t: raise ValueError("patient-level leakage detected")
def patient_split(df, patient_col="subject_id", seed=0, frac=(0.7, 0.15, 0.15)):
    """Split by PATIENT so no patient appears in more than one split."""
    ids = df[patient_col].drop_duplicates().sample(frac=1, random_state=seed).tolist(); n = len(ids)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    sets = [set(ids[:a]), set(ids[a:b]), set(ids[b:])]; assert_no_leakage(*sets)
    return [df[df[patient_col].isin(s)] for s in sets]
# Join MIMIC-CXR and MIMIC-IV only on subject_id; never link unrelated patients.
