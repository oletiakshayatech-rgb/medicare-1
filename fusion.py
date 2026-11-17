RANK = {"Low": 0, "Low-Medium": 1, "Medium": 2, "High": 3}
def late_fusion(img_probs, clin):
    warn = []
    if img_probs is None: warn.append("Image model unavailable: image evidence not used.")
    if clin is None: warn.append("No clinical assessment entered.")
    elif clin["missing"]: warn.append("Missing vitals: " + ", ".join(clin["missing"]))
    img_cat = None
    if img_probs: img_cat = "High" if max(img_probs.values()) >= 0.7 else "Medium" if max(img_probs.values()) >= 0.4 else "Low"
    cats = [c for c in (img_cat, clin["priority"] if clin else None) if c]
    status = "Fused (image + clinical)" if img_probs and clin else "Single modality only" if cats else "No evidence"
    # conservative rule: higher-priority category wins; thresholds are UNVALIDATED placeholders
    return {"status": status, "priority": max(cats, key=RANK.get) if cats else "Unassessed", "warnings": warn}
