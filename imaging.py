"""Chest X-ray quality validation + DenseNet-121 inference (only if a real checkpoint exists)."""
import io, os
import numpy as np
from PIL import Image
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
CKPT = os.environ.get("MEDSCOPE_IMG_CKPT", "checkpoints/densenet121_cxr.pt")
def validate(data: bytes) -> dict:
    try: im = Image.open(io.BytesIO(data)).convert("L")
    except Exception as e: return {"ok": False, "issues": [f"unreadable image: {e}"]}
    a = np.asarray(im, dtype=float); issues = []
    if min(a.shape) < 224: issues.append("resolution below 224px")
    if a.std() < 20: issues.append("low contrast")
    if a.mean() < 30 or a.mean() > 225: issues.append("under/over-exposed")
    if (np.abs(np.diff(a, 2, axis=0)).var() if a.shape[0] > 2 else 0) < 5: issues.append("possible blur")
    return {"ok": not issues, "issues": issues, "size": im.size, "mean": round(a.mean(), 1), "contrast_std": round(a.std(), 1)}
def model_available():
    try: import torch, torchvision  # noqa
    except ImportError: return False, "torch/torchvision not installed"
    return (True, "checkpoint found") if os.path.exists(CKPT) else (False, f"no checkpoint at {CKPT}")
def _load():
    import torch, torchvision
    m = torchvision.models.densenet121(); m.classifier = torch.nn.Linear(1024, len(FINDINGS))
    m.load_state_dict(torch.load(CKPT, map_location="cpu")); return m.eval()
def _prep(data):
    import torch
    im = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    a = (np.asarray(im, dtype="float32") / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float32)[None]
def predict(data: bytes):
    if not model_available()[0]: return None
    import torch
    with torch.no_grad(): p = torch.sigmoid(_load()(_prep(data)))[0].tolist()
    return dict(zip(FINDINGS, p))
def gradcam(data: bytes, idx: int):
    """Grad-CAM on DenseNet features. UNTESTED without a trained checkpoint."""
    if not model_available()[0]: return None
    import torch.nn.functional as F
    m = _load(); st = {}
    def fwd(_m, _i, o): st["a"] = o; o.register_hook(lambda g: st.__setitem__("g", g))
    m.features.register_forward_hook(fwd); m(_prep(data))[0, idx].backward()
    cam = F.relu((st["g"].mean((2, 3), keepdim=True) * st["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, (224, 224), mode="bilinear")[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
