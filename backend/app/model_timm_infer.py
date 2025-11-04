# app/model_timm_infer.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------- constants / defaults ----------
TIMM_MEAN = (0.485, 0.456, 0.406)
TIMM_STD  = (0.229, 0.224, 0.225)
DEFAULT_INPUT = int(os.getenv("MODEL_INPUT_SIZE", "300"))

# We ALWAYS present these in the API, regardless of what the ckpt stored.
PRETTY = {
    "AiArtData": "Bot-Made",
    "AI Art": "Bot-Made",
    "AI_Art": "Bot-Made",
    "ai": "Bot-Made",
    "RealArt": "Brush-Made",
    "Real Art": "Brush-Made",
    "Real_Art": "Brush-Made",
    "real": "Brush-Made",
}
DISPLAY_CLASS_NAMES = ["Bot-Made", "Brush-Made"]  # 0 = AI, 1 = Real

def _to_pretty_name(name: str) -> str:
    return PRETTY.get(name, name)

def _auto_find_latest_model(root: Path) -> str:
    # newest *.pth under runs
    candidates = list(root.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth found under {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])

def _build_transform(input_size: int):
    # ✅ No unnecessary upscale to 384
    return A.Compose([
        A.SmallestMaxSize(input_size),
        A.CenterCrop(input_size, input_size),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=TIMM_MEAN, std=TIMM_STD),
        ToTensorV2(),
    ])

def _load_ckpt(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

class ArtModel:
    """
    timm model + albumentations preprocessing + temperature scaling.
    Always returns 'Bot-Made' / 'Brush-Made' labels.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        # Resolve path
        if checkpoint_path is None:
            env_path = os.getenv("MODEL_PATH")
            if env_path and Path(env_path).exists():
                checkpoint_path = env_path
            else:
                checkpoint_path = _auto_find_latest_model(Path(r"C:\ArtSentinel\models\runs"))

        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = _load_ckpt(self.checkpoint_path, map_location=device)

        # Pull metadata (all optional in legacy ckpts)
        self.model_name  = ckpt.get("model_name", os.getenv("MODEL_NAME"))
        if not self.model_name:
            raise RuntimeError(
                "No model_name found in checkpoint or environment. "
                "Set MODEL_NAME in .env (e.g., tf_efficientnet_b3) or "
                "use the inspector script to detect it."
            )

        self.temperature = float(ckpt.get("temperature", 1.0))
        ckpt_classes     = ckpt.get("class_names", ["AiArtData", "RealArt"])
        # Normalize class order to pretty names for downstream display
        self.class_names = [_to_pretty_name(x) for x in ckpt_classes]

        # Input size (int or (h,w))
        input_size = ckpt.get("input_size", DEFAULT_INPUT)
        if isinstance(input_size, (list, tuple)):
            input_size = int(input_size[0])
        self.input_size = int(input_size)

        # Build model & load state dict
        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.class_names))
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # strip "module." if present
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[model_timm_infer] Unexpected keys in state_dict: {unexpected}")
        if missing:
            print(f"[model_timm_infer] Missing keys in state_dict: {missing}")

        # ✅ CPU runs in float32 (avoid BF16 issues); CUDA can still use AMP inside predict()
        self.model.eval().to(device=torch.device(device), dtype=torch.float32)
        self.device = device
        self.transform = _build_transform(self.input_size)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        img = np.array(image.convert("RGB"))
        x = self.transform(image=img)["image"].unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32)

        with torch.inference_mode():
            if self.device == "cuda":
                # use fp16 autocast on GPU only
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = self.model(x)
            else:
                # CPU path: stay in float32 (no autocast)
                logits = self.model(x)

            logits = logits / max(1e-6, self.temperature)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        display_probs = {
            "Bot-Made": float(probs[0]) if len(probs) > 0 else 0.0,
            "Brush-Made": float(probs[1]) if len(probs) > 1 else 0.0,
        }
        return {
            "label": DISPLAY_CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "all_probs": display_probs,
            "model_name": self.model_name,
            "temperature": float(self.temperature),
            "input_size": self.input_size,
            "checkpoint_path": self.checkpoint_path,
        }


# ---------- hot-swap support ----------
_active_model: Optional[ArtModel] = None

def get_active_model() -> ArtModel:
    global _active_model
    if _active_model is None:
        _active_model = ArtModel()  # env/auto-detect
    return _active_model

def load_model(new_path: Optional[str]) -> ArtModel:
    """
    Force-reload a different checkpoint at runtime.
    If new_path is None, will reuse env/auto-detect logic.
    """
    global _active_model
    _active_model = ArtModel(new_path)
    return _active_model
