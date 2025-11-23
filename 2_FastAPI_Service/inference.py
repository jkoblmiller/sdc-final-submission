import os
import torch
import io
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# Model / labels configuration
IMG_SIZE = 224
NUM_CLASSES = 37

# A canonical list of 37 Oxford-IIIT Pet breed names (lowercase/underscore)
LABELS = [
    "Abyssinian","american_bulldog","american_pit_bull_terrier","basset_hound","beagle",
    "Bengal","Birman","Bombay","boxer","British_Shorthair","chihuahua","Egyptian_Mau",
    "english_cocker_spaniel","english_setter","german_shorthaired","great_pyrenees","havanese",
    "japanese_chin","keeshond","leonberger","Maine_Coon","miniature_pinscher","newfoundland",
    "Persian","pomeranian","pug","Ragdoll","Russian_Blue","saint_bernard","samoyed",
    "scottish_terrier","shiba_inu","Siamese","Sphynx","staffordshire_bull_terrier","wheaten_terrier",
    "yorkshire_terrier",
]


def _build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    return model


def _get_model():
    global _MODEL
    if "_MODEL" in globals() and _MODEL is not None:
        return _MODEL

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "pet-classifier-resnet50.pth"))

    # Try to load checkpoint first — support full-model, state_dict, and wrapper dicts
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=torch.device("cpu"))

            # If the saved object is a full Module, use it directly
            if isinstance(state, torch.nn.Module):
                model = state

            # If it's a dict, decide whether it contains a state_dict or is the state_dict itself
            elif isinstance(state, dict):
                # Case: {'model_state_dict': ...}
                if 'model_state_dict' in state:
                    model = _build_model()
                    model.load_state_dict(state['model_state_dict'], strict=False)
                else:
                    # Try to detect if dict *is* a state_dict (mapping of parameter tensors)
                    # Heuristic: keys contain 'weight' or 'bias' strings
                    keys = list(state.keys())
                    if keys and any(('weight' in k or 'bias' in k) for k in keys):
                        model = _build_model()
                        model.load_state_dict(state, strict=False)
                    else:
                        # Maybe user saved a wrapper dict; scan values for a Module or nested state_dict
                        found = False
                        for v in state.values():
                            if isinstance(v, torch.nn.Module):
                                model = v
                                found = True
                                break
                            if isinstance(v, dict):
                                inner_keys = list(v.keys())
                                if inner_keys and any(('weight' in k or 'bias' in k) for k in inner_keys):
                                    model = _build_model()
                                    model.load_state_dict(v, strict=False)
                                    found = True
                                    break
                        if not found:
                            # Fall back to building model without weights
                            print(f"Warning: checkpoint at {model_path} did not contain recognizable weights; using fresh model")
                            model = _build_model()

            else:
                # Unknown type; fall back to building model
                print(f"Warning: unrecognized checkpoint type {type(state)}; using fresh model")
                model = _build_model()

        except Exception as e:
            print(f"Warning: failed to load checkpoint {model_path}: {e}; using fresh model")
            model = _build_model()

    else:
        # No checkpoint found — build fresh model
        model = _build_model()

    model.eval()
    _MODEL = model
    return _MODEL


_VAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_pil(image: Image.Image):
    """Return (label, confidence) for a PIL image."""
    model = _get_model()
    img_t = _VAL_TRANSFORM(image).unsqueeze(0)  # 1,C,H,W
    with torch.no_grad():
        out = model(img_t)
        probs = F.softmax(out, dim=1)
        conf, idx = probs.max(1)
        idx = int(idx.item())
        conf = float(conf.item())

    # safe label lookup
    if 0 <= idx < len(LABELS):
        label = LABELS[idx]
    else:
        label = f"class_{idx}"

    return label, conf


def predict_bytes(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict_pil(img)
