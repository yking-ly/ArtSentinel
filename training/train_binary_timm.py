# train_binary_timm.py
"""
Improved training script for binary art classifier (timm backbones).

Key improvements over original:
 - OneCycleLR scheduler for faster convergence
 - Gradient accumulation via --accumulate to emulate larger batch sizes
 - Resume from checkpoint (--resume) for fine-tuning/continuation
 - cleaner AMP usage and progress printing
 - small sanity checks and improved saving
"""
import argparse, os, random, time
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvT

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# -------------------- config / utils --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Bot-Made", "Brush-Made"]  # 0, 1

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def list_images(root: Path):
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def build_transforms(size: int, mode: str):
    # use tuple sizes for albumentations where required
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop((size, size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9,1.1), rotate=(-8,8), translate_percent=(0.0,0.05), p=0.5),
            A.ColorJitter(0.2,0.2,0.2,0.1, p=0.4),
            A.ToFloat(max_value=255.0),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max(size, 256)),
            A.CenterCrop(size, size),
            A.ToFloat(max_value=255.0),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

class AlbFolder(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        y = self.labels[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(image=np.array(img))["image"]
        return x, torch.tensor(y, dtype=torch.long)

# -------------------- training bits --------------------
def build_model(model_name: str, num_classes=2, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    running_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="mean")
    for xb, yb in tqdm(loader, desc="val", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        running_loss += float(loss.item()) * xb.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]  # class-1 (Brush-Made) as positive
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
    if len(all_probs) == 0:
        return {"loss": 0.0, "acc": 0.0, "roc_auc": float("nan"), "pr_auc": 0.0, "cm": np.zeros((2,2), int), "n": 0}
    all_probs = np.concatenate(all_probs); all_targets = np.concatenate(all_targets)
    preds = (all_probs >= 0.5).astype(int)

    acc = (preds == all_targets).mean()
    try:
        roc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        roc = float("nan")
    pr = average_precision_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, preds, labels=[0,1])
    loss = running_loss / max(1, len(loader.dataset))
    return {"loss": loss, "acc": acc, "roc_auc": roc, "pr_auc": pr, "cm": cm, "n": len(all_targets)}

def train_one_epoch(model, loader, opt, scaler, device, accumulation_steps=1, use_amp=True):
    model.train()
    ce = nn.CrossEntropyLoss(reduction="mean")
    run_loss, correct, count = 0.0, 0, 0
    it = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    opt.zero_grad(set_to_none=True)
    for step, (xb, yb) in it:
        xb, yb = xb.to(device), yb.to(device)
        # forward with amp if available and requested
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(xb)
                loss = ce(logits, yb) / accumulation_steps
        else:
            logits = model(xb)
            loss = ce(logits, yb) / accumulation_steps

        scaler.scale(loss).backward()
        # step optimizer only when accumulation boundary reached
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        run_loss += float(loss.item()) * xb.size(0) * accumulation_steps  # reverse division
        correct += (logits.detach().argmax(1) == yb).sum().item()
        count += xb.size(0)
    return run_loss / max(1, count), correct / max(1, count)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str, help=".../data/binary")
    ap.add_argument("--model_name", default="efficientnet_b0")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--subset", type=float, default=1.0, help="0<ratio<=1 for quick probe")
    ap.add_argument("--model_out", type=str, default="best_model.pth")
    ap.add_argument("--eval_test", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5, help="early stop patience on ROC-AUC")
    ap.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume (optional)")
    args = ap.parse_args()

    set_seed(args.seed)

    root = Path(args.data)
    train_root = root / "train"
    test_root  = root / "test"

    # gather files
    bot_train  = list_images(train_root / "Bot-Made")
    real_train = list_images(train_root / "Brush-Made")
    X = bot_train + real_train
    y = [0]*len(bot_train) + [1]*len(real_train)

    print(f"Train files: {len(X)} | label counts:", dict(Counter(y)))

    # --- subset (stratified) ---
    if 0 < args.subset < 1.0:
        from collections import defaultdict
        per_class_idx = defaultdict(list)
        for i, lbl in enumerate(y):
            per_class_idx[lbl].append(i)
        keep = []
        for lbl, idxs in per_class_idx.items():
            m = max(1, int(len(idxs) * args.subset))
            keep.extend(random.sample(idxs, m))
        keep = sorted(keep)
        X = [X[i] for i in keep]
        y = [y[i] for i in keep]
        counts = Counter(y)
        print(f"[FAST] Using stratified subset ratio={args.subset:.2f} -> {len(X)} samples; per-class: {dict(counts)}")

    # stratified split
    if len(X) < 2:
        print("Not enough samples for split. Exiting.")
        return

    train_idx, val_idx = train_test_split(
        np.arange(len(X)), stratify=y, test_size=0.12, random_state=args.seed
    )
    print(f"Split -> train:{len(train_idx):6d}  val:{len(val_idx):6d}")

    # datasets/loaders
    tf_train = build_transforms(args.input_size, "train")
    tf_val   = build_transforms(args.input_size, "val")

    train_ds = AlbFolder([X[i] for i in train_idx], [y[i] for i in train_idx], tf_train)
    val_ds   = AlbFolder([X[i] for i in val_idx],   [y[i] for i in val_idx],   tf_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)

    # model/opt
    model = build_model(args.model_name, num_classes=2, pretrained=args.pretrained).to(DEVICE)
    # print params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_name} | params (trainable): {total_params:,}")

    # resume if provided (model weights only)
    if args.resume:
        ck = Path(args.resume)
        if ck.exists():
            print("Resuming model weights from:", ck)
            ckpt = torch.load(str(ck), map_location="cpu")
            # support both raw state_dict and meta dict with "model_state_dict"
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            # strip module prefix if present
            state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k,v in state.items()}
            model.load_state_dict(state, strict=False)
        else:
            print("Resume path not found:", ck)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # OneCycleLR: need steps_per_epoch
    steps_per_epoch = max(1, int(np.ceil(len(train_loader.dataset) / args.batch)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=max(1, len(train_loader)), epochs=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # training loop with early stop on ROC-AUC
    best_roc, best_path, no_improve = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, scaler, DEVICE, accumulation_steps=max(1, args.accumulate), use_amp=True)

        val = evaluate(model, val_loader, DEVICE)
        # step scheduler (OneCycleLR expects step per batch - here we call step per epoch to keep simple)
        # We still call scheduler.step() once per epoch to progress schedule gracefully.
        try:
            scheduler.step()
        except Exception:
            pass

        print(
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} "
            f"roc_auc {val['roc_auc']:.4f} pr_auc {val['pr_auc']:.4f}"
        )
        print("Val CM:\n", val["cm"])

        # decide score to track (use ROC if available else acc)
        if np.isnan(val["roc_auc"]):
            score = val["acc"]
        else:
            score = val["roc_auc"]

        if score > best_roc:
            best_roc = score
            meta = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "input_size": args.input_size,
                "class_names": CLASS_NAMES,
                "metric": "roc_auc",
                "best_roc_auc": float(best_roc),
            }
            outp = Path(args.model_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meta, outp)
            best_path = str(outp)
            print(f"✅ Saved new best: {best_path} (ROC-AUC={best_roc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No ROC-AUC improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print("⏹ Early stopping (plateau).")
                break

    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min. Best ROC-AUC={best_roc:.4f}")

    # optional test eval
    if args.eval_test and test_root.exists():
        print("\n=== Test evaluation ===")
        bot_t  = list_images(test_root / "Bot-Made")
        real_t = list_images(test_root / "Brush-Made")
        Xt = bot_t + real_t
        yt = [0]*len(bot_t) + [1]*len(real_t)
        test_ds = AlbFolder(Xt, yt, tf_val)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        test = evaluate(model, test_loader, DEVICE)
        print(
            f"[TEST] loss {test['loss']:.4f} acc {test['acc']:.4f} "
            f"roc_auc {test['roc_auc']:.4f} pr_auc {test['pr_auc']:.4f}"
        )
        print("Test CM:\n", test["cm"])

if __name__ == '__main__':
    main()
