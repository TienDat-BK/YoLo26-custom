import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torchvision.transforms as T
except ImportError:
    T = None

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            pass
        def set_description(self, *args, **kwargs):
            pass

# Ensure Modules directory is in path
sys.path.insert(0, os.path.abspath("."))

from Modules.Model import yolo26n_custom
from Modules.DistillationLoss import YOLO26DistillationLoss


# ==============================================================================
# --- 1. Teacher Feature Hook ---
# ==============================================================================
class TeacherDetectHook:
    """Hook to capture intermediate Neck feature inputs to the Teacher Detect head."""
    def __init__(self, detect_module):
        self.hook = detect_module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        # input[0] = list of neck feature maps [p3, p4, p5] fed into Detect head
        self.features = input[0]

    def close(self):
        self.hook.remove()


# ==============================================================================
# --- 2. COCO Person Dataset (folder + JSON annotation, no external library) ---
# ==============================================================================
class COCOPersonDataset(Dataset):
    """
    Reads images from a COCO-format folder and filters only images
    that contain at least one 'person' annotation, using the instances JSON.

    Usage:
        dataset = COCOPersonDataset(
            img_dir  = "/content/train2017",
            ann_file = "/content/annotations/instances_train2017.json",
            max_samples = 10000,
            img_size = 640,
        )
    """
    # COCO category_id for 'person' is always 1
    PERSON_CAT_ID = 1

    def __init__(self, img_dir, ann_file, max_samples=None, img_size=640):
        import json
        self.img_dir  = img_dir
        self.img_size = img_size

        print(f"[Dataset] Parsing COCO annotation: {ann_file}")
        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Collect image_ids that have at least one person annotation
        person_img_ids = set(
            ann["image_id"] for ann in coco["annotations"]
            if ann["category_id"] == self.PERSON_CAT_ID
        )

        # Build id → filename map
        id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        # Filter + build full paths
        all_paths = [
            os.path.join(img_dir, id2file[img_id])
            for img_id in person_img_ids
            if img_id in id2file
        ]
        # Keep only files that actually exist on disk
        all_paths = [p for p in all_paths if os.path.exists(p)]

        # Optionally cap the number of samples
        if max_samples is not None:
            all_paths = all_paths[:max_samples]

        self.img_paths = all_paths

        if T is not None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])
        else:
            self.transform = None

        print(f"[Dataset] Found {len(self.img_paths)} person images (cap={max_samples}).")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        try:
            img = Image.open(self.img_paths[index]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception:
            return self.__getitem__(0)


# ==============================================================================
# --- 3. Cached Teacher Dataset (RAM) ---
# ==============================================================================
class CachedTeacherDataset(Dataset):
    """
    Dataset backed by in-RAM teacher cache.
    Each item is a dict: {'img', 't_preds', 't_feats'}
    """
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        return self.cache[index]


def cached_collate_fn(batch):
    """Custom collate to batch cached teacher outputs."""
    imgs = torch.stack([item["img"] for item in batch])
    t_preds = {
        "one2many": {
            "scores": torch.cat([item["t_preds"]["one2many"]["scores"] for item in batch], dim=0),
            "boxes":  torch.cat([item["t_preds"]["one2many"]["boxes"]  for item in batch], dim=0),
        },
        "one2one": {
            "scores": torch.cat([item["t_preds"]["one2one"]["scores"]  for item in batch], dim=0),
            "boxes":  torch.cat([item["t_preds"]["one2one"]["boxes"]   for item in batch], dim=0),
        },
    }
    t_feats = [
        torch.cat([item["t_feats"][i] for item in batch], dim=0)
        for i in range(len(batch[0]["t_feats"]))
    ]
    return imgs, t_preds, t_feats


# ==============================================================================
# --- 4. Phase 1: Cache all Teacher outputs in RAM ---
# ==============================================================================
PERSON_CLASS_IDX = 0  # In COCO 80-class order, 'person' is index 0

def cache_teacher_outputs(teacher, teacher_hook, loader, device):
    """
    Run Teacher on the entire dataset once and store all outputs in RAM.
    Teacher scores/boxes are sliced to person class only (nc=1) before caching.

    Returns:
        cache (list of dict): keys = 'img', 't_preds', 't_feats' (all on CPU)
    """
    teacher.eval()
    cache = []

    print("\n[Phase 1] Running Teacher forward pass to build RAM cache...")
    pbar = tqdm(loader, desc="[Teacher Cache]")

    with torch.no_grad():
        for imgs in pbar:
            imgs = imgs.to(device)

            t_preds = teacher(imgs)
            # Ultralytics eval mode may return (postprocessed, raw_preds) tuple
            if isinstance(t_preds, tuple):
                t_preds = t_preds[1]  # raw dict: {'one2many': {...}, 'one2one': {...}}

            # Capture neck features from hook
            t_feats_gpu = teacher_hook.features  # list of Tensor [B, C, H, W]

            # Slice Teacher output to person class only (index 0 → keep as nc=1)
            t_preds_person = {}
            for branch in ["one2many", "one2one"]:
                scores = t_preds[branch]["scores"]  # [B, num_queries, 80]
                boxes  = t_preds[branch]["boxes"]   # [B, num_queries, 4]
                t_preds_person[branch] = {
                    "scores": scores[:, :, PERSON_CLASS_IDX : PERSON_CLASS_IDX + 1].cpu(),
                    "boxes":  boxes.cpu(),
                }

            # Move neck features to CPU
            t_feats_cpu = [f.cpu() for f in t_feats_gpu]

            # Store per-sample so DataLoader can batch freely
            B = imgs.shape[0]
            for b in range(B):
                cache.append({
                    "img": imgs[b].cpu(),
                    "t_preds": {
                        branch: {
                            "scores": t_preds_person[branch]["scores"][b].unsqueeze(0),
                            "boxes":  t_preds_person[branch]["boxes"][b].unsqueeze(0),
                        }
                        for branch in ["one2many", "one2one"]
                    },
                    "t_feats": [f[b].unsqueeze(0) for f in t_feats_cpu],
                })

    print(f"[Phase 1] Done. Cached {len(cache)} samples in RAM.")
    return cache


# ==============================================================================
# --- 5. Main ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Distillation: COCO Person Only — 2-Phase RAM Cache Strategy"
    )
    parser.add_argument("--epochs",      type=int,   default=10,
                        help="Number of student training epochs (Phase 2)")
    parser.add_argument("--batch-size",  type=int,   default=4,
                        help="Batch size for both phases")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate for AdamW")
    parser.add_argument("--max-samples", type=int,   default=10000,
                        help="Max number of COCO person images to use")
    parser.add_argument("--img-size",    type=int,   default=640,
                        help="Input image resolution")
    parser.add_argument("--data-dir",    type=str,   default="/content/train2017",
                        help="Path to COCO images folder (e.g. train2017/)")
    parser.add_argument("--ann-file",    type=str,   default="/content/annotations/instances_train2017.json",
                        help="Path to COCO instances annotation JSON")
    parser.add_argument("--weight",      type=str,   default="",
                        help="Path to student pretrain weight (.pt)")
    parser.add_argument("--save-path",   type=str,   default="yolo26n_person_distilled.pt",
                        help="Output path for distilled student weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")

    # ------------------------------------------------------------------
    # Step A: Load COCO Person images from folder + instances JSON
    # ------------------------------------------------------------------
    print("\n[Data] Loading COCO person images from folder + annotation JSON...")
    if os.path.isdir(args.data_dir) and os.path.isfile(args.ann_file):
        raw_dataset = COCOPersonDataset(
            img_dir     = args.data_dir,
            ann_file    = args.ann_file,
            max_samples = args.max_samples,
            img_size    = args.img_size,
        )
    else:
        print(f"[Data] WARNING: --data-dir '{args.data_dir}' or "
              f"--ann-file '{args.ann_file}' not found.")
        print("[Data] Falling back to dummy dataset for pipeline testing.")
        class _DummyDS(Dataset):
            def __len__(self): return 32
            def __getitem__(self, i): return torch.randn(3, args.img_size, args.img_size)
        raw_dataset = _DummyDS()
        print(f"[Data] Dummy dataset: {len(raw_dataset)} synthetic samples.")

    raw_loader = DataLoader(
        raw_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # Keep order for deterministic caching
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Step B: Initialize Student Model — nc=1 (person only)
    # ------------------------------------------------------------------
    print("\n[Model] Initializing Student (YOLO26 Nano, nc=1 — person only)...")
    student = yolo26n_custom(nc=1, end2end=True).to(device)
    if args.weight:
        state_dict = torch.load(args.weight, map_location=device)
        student.load_state_dict(state_dict)
        for p in student.backbone.parameters():
            p.requires_grad = False
        print(f"[Model] Student loaded pretrain weight: {args.weight}")
    print("[Model] Student ready.")

    # ------------------------------------------------------------------
    # Step C: Load Teacher Model — nc=80 (full COCO)
    # ------------------------------------------------------------------
    print("\n[Model] Loading Teacher (YOLO26m, nc=80) from Ultralytics...")
    saved_paths = list(sys.path)
    try:
        current_dir = os.path.abspath(".")
        sys.path = [p for p in sys.path if p != "" and os.path.abspath(p) != current_dir]
        from ultralytics import YOLO
        teacher_wrapper = YOLO("yolo26m.pt")
        teacher = teacher_wrapper.model.to(device)
        print("[Model] Teacher loaded from Ultralytics package.")
    except ImportError:
        print("[Model] Ultralytics not found — using mock large YOLO26_Custom as teacher.")
        from Modules.Model import YOLO26_Custom
        teacher = YOLO26_Custom(nc=80, end2end=True, w=1.00, d=1.00, mc=512).to(device)
    finally:
        sys.path = saved_paths

    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # Register hook on Teacher Detect head to capture Neck features
    detect_module = teacher.model[-1] if hasattr(teacher, "model") else teacher.head
    teacher_hook = TeacherDetectHook(detect_module)

    # ------------------------------------------------------------------
    # PHASE 1 — Cache all Teacher outputs in RAM
    # ------------------------------------------------------------------
    ram_cache = cache_teacher_outputs(teacher, teacher_hook, raw_loader, device)

    # Free Teacher from GPU — no longer needed after caching
    teacher_hook.close()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[Phase 1] Teacher removed from GPU. VRAM freed for student training.")

    # ------------------------------------------------------------------
    # Build Phase 2 DataLoader from RAM cache
    # ------------------------------------------------------------------
    cached_dataset = CachedTeacherDataset(ram_cache)
    cached_loader  = DataLoader(
        cached_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # Data already in RAM — no I/O workers needed
        collate_fn=cached_collate_fn,
    )

    # ------------------------------------------------------------------
    # Step D: Distillation Loss & Optimizer
    # ------------------------------------------------------------------
    # Student Neck channels (w=0.25 Nano): [64, 128, 256]
    # Teacher Neck channels (w=1.00 Large): [256, 512, 512]
    distill_loss_fn = YOLO26DistillationLoss(
        student_channels=(64, 128, 256),
        teacher_channels=(256, 512, 512),
        tau=2.0,
    ).to(device)

    student_para_trainable = filter(lambda p : p.requires_grad , student.parameters())
    optimizer = optim.AdamW(
        list(student_para_trainable) + list(distill_loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # PHASE 2 — Train Student from cached Teacher outputs
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Training Student for {args.epochs} epochs "
          f"| {len(cached_loader)} steps/epoch...")

    student.train()
    distill_loss_fn.train()

    for epoch in range(args.epochs):
        pbar = tqdm(cached_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, t_preds, t_feats in pbar:
            imgs = imgs.to(device)
            t_preds = {
                branch: {k: v.to(device) for k, v in t_preds[branch].items()}
                for branch in ["one2many", "one2one"]
            }
            t_feats = [f.to(device) for f in t_feats]

            # Student forward — returns (preds, neck_features)
            s_preds, s_feats = student(imgs, return_features=True)

            # Compute distillation losses
            losses = distill_loss_fn(
                student_outputs=s_preds,
                teacher_outputs=t_preds,
                student_neck_feats=s_feats,
                teacher_neck_feats=t_feats,
            )

            loss_feat  = losses["loss_feat"]
            loss_cls   = losses["loss_cls"]
            loss_bbox  = losses["loss_bbox"]
            total_loss = 1.0 * loss_feat + 1.0 * loss_cls + 1.5 * loss_bbox

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "Loss":  f"{total_loss.item():.4f}",
                "Feat":  f"{loss_feat.item():.4f}",
                "Cls":   f"{loss_cls.item():.4f}",
                "BBox":  f"{loss_bbox.item():.4f}",
            })

    # ------------------------------------------------------------------
    # Save distilled Student weights
    # ------------------------------------------------------------------
    print(f"\n[Save] Saving student weights to '{args.save_path}'...")
    torch.save(student.state_dict(), args.save_path)
    print(f"[Save] Done. Distilled model saved: {args.save_path}")


if __name__ == "__main__":
    main()
