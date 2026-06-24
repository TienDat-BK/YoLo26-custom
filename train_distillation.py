from Modules import DistillationLoss
from torch._utils import _get_async_or_non_blocking
from operator import imod
import math
import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
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

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from Modules.Model import yolo26n_custom
from Modules.DistillationLoss import YOLO26DistillationLoss


# ==============================================================================
# 1. Teacher Feature Hook
# ==============================================================================
class TeacherDetectHook:
    """Capture Neck feature maps fed into Teacher Detect head."""
    def __init__(self, detect_module):
        self.hook = detect_module.register_forward_hook(self._fn)
        self.features = None

    def _fn(self, module, input, output):
        self.features = input[0]   # list [p3, p4, p5]

    def close(self):
        self.hook.remove()


# ==============================================================================
# 2. COCO Person Dataset  (folder + instances JSON, no extra library)
# ==============================================================================
class COCOPersonDataset(Dataset):
    """
    Filter COCO images that contain at least one 'person' annotation.
    Needs:
      img_dir  — folder of raw JPEG images (e.g. train2017/)
      ann_file — instances JSON (e.g. annotations/instances_train2017.json)
    """
    PERSON_CAT_ID = 1   # Fixed in every COCO release

    def __init__(self, img_dir, ann_file, max_samples=None, img_size=640, cache = True):
        self.img_size = img_size
        self.cache = cache
        print(f"[Data] Parsing annotation file: {ann_file}")
        with open(ann_file, "r") as f:
            coco = json.load(f)

        # image_ids that have ≥1 person annotation
        person_ids = set(
            a["image_id"] for a in coco["annotations"]
            if a["category_id"] == self.PERSON_CAT_ID
        )
        id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        paths = [
            os.path.join(img_dir, id2file[i])
            for i in person_ids if i in id2file
        ]
        paths = [p for p in paths if os.path.exists(p)]

        if max_samples:
            paths = paths[:max_samples]
        self.paths = paths

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]) if T else None

        self.img_cache = []
        
        if self.cache:
            print("Caching Dataset ...............")
            for idx in range(len(self.paths)):
                img = Image.open(self.paths[idx]).convert("RGB")
                img_resized = img.resize((self.img_size, self.img_size))
                img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).contiguous()
                self.img_cache.append(img_tensor)
            print("Caching Done!!!!!!!!!!!!!")

        print(f"[Data] {len(self.paths)} person images loaded "
              f"(cap={max_samples}).")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            if self.cache:
                return self.img_cache[idx].float() / 255.0
            else:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img) if self.transform else img
        except Exception:
            return self.__getitem__(0)


# ==============================================================================
# 3. Main
# ==============================================================================
PERSON_IDX = 0   # Index of 'person' in COCO 80-class Teacher output

def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Knowledge Distillation — Online (Teacher+Student per step)"
    )
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int,   default=10000,
                        help="Max COCO person images")
    parser.add_argument("--img-size",    type=int,   default=640)
    parser.add_argument("--data-dir",    type=str,   default="/content/train2017",
                        help="COCO images folder")
    parser.add_argument("--ann-file",    type=str,
                        default="/content/annotations/instances_train2017.json",
                        help="COCO instances annotation JSON")
    parser.add_argument("--weight",      type=str,   default="",
                        help="Student pretrain weight path")
    parser.add_argument("--save-path",   type=str,
                        default="yolo26n_person_distilled.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Device: {device}")

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    if os.path.isdir(args.data_dir) and os.path.isfile(args.ann_file):
        dataset = COCOPersonDataset(
            img_dir     = args.data_dir,
            ann_file    = args.ann_file,
            max_samples = args.max_samples,
            img_size    = args.img_size,
        )
    else:
        print("[Data] data-dir / ann-file not found — using dummy dataset.")
        class _Dummy(Dataset):
            def __len__(self): return 32
            def __getitem__(self, i): return torch.randn(3, args.img_size, args.img_size)
        dataset = _Dummy()

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Student  (nc=1 — person only)
    # ------------------------------------------------------------------
    print("\n[Model] Student: YOLO26 Nano  nc=1")
    student = yolo26n_custom(nc=80, end2end=True).to(device)
    if args.weight:
        sd = torch.load(args.weight, map_location=device)
        student.load_state_dict(sd)

        # KO ĐÓNG BĂNG BACKBONE
        # for p in student.backbone.parameters():
        #     p.requires_grad = False

        print(f"[Model] Loaded pretrain weight: {args.weight}")

    # ------------------------------------------------------------------
    # Teacher  (nc=80, frozen)
    # ------------------------------------------------------------------
    print("[Model] Teacher: YOLO26m  nc=80")
    saved_paths = list(sys.path)
    try:
        sys.path = [p for p in sys.path
                    if p != "" and os.path.abspath(p) != os.path.abspath(".")]
        from ultralytics import YOLO
        teacher = YOLO("yolo26m.pt").model.to(device)
        print("[Model] Teacher loaded from Ultralytics.")
    except ImportError:
        print("[Model] Ultralytics missing — using mock large YOLO26_Custom.")
        from Modules.Model import YOLO26_Custom
        teacher = YOLO26_Custom(nc=80, end2end=True, w=1.00, d=1.00, mc=512).to(device)
    finally:
        sys.path = saved_paths

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    detect_module = teacher.model[-1] if hasattr(teacher, "model") else teacher.head
    teacher_hook  = TeacherDetectHook(detect_module)

    # ------------------------------------------------------------------
    # Loss & Optimizer
    # ------------------------------------------------------------------
    # Student Neck ch (w=0.25): [64, 128, 256]
    # Teacher Neck ch (w=1.00): [256, 512, 512]
    distill_loss = YOLO26DistillationLoss(
        student_channels = (64, 128, 256),
        teacher_channels = (256, 512, 512),
        tau = 1.5,
    ).to(device)

    trainable = filter(lambda p: p.requires_grad, student.parameters())
    optimizer  = optim.AdamW(
        list(trainable) + list(distill_loss.parameters()),
        lr           = args.lr,
        weight_decay = 1e-5,
    )
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_step = len(loader)* args.epochs
    
    lrScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max=total_step,
        eta_min=1e-6,
    )

    # ------------------------------------------------------------------
    # Training loop  — Teacher + Student forward every step
    # ------------------------------------------------------------------
    print(f"\n[Train] {args.epochs} epochs  |  {len(loader)} steps/epoch")
    student.train()
    distill_loss.train()

    initiative_feat_weight = 0
    final_feat_weight = 0

    num_itr = 0

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs in pbar:
            num_itr+=1
            imgs = imgs.to(device, non_blocking=True)

            # --- Teacher forward (no grad, no weight update) ---
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    t_out = teacher(imgs)
                    if isinstance(t_out, tuple):
                        t_out = t_out[1]   # raw dict

                    t_feats = teacher_hook.features   # [p3, p4, p5] on GPU

                    # Slice to person class only → nc=1
                    # Teacher scores shape: [B, 80, num_queries]
                    t_preds = {
                        branch: {
                            "scores": t_out[branch]["scores"],
                            "boxes":  t_out[branch]["boxes"],
                        }
                        for branch in ["one2many", "one2one"]
                    }

            # --- Student forward ---
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                s_preds, s_feats = student(imgs, return_features=True)

            # --- Distillation losses ---
            losses = distill_loss(
                student_outputs    = s_preds,
                teacher_outputs    = t_preds,
                student_neck_feats = s_feats,
                teacher_neck_feats = t_feats,
            )

            loss_decay = 0.5 * (1 + math.cos(math.pi * num_itr / total_step))
            feat_weight = final_feat_weight + (initiative_feat_weight - final_feat_weight) * loss_decay

            loss_feat  = losses["loss_feat"]
            loss_cls   = losses["loss_cls"]
            loss_bbox_many  = losses["loss_bbox_many"]
            loss_bbox_one = losses["loss_bbox_one"]
            total_loss = feat_weight * loss_feat + 1.0 * loss_cls + 1 * loss_bbox_many + 1 * loss_bbox_one

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()


            lrScheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Feat": f"{loss_feat.item():.4f}",
                "Cls":  f"{loss_cls.item():.4f}",
                "BBox_many": f"{loss_bbox_many.item():.4f}",
                "BBox_one": f"{loss_bbox_one.item():.4f}",
                "LR": f"{current_lr:.6f}",
                "Feat_weigh": f"{feat_weight:.6f}",
            })

    # ------------------------------------------------------------------
    # Save student weights
    # ------------------------------------------------------------------
    teacher_hook.close()
    torch.save(student.state_dict(), args.save_path)
    print(f"\n[Done] Student saved to '{args.save_path}'")


if __name__ == "__main__":
    main()
