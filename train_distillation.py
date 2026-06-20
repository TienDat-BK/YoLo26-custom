import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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

# --- 1. Teacher Feature Hook Setup ---
class TeacherDetectHook:
    """Hook to capture intermediate Neck outputs from Teacher Detect head."""
    def __init__(self, detect_module):
        self.hook = detect_module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        # input[0] contains the input features to the Detect head: [p3, p4, p5]
        self.features = input[0]

    def close(self):
        self.hook.remove()

# --- 2. Dummy Dataset for testing setup ---
class DummyCOCODataset(Dataset):
    def __init__(self, size=64):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Dummy image of size 640x640
        x = torch.randn(3, 640, 640)
        return x

def main():
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description="YOLO26 Knowledge Distillation Training Script")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print("=== Preparing Google Colab Distillation setup for YOLO26 ===")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Instantiate Student Model ---
    nc = 80  # COCO dataset classes
    print("\nInitializing Student Model (YOLO26 Custom Nano)...")
    student = yolo26n_custom(nc=nc, end2end=True).to(device)
    print("Student successfully loaded!")

    # --- 4. Instantiate Teacher Model ---
    print("\nLoading Teacher Model (YOLO26 Large from Ultralytics)...")
    
    # Force python to import global/pip installed 'ultralytics' instead of local workspace folder
    saved_paths = list(sys.path)
    try:
        current_dir = os.path.abspath(".")
        sys.path = [p for p in sys.path if p != "" and os.path.abspath(p) != current_dir]
        
        from ultralytics import YOLO
        # Load pre-trained weights.
        # This will download the file 'yolo26l.pt' if it does not exist.
        teacher_wrapper = YOLO("yolo26l.pt")
        teacher = teacher_wrapper.model.to(device)  # Access the underlying nn.Module
        print("Teacher successfully loaded from global Ultralytics package!")
    except ImportError:
        print("Warning: Global 'ultralytics' library not installed.")
        print("If you are on Google Colab, please run: !pip install ultralytics")
        print("Mocking Teacher model for local testing...")
        # Mock teacher using custom class at large scale
        from Modules.Model import YOLO26_Custom
        teacher = YOLO26_Custom(nc=nc, end2end=True, w=1.00, d=1.00, mc=512).to(device)
    finally:
        sys.path = saved_paths

    # Freeze Teacher weights
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # --- 5. Register Hook on Teacher Detect head ---
    # In Ultralytics models, the last module model.model[-1] is the Detect head.
    detect_module = teacher.model[-1] if hasattr(teacher, "model") else teacher.head
    teacher_hook = TeacherDetectHook(detect_module)

    # --- 6. Loss and Optimizer ---
    # Student channels: [64, 128, 256], Teacher channels: [256, 512, 512]
    # Under w=0.25 (Nano) and w=1.00 (Large)
    distill_loss_fn = YOLO26DistillationLoss(
        student_channels=(64, 128, 256),
        teacher_channels=(256, 512, 512),
        tau=2.0
    ).to(device)

    # Trainable parameters include Student model and the Conv1x1 projections inside the Loss class
    optimizer = optim.AdamW(
        list(student.parameters()) + list(distill_loss_fn.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # --- 7. DataLoader Setup ---
    # Replace DummyCOCODataset with your actual COCO dataset loader during training.
    train_dataset = DummyCOCODataset(size=8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 8. Distillation Training Loop ---
    print(f"\nRunning training loop for {args.epochs} epochs...")
    student.train()
    
    for epoch in range(args.epochs):
        # progress bar loading effect
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, x in enumerate(pbar):
            x = x.to(device)
            
            # Forward pass: Student (returning intermediate features too)
            s_preds, s_feats = student(x, return_features=True)
            
            # Forward pass: Teacher (Features captured by forward hook)
            with torch.no_grad():
                t_preds = teacher(x)
                # If teacher outputs (y, preds) in eval mode, extract the preds dict
                if isinstance(t_preds, tuple):
                    t_preds = t_preds[1]
                t_feats = teacher_hook.features
            
            # Calculate Distillation Losses
            losses = distill_loss_fn(
                student_outputs=s_preds,
                teacher_outputs=t_preds,
                student_neck_feats=s_feats,
                teacher_neck_feats=t_feats
            )
            
            # Combine losses with weights
            loss_feat = losses["loss_feat"]
            loss_cls = losses["loss_cls"]
            loss_bbox = losses["loss_bbox"]
            
            total_loss = 1.0 * loss_feat + 2.0 * loss_cls + 1.5 * loss_bbox
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update progress bar details dynamically
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Feat": f"{loss_feat.item():.4f}",
                "Cls": f"{loss_cls.item():.4f}",
                "BBox": f"{loss_bbox.item():.4f}"
            })

    # Remove the hook
    teacher_hook.close()
    
    # --- 9. Save Student Model ---
    save_path = "yolo26n_custom_distilled.pt"
    print(f"\nSaving trained student model weights to {save_path}...")
    torch.save(student.state_dict(), save_path)
    print("Model saved successfully!")
    
    print("\nSetup verified and ready to run on Google Colab!")

if __name__ == "__main__":
    main()
