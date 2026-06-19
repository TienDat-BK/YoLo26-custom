import torch, sys
sys.path.insert(0, '.')
from Modules.Head import Detect

m = Detect(nc=4, end2end=True, strides=(8, 16, 32), ch=(64, 128, 256))
print("stride:", m.stride)

x = [torch.randn(2, 64, 80, 80, requires_grad=True),
     torch.randn(2, 128, 40, 40, requires_grad=True),
     torch.randn(2, 256, 20, 20, requires_grad=True)]

# --- Training ---
m.train()
out = m(x)
print("\n=== TRAINING ===")
for k, d in out.items():
    print(f"  {k}: boxes={d['boxes'].shape}  scores={d['scores'].shape}")

# Check gradient: one2many should flow grad to x, one2one should NOT
loss = out["one2many"]["boxes"].sum() + out["one2one"]["boxes"].sum()
loss.backward(retain_graph=True)

print("\n=== GRADIENT CHECK ===")
for i, xi in enumerate(x):
    has_grad = xi.grad is not None and xi.grad.abs().sum().item() > 0
    print(f"  x[{i}] has gradient from one2many: {has_grad}")

# Reset grad, test only one2one
for xi in x:
    xi.grad = None

loss2 = out["one2one"]["boxes"].sum()
loss2.backward()
print()
for i, xi in enumerate(x):
    has_grad = xi.grad is not None and xi.grad.abs().sum().item() > 0
    print(f"  x[{i}] grad from one2one ONLY (should be False): {has_grad}")

# --- Inference ---
m.eval()
with torch.no_grad():
    y, preds = m(x)
print("\n=== INFERENCE ===")
print(f"  y shape: {y.shape}  -> (B, max_det, 6) = [x1,y1,x2,y2, score, cls]")
print(f"  preds keys: {list(preds.keys())}")
