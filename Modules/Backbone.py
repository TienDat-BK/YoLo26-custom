import math
import torch
import torch.nn as nn 
from Modules.blocks import Conv, C3k2

def make_divisible(x, divisor=8):
    """Ensure channel dimensions are divisible by 8."""
    return int(math.ceil(x / divisor) * divisor)

class Backbone(nn.Module):
    def __init__(self, w=0.25, d=0.50, mc=1024):
        super().__init__()
        
        # YOLO26 scaling calculations with capping
        c1 = make_divisible(min(64, mc) * w)   # 16
        c2 = make_divisible(min(128, mc) * w)  # 32
        c3 = make_divisible(min(256, mc) * w)  # 64
        c4 = make_divisible(min(512, mc) * w)  # 128
        c5 = make_divisible(min(1024, mc) * w) # 256

        
        # Depth scaling (number of bottleneck blocks)
        n = max(round(2 * d), 1)      # 1
        
        self.sq1 = nn.Sequential(
            Conv(3, c1, k=3, s=2),
            Conv(c1, c2, k=3, s=2),
            C3k2(c2, c3, n=n, e=0.25, c3k=False),
            Conv(c3, c3, k=3, s=2),
            C3k2(c3, c4, n=n, e=0.25, c3k=False),
        )

        self.sq2 = nn.Sequential(
            Conv(c4, c4, k=3, s=2),
            C3k2(c4, c4, n=n, e=0.5, c3k=True),
        )

        self.sq3 = nn.Sequential(
            Conv(c4, c5, k=3, s=2),
            C3k2(c5, c5, n=n, e=0.5, c3k=True),
        )
    
    def forward(self, x) -> list[torch.Tensor]:
        y1 = self.sq1(x)
        y2 = self.sq2(y1)
        y3 = self.sq3(y2)
        return [y1, y2, y3]





