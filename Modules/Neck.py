import math
import torch
import torch.nn as nn
from Modules.blocks import Conv, C3k2, SPPF

def make_divisible(x, divisor=8):
    """Ensure channel dimensions are divisible by 8."""
    return int(math.ceil(x / divisor) * divisor)

class Neck(nn.Module):
    def __init__(self, d=0.50, w=0.25, mc=1024):
        super().__init__()
        
        # Calculate scaled channel sizes matching the backbone outputs
        c3 = make_divisible(min(256, mc) * w)  # 64 (P3 output)
        c4 = make_divisible(min(512, mc) * w)  # 128 (P4 output)
        c5 = make_divisible(min(1024, mc) * w) # 256 (P5 output)
        
        # Scaling for C3k2 repeats
        n_block = max(round(2 * d), 1)
        n_block_last = max(round(1 * d), 1)
        
        # --- Upsample / Top-down Path ---
        # SPPF (Block 9)
        self.sppf = SPPF(c5, c5, k=5, n=3, shortcut=True)
        
        # Upsample 11
        self.up11 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # C3k2 (Block 13)
        self.c3k2_13 = C3k2(c5 + c4, c4, n=n_block, c3k=True)
        
        # Upsample 14
        self.up14 = nn.Upsample(scale_factor=2, mode="nearest")
        
        # C3k2 (Block 16)
        self.c3k2_16 = C3k2(c4 + c4, c3, n=n_block, c3k=True)
        
        # --- Downsample / Bottom-up Path ---
        # Conv 17
        self.conv17 = Conv(c3, c3, k=3, s=2)
        
        # C3k2 (Block 19)
        self.c3k2_19 = C3k2(c3 + c4, c4, n=n_block, c3k=True)
        
        # Conv 20
        self.conv20 = Conv(c4, c4, k=3, s=2)
        
        # C3k2 (Block 22) - Note: attn=False as requested
        self.c3k2_22 = C3k2(c4 + c5, c5, n=n_block_last, c3k=True)

    def forward(self, y: list[torch.Tensor]) -> list[torch.Tensor]:
        # y = [y1, y2, y3] from Backbone
        # y1 shape: [B, c4, 80, 80] (P3)
        # y2 shape: [B, c4, 40, 40] (P4)
        # y3 shape: [B, c5, 20, 20] (P5)
        y1, y2, y3 = y
        
        # 1. SPPF (9)
        feat_sppf = self.sppf(y3) # [B, c5, 20, 20]
        
        # 2. Top-down path
        up_sppf = self.up11(feat_sppf) # [B, c5, 40, 40]
        concat12 = torch.cat([up_sppf, y2], dim=1) # [B, c5 + c4, 40, 40]
        feat13 = self.c3k2_13(concat12) # [B, c4, 40, 40]
        
        up13 = self.up14(feat13) # [B, c4, 80, 80]
        concat15 = torch.cat([up13, y1], dim=1) # [B, c4 + c4, 80, 80]
        p3_out = self.c3k2_16(concat15) # [B, c3, 80, 80] (P3 Output)
        
        # 3. Bottom-up path
        down_p3 = self.conv17(p3_out) # [B, c3, 40, 40]
        concat18 = torch.cat([down_p3, feat13], dim=1) # [B, c3 + c4, 40, 40]
        p4_out = self.c3k2_19(concat18) # [B, c4, 40, 40] (P4 Output)
        
        down_p4 = self.conv20(p4_out) # [B, c4, 20, 20]
        concat21 = torch.cat([down_p4, feat_sppf], dim=1) # [B, c4 + c5, 20, 20]
        p5_out = self.c3k2_22(concat21) # [B, c5, 20, 20] (P5 Output)
        
        return [p3_out, p4_out, p5_out]