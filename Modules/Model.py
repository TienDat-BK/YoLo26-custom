import torch
import torch.nn as nn
from Modules.Backbone import Backbone, make_divisible
from Modules.Neck import Neck
from Modules.Head import Detect

class YOLO26_Custom(nn.Module):
    def __init__(self, nc=80, end2end=True, w=0.25, d=0.50, mc=1024):
        super().__init__()
        # 1. Backbone
        self.backbone = Backbone(w=w, d=d, mc=mc)
        
        # 2. Neck
        self.neck = Neck(w=w, d=d, mc=mc)
        
        # 3. Calculate Neck output channel dimensions
        c3 = make_divisible(min(256, mc) * w)  # 64
        c4 = make_divisible(min(512, mc) * w)  # 128
        c5 = make_divisible(min(1024, mc) * w) # 256
        
        # 4. Detect Head
        self.head = Detect(nc=nc, end2end=end2end, strides=(8, 16, 32), ch=(c3, c4, c5))

    def forward(self, x, return_features=False):
        feat_backbone = self.backbone(x)
        feat_neck = self.neck(feat_backbone)
        preds = self.head(feat_neck)
        if return_features:
            return preds, feat_neck
        return preds

class yolo26n_custom(YOLO26_Custom):
    """Wrapper class specialized for YOLO26 Nano configuration."""
    def __init__(self, nc=80, end2end=True):
        super().__init__(nc=nc, end2end=end2end, w=0.25, d=0.50, mc=1024)
