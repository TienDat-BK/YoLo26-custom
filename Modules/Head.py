import torch
import torch.nn.functional as F
from torch import nn
from Modules.blocks import Conv, DWConv
import copy
import math


class Detect(nn.Module):
    """YOLO Detect head — end-to-end, no DFL.

    Boxes are predicted directly as (x1, y1, x2, y2) in feature-map space
    and are scaled to pixel space by per-level strides at inference time.

    Attributes:
        dynamic (bool): Force stride-vec recompute every forward (for dynamic input sizes).
        export (bool): Export mode — return y only (no preds dict).
        max_det (int): Maximum detections per image.
        agnostic_nms (bool): Class-agnostic NMS.
        nc (int): Number of classes.
        nl (int): Number of detection levels.
        no (int): Outputs per anchor (4 + nc).
        stride (torch.Tensor): Per-level stride values — registered buffer, moves with .cuda().
        cv2 (nn.ModuleList): Box regression heads (one-to-many).
        cv3 (nn.ModuleList): Classification heads (one-to-many).
        one2one_cv2 (nn.ModuleList): Box regression heads (one-to-one).
        one2one_cv3 (nn.ModuleList): Classification heads (one-to-one).
    """

    dynamic      = False
    export       = False
    max_det      = 300
    agnostic_nms = False

    def __init__(
        self,
        nc: int = 80,
        end2end: bool = True,
        strides: tuple = (8, 16, 32),
        ch: tuple = (),
    ):
        """Initialize Detect head.

        Args:
            nc (int): Number of classes.
            end2end (bool): Use end-to-end (one-to-one + one-to-many) heads.
            strides (tuple): Downsampling strides per detection level relative to
                the original input image, e.g. (8, 16, 32) means the first feature
                map has 1 cell per 8x8 pixels. Must have the same length as `ch`.
            ch (tuple): Input channel sizes from the backbone/neck for each level.
        """
        super().__init__()
        assert len(strides) == len(ch), (
            f"len(strides)={len(strides)} must equal len(ch)={len(ch)}"
        )
        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 4  # outputs per anchor: 4 box coords + nc scores

        # FIX 1: register_buffer → stride auto-moves with .to(device) / .cuda()
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float))

        c2 = max(16, ch[0] // 4)        # hidden channels for box head
        c3 = max(ch[0], min(nc, 100))   # hidden channels for cls head

        # one-to-many heads (auxiliary, used during training)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        # one-to-one heads (primary inference branch)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        # FIX 2: stride_vec as buffer so it persists across calls and moves with .cuda()
        # Initialized to None; built on first forward() call
        self.register_buffer("_stride_vec", None)
        self._cached_feat_shape: tuple | None = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def one2many(self) -> dict:
        """One-to-many head components (box + cls)."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self) -> dict:
        """One-to-one head components (box + cls)."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self) -> bool:
        """True if one-to-one heads were created."""
        return hasattr(self, "one2one_cv2")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList,
        cls_head: nn.ModuleList,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run one detection head over all feature levels and concatenate.

        Returns:
            dict with keys:
                "boxes"  : (B, 4,  A) - raw box coords in feature-map space
                "scores" : (B, nc, A) - raw class logits (pre-sigmoid)
                "feats"  : list[Tensor] - original feature maps (for stride lookup)
        """
        bs = x[0].shape[0]
        # FIX 3: no Python-level branching on None (heads are always valid here)
        boxes  = torch.cat([box_head[i](x[i]).view(bs, 4,       -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x: list[torch.Tensor]):
        """Forward pass.

        Training output (end2end=True):
            {
                "one2many": {"boxes":(B,4,A), "scores":(B,nc,A), "feats":[...]},
                "one2one" : {"boxes":(B,4,A), "scores":(B,nc,A), "feats":[...]},
            }
            NOTE: Both branches receive full gradients during training.

        Inference output (export=False):
            (y, preds)
            y     : (B, max_det, 6) - [x1, y1, x2, y2, score, class_idx]
            preds : same dict as training (useful for val loss)

        Inference output (export=True):
            y : (B, max_det, 6)
        """
        # one-to-many pass (auxiliary branch, always computed)
        preds = self.forward_head(x, **self.one2many)

        if self.end2end:
            # one-to-one pass — detach features so this branch does NOT
            # send gradients back to the neck/backbone.
            # Only the one2one head weights are updated via this branch.
            x_det   = [xi.detach() for xi in x]
            one2one = self.forward_head(x_det, **self.one2one)
            preds   = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds

        # ── inference path ────────────────────────────────────────────────────
        raw = preds["one2one"] if self.end2end else preds
        y = self._inference(raw)   # (B, max_det, 6)
        return y if self.export else (y, preds)

    # ── inference helpers ─────────────────────────────────────────────────────

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode boxes, sigmoid scores, then run postprocess.

        Returns:
            Tensor (B, max_det, 6): [x1,y1,x2,y2, score, class_idx]
        """
        dbox   = self._get_decode_boxes(x)                   # (B, 4,  A)
        scores = x["scores"].sigmoid()                        # (B, nc, A)
        # FIX 5: keep channel-first (B, 4+nc, A), then do a single permute inside postprocess
        # avoids a redundant contiguous() copy
        fused  = torch.cat((dbox, scores), dim=1)            # (B, 4+nc, A)
        return self.postprocess(fused.permute(0, 2, 1))       # (B, max_det, 6)

    def _build_stride_vec(self, feats: list[torch.Tensor]) -> None:
        strides = []
        grids = []
        
        for i, f in enumerate(feats):
            _, _, h, w = f.shape
            stride_val = self.stride[i]
            
            # Tạo tọa độ (x, y) cho từng ô lưới
            y, x = torch.meshgrid(torch.arange(h, device=f.device),
                                torch.arange(w, device=f.device), indexing='ij')
            
            # Lấy tâm của ô lưới (+0.5)
            grid = torch.stack([x, y], dim=-1).float() + 0.5  # (h, w, 2)
            
            # Mạng của bạn output 4 tọa độ (x1, y1, x2, y2)
            # Ta nhân đôi grid lên để cộng (grid_x, grid_y) cho cả 2 điểm
            grid = grid.repeat(1, 1, 2).view(-1, 4)           # (h*w, 4)
            grids.append(grid)
            
            # Tạo mảng stride tương ứng
            strides.append(torch.full((h * w, 1), stride_val, device=f.device))

        # Đăng ký thành buffer để tính toán cực nhanh qua các Batch
        self.register_buffer('_anchor_grid', torch.cat(grids, dim=0).unsqueeze(0).permute(0, 2, 1)) # (1, 4, A)
        self.register_buffer('_stride_vec', torch.cat(strides, dim=0).unsqueeze(0).permute(0, 2, 1)) # (1, 1, A)
        self._cached_feat_shape = tuple(f.shape[2:] for f in feats)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = x["feats"]
        feat_shape = tuple(f.shape[2:] for f in feats)
        if self.dynamic or self._cached_feat_shape != feat_shape:
            self._build_stride_vec(feats)

        # ĐÃ SỬA: Phải cộng hệ quy chiếu Grid vào trước khi nhân Stride!
        # Công thức: x_img = (x_feat + grid) * stride
        return (x["boxes"] + self._anchor_grid) * self._stride_vec

        # ── post-processing ───────────────────────────────────────────────────────

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Select top-k detections.

        Args:
            preds (Tensor): (B, A, 4+nc) - boxes + class scores.

        Returns:
            Tensor (B, max_det, 6): [x1, y1, x2, y2, max_score, class_idx].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)  # (B,A,4), (B,A,nc)
        
        # Hứng đúng tên bản chất
        topk_scores, topk_class_idx, topk_anchor_idx = self.get_topk_index(scores, self.max_det)

        boxes = boxes.gather(dim=1, index=topk_anchor_idx.expand(-1, -1, 4))
        return torch.cat([boxes, topk_scores, topk_class_idx], dim=-1)

    def get_topk_index(
        self, scores: torch.Tensor, max_det: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top-k detection indices by max class score.

        Args:
            scores (Tensor): (B, A, nc)
            max_det (int): k

        Returns:
            scores (B, k, 1), class_idx (B, k, 1), anchor_idx (B, k, 1)
        """
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)

        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices

        # FIX 8: replace idx.repeat with idx.expand — no memory copy
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)         # (B, k, 1)
        scores    = scores.gather(dim=1, index=ori_index.expand(-1, -1, nc))
        scores, index = scores.flatten(1).topk(k)

        # FIX 9: cache arange on same device to avoid re-allocation every call
        batch_idx = torch.arange(batch_size, device=scores.device).unsqueeze(1)  # (B,1)
        idx = ori_index[batch_idx, index // nc]  # (B, k, 1)

        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Drop the one-to-many head to speed up inference."""
        self.cv2 = self.cv3 = None
