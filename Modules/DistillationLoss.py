import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLO26DistillationLoss(nn.Module):
    def __init__(self, student_channels=(64, 128, 256), teacher_channels=(256, 512, 512), tau=2.0):
        super().__init__()
        self.tau = tau
        
        # Conv1x1 layer to map Student neck features to Teacher neck features channel-wise
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
        
        self.bce_soft = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, student_outputs, teacher_outputs, student_neck_feats, teacher_neck_feats):
        """
        Args:
            student_outputs (dict): Raw `preds` from Student containing "one2many" and "one2one".
            teacher_outputs (dict): Raw `preds` from Teacher containing "one2many" and "one2one".
            student_neck_feats (list[Tensor]): Neck feature outputs of Student ([S3, S4, S5]).
            teacher_neck_feats (list[Tensor]): Neck feature outputs of Teacher ([T3, T4, T5]).
        """
        # --- 1. Feature Map Distillation Loss (MSE) ---
        loss_feat = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_neck_feats, teacher_neck_feats)):
            proj_s_feat = self.proj_layers[i](s_feat)
            loss_feat += F.mse_loss(proj_s_feat, t_feat)
            
        # --- 2. Classification Distillation Loss (Sigmoid BCE with Soft Targets) ---
        loss_cls = 0.0
        for branch in ["one2many", "one2one"]:
            s_scores = student_outputs[branch]["scores"]  # Student raw logits
            t_scores = teacher_outputs[branch]["scores"]  # Teacher raw logits
            
            # Make Teacher target soft by applying sigmoid with temperature
            t_soft_targets = torch.sigmoid(t_scores / self.tau)
            
            # Compute soft-BCE loss
            loss_cls += self.bce_soft(s_scores / self.tau, t_soft_targets) * (self.tau ** 2)
            
        # --- 3. Bounding Box Distillation Loss (L1 on raw boxes) ---
        loss_bbox = 0.0
        for branch in ["one2many", "one2one"]:
            s_boxes = student_outputs[branch]["boxes"]  # Student raw boxes
            t_boxes = teacher_outputs[branch]["boxes"]  # Teacher raw boxes
            
            # L1 Loss directly on the coordinate logits
            loss_bbox += F.mse_loss(s_boxes, t_boxes)

        return {
            "loss_feat": loss_feat,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox
        }
