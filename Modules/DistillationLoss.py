import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantityFocalLoss(nn.Module):
    def __init__(self, beta = 2, threshold = 0.25, gamma = 2.0, alpha = 0.25):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred : torch.Tensor, target : torch.Tensor, reduction : str = "yes", hard_label : bool = False):
        """
            pred : [B, nc, H, W] raw
            target : [B, nc, H, W] (non-raw)

            formula : |p - y|^beta * BCE(p, y)
            hardlabel -> filter anchor which > threshold
        """
        if hard_label:
            # 1. Standard Focal Loss
            hard_target = (target > self.threshold).float()
            p = torch.sigmoid(pred)
            
            # Tính toán p_t và alpha_t
            # Nếu hard_target == 1 -> p_t = p, alpha_t = self.alpha
            # Nếu hard_target == 0 -> p_t = 1 - p, alpha_t = 1 - self.alpha
            p_t = p * hard_target + (1 - p) * (1 - hard_target)
            alpha_t = self.alpha * hard_target + (1 - self.alpha) * (1 - hard_target)
            
            bce_loss = F.binary_cross_entropy_with_logits(pred, hard_target, reduction='none')
            focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)
            
            if reduction == 'none':
                return focal_weight * bce_loss
            else:
                with torch.no_grad():
                    num_obj = hard_target.sum().clamp(min=1.0)
                return (focal_weight * bce_loss).sum() / num_obj
        else:
            # Dùng QuantityFocalLoss: |p - y|^beta * BCE(pred, soft_target)
            p = torch.sigmoid(pred)
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            if reduction == 'none':
                return torch.abs(p - target) ** self.beta * bce_loss
            else:
                # mask lấy các ô > threshold
                with torch.no_grad():
                    num_obj = (target > self.threshold).sum().clamp(min=1.0)
                return (torch.abs(p - target) ** self.beta * bce_loss).sum() / num_obj

class YOLO26DistillationLoss(nn.Module, ):
    def __init__(self, student_channels=(64, 128, 256), teacher_channels=(256, 512, 512), tau=2.0, hard_label_for_o2o : bool = True):
        super().__init__()
        self.tau = tau
        self.hard_label_for_o2o = hard_label_for_o2o
        # Conv1x1 layer to map Student neck features to Teacher neck features channel-wise
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
        self.quantityFocalLoss = QuantityFocalLoss()

    def forward(self, student_outputs, teacher_outputs, student_neck_feats, teacher_neck_feats):
        """
        Args:
            student_outputs (dict): Raw `preds` from Student containing "one2many" and "one2one".
            teacher_outputs (dict): Pre-processed `preds` from Teacher (already sliced to nc=1).
            student_neck_feats (list[Tensor]): Neck feature outputs of Student ([S3, S4, S5]).
            teacher_neck_feats (list[Tensor]): Neck feature outputs of Teacher ([T3, T4, T5]).
        """
        #  Feature Map Distillation Loss (MSE)
        loss_feat = 0.0
        for i, (s_feat, t_feat) in enumerate(zip(student_neck_feats, teacher_neck_feats)):
            proj_s_feat = self.proj_layers[i](s_feat)
            loss_feat += F.mse_loss(proj_s_feat, t_feat)
            
        # Khởi tạo các biến chứa giá trị Loss dạng scalar tensor
        loss_cls_one = torch.tensor(0.0, device=student_neck_feats[0].device)
        loss_cls_many = torch.tensor(0.0, device=student_neck_feats[0].device)
        loss_bbox_many = torch.tensor(0.0, device=student_neck_feats[0].device)
        loss_bbox_one = torch.tensor(0.0, device=student_neck_feats[0].device)
        
        gt_thresh = 0.25  # Ngưỡng lọc nền

        

        # Chạy vòng lặp tính toán song song cho cả 2 chiến lược gán nhãn
        for branch in ["one2many", "one2one"]:
            s_scores = student_outputs[branch]["scores"]  # (B, 1, Anchors)
            t_scores = teacher_outputs[branch]["scores"]  # (B, 1, Anchors) - Đã xử lý từ ngoài
            
            s_boxes = student_outputs[branch]["boxes"]    # (B, 4*reg_max, Anchors)
            t_boxes = teacher_outputs[branch]["boxes"]    # (B, 4*reg_max, Anchors)

            #  Tạo mask:
            with torch.no_grad():
                t_probs = torch.sigmoid(t_scores)
                # Giữ lại .max(dim=1) như một lớp bảo vệ (safeguard) nếu sau này bạn đổi số class
                max_prob, _ = t_probs.max(dim=1, keepdim=True)
                mask = (max_prob > gt_thresh).float()  # Shape chuẩn: (B, nc, anchors)
            
            # . Dùng QuantityFocalLoss ko mask input(pred : raw, target : sigmoided)
            t_soft_targets = torch.sigmoid(t_scores)
            
            loss_cls = self.quantityFocalLoss(
                s_scores ,
                t_soft_targets,
                hard_label = self.hard_label_for_o2o and (branch == 'one2one')
            )
            if branch == 'one2one':
                loss_cls_one = loss_cls
            else:
                loss_cls_many = loss_cls
            
            # . Bounding Box Distillation Loss 

            # THÊM .clamp(min=1.0) ĐỂ TRÁNH LỖI CHIA CHO 0
            num_masked = mask.sum().clamp(min=1.0)

            loss_bbox_elementwise = F.smooth_l1_loss(s_boxes, t_boxes, reduction='none')
            loss_bbox_scalar = (loss_bbox_elementwise * mask).sum() / num_masked
            
            #  về đúng biến đầu ra tương ứng
            if branch == "one2many":
                loss_bbox_many = loss_bbox_scalar
            elif branch == "one2one":
                loss_bbox_one = loss_bbox_scalar

        return {
            "loss_feat": loss_feat,
            "loss_cls_one" : loss_cls_one,
            "loss_cls_many" : loss_cls_many,
            "loss_bbox_many": loss_bbox_many,
            "loss_bbox_one": loss_bbox_one
        }
