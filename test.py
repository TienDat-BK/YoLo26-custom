from ultralytics import YOLO
from Modules.Model import yolo26n_custom
import torch

my_model = yolo26n_custom(1)
teacher = YOLO("yolo26n.pt").model
yolo_backbone = teacher.model[:9]

with torch.no_grad():
    for my_para, yolo_para in zip(my_model.backbone.parameters(), yolo_backbone.parameters()):
        if my_para.shape == yolo_para.shape:
            my_para.copy_(yolo_para)
        else:
            print(f"not same {my_para.shape} and {yolo_para.shape}")

torch.save(my_model.state_dict(), "yolo26n_custom_backboneLoaded.pt")


