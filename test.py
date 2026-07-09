from ultralytics import YOLO
from Modules.Model import yolo26n_custom
import torch
import itertools

my_model = yolo26n_custom(nc=80)
teacher = YOLO("yolo26n.pt").model
yolo_backbone = teacher.model[:22]

with torch.no_grad():
    # Sử dụng itertools.chain để nối các tham số của backbone và neck thành 1 chuỗi liên tục
    my_parameters = itertools.chain(my_model.backbone.parameters(), my_model.neck.parameters())
    yolo_parameters = yolo_backbone.parameters()
    i = 0
    for my_para, yolo_para in zip(my_parameters, yolo_parameters):
        i+=1
        if my_para.shape == yolo_para.shape:
            my_para.copy_(yolo_para)
        else:
            print(f"Không trùng khớp kích thước: {my_para.shape} và {yolo_para.shape}  - idx {i}")

torch.save(my_model.state_dict(), "yolo26n_custom_backboneNeckLoaded.pt")


