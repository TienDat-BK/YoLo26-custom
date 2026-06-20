from Modules.Backbone import Backbone
from Modules.Model import yolo26n_custom

total_para = sum(p.numel() for p in yolo26n_custom().parameters())
print(total_para)