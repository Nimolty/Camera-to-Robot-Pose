import csv
import math
import os
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from pyrr import Quaternion
import dream
from ruamel.yaml import YAML

image_path ="/root/autodl-tmp/yangtian/summer_ty/DREAM-master/scripts/overlay_imgs/0001_color.png"
json_path = "/root/autodl-tmp/yangtian/summer_ty/DREAM-master/scripts/overlay_imgs/0001_meta.json"
kp_projs = []
data_parser = YAML(typ="safe")
with open(json_path, "r") as f:
    data = data_parser.load(f.read().replace('\t',''))[0]
    kps = data["keypoints"]
    for kp in kps:
        kp_projs.append(kp["projected_location"])

images = []
for n in range(len(kp_projs)):
    image = dream.image_proc.overlay_points_on_image(image_path, [kp_projs[n]], annotation_color_dot = 'yellow',point_diameter=4)
    images.append(image)
    

img = dream.image_proc.mosaic_images(
            images, rows=3, cols=4, inner_padding_px=10
        )
img.save("/root/autodl-tmp/yangtian/summer_ty/DREAM-master/scripts/check_kuka.png")






