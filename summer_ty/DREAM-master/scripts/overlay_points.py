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

image_path = "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/testdata/00001/0024_color.png"
kp_projs = [
   [
   294.4300469519465,
     244.85119671324833
   ],
   [
  289.04558753534405,
     164.6733953981293
   ],
   [
   326.5190375531059,
     95.66846753849717
   ],
   [
    344.4961131795966,
     105.69324755108597
   ],
   [
   438.597441640145,
     78.08369046577525
   ],
   [
   425.4327203319768,
     65.05605799351973
   ],
   [
  445.893265393855,
     48.46382228839689
   ]
  ]
images = []
for n in range(len(kp_projs)):
    image = dream.image_proc.overlay_points_on_image(image_path, [kp_projs[n]], annotation_color_dot = 'yellow',point_diameter=4)
    images.append(image)
    

img = dream.image_proc.mosaic_images(
            images, rows=2, cols=4, inner_padding_px=10
        )
img.save("/root/autodl-tmp/yangtian/summer_ty/DREAM-master/scripts/check_raw.png")






