# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:52:39 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tools._init_paths as _init_paths

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import cv2
import json
import copy
import numpy as np
from lib.opts import opts
from lib.Dream_detector import DreamDetector
import torch
from tqdm import tqdm
import dream_geo as dream
from ruamel.yaml import YAML
from PIL import Image as PILImage

def find_dataset(opt):
    keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]

    real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    input_dir = opt.infer_dataset
    input_dir = os.path.expanduser(input_dir) # 输入的是../../franka_data_0825
    assert os.path.exists(input_dir),\
    'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir) # 现在变成List 从00000到02000了，目前生成了2000个视频序列了
    
    found_videos = []
    for each_dir in dirlist:
        output_dir = os.path.join(input_dir, each_dir)
        # output_dir = ../../franka_data_0825/xxxxx
        found_video = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                       if f.endswith('.png')]
        found_json = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                        if f.endswith("meta.json")]
        found_video.sort()
        found_json.sort()
        if len(found_video) != 30 or len(found_json) != 30:
            continue
        found_videos.append([found_video, found_json])
    
    return found_videos

def inference(opt):
    keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]
    
    with torch.no_grad():
        found_videos = find_dataset(opt)
        json_list, detected_kps_list = [], []
        for found_video_0 in tqdm(found_videos[:50]):
        # found_video_0 = found_videos[j]
        # print('found_video_0', found_video_0) 
        # print('json_path', found_video_0[1])
            length = len(found_video_0[0])
            # print(length)
            detector = DreamDetector(opt, keypoint_names, is_real=False, is_ct=True)
            for i, img_path in enumerate(found_video_0[0]):
                if i == 0:
                    continue
                json_path = found_video_0[1][i]
                img = cv2.imread(img_path)
#                img = PILImage.open(img_path).convert("RGB")
#                img_shrink_and_crop = dream.image_proc.preprocess_image(
#                img, (opt.input_w, opt.input_h), "shrink-and-crop"
#                )
#                img_shrink_and_crop = np.asarray(img_shrink_and_crop)
                print('img.size', img.shape)
                
#                ret, detected_kps_netin = detector.run(img, i, json_path, is_final=True)
#                detected_kps_np = dream.image_proc.convert_keypoints_to_raw_from_netin(
#                    detected_kps_netin,
#                    (opt.input_w, opt.input_h),
#                    (640, 360),
#                    "shrink-and-crop", 
#                )
                ret, detected_kps_np = detector.run(img, i, json_path, is_final=True)            
                output_dir = img_path.rstrip('png')
                np.savetxt(output_dir + 'txt', detected_kps_np)
                json_list.append(json_path)
                detected_kps_list.append(detected_kps_np) 
                    # print(detected_kps)
    
    exp_dir = opt.exp_dir
    pth_order = opt.load_model.split('/')[-1]
    exp_id = opt.load_model.split('/')[-3]
    pth_order = pth_order.rstrip('.pth')
    output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
    dream.utilities.exists_or_mkdir(output_dir)
    
    analysis_info = dream.analysis.analyze_ndds_center_dream_dataset(
    json_list, # 在外面直接写一个dataset就好了，需要注意它的debug_node为LIGHT
    detected_kps_list,
    opt, 
    keypoint_names,
    [640, 360],
    output_dir,
    is_real=False)
    return analysis_info



def inference_real(opt):
    real_info_path = "/root/autodl-tmp/dream_data/data/real/dream_real_info/"
    real_info_path = os.path.join(real_info_path, opt.is_real + "_split_info.json")
    real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    parser = YAML(typ="safe")
    with open(real_info_path, "r") as f:
        real_data = parser.load(f.read().replace('\t', ' '))
        real_jsons = real_data["json_paths"]
        real_images = real_data["img_paths"] # 为一个list里面是10个video
    json_list, detected_kps_list = [], []
    
    with torch.no_grad():
        for idx, (video_images, video_jsons) in tqdm(enumerate((zip(real_images, real_jsons)))):
            if idx >= 0 : 
                detector = DreamDetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=True)
                assert len(video_images) == len(video_jsons)
                length = len(video_images)
                
                for j, (img_path, json_path) in enumerate(zip(video_images, video_jsons)):
                    if j == 0:
                        continue 
                    
                    img = cv2.imread(img_path)
                    # print('img.shape', img.shape)
#                    img = PILImage.open(img_path).convert("RGB")
#                    print("#################### SIZE ####################")
#                    print('size', img.size)
#                    print("#################### SIZE ####################")
#                    img_shrink_and_crop = dream.image_proc.preprocess_image(
#                    img, (opt.input_w, opt.input_h), "shrink-and-crop"
#                    )
#                    img_shrink_and_crop = np.asarray(img_shrink_and_crop)

                    ret, detected_kps_np = detector.run(img, j, json_path, is_final=True)
#                    detected_kps_np = dream.image_proc.convert_keypoints_to_raw_from_netin(
#                        detected_kps_netin,
#                        (opt.input_w, opt.input_h),
#                        img.size,
#                        "shrink-and-crop", 
#                    )
                    # output_dir = img_path.rstrip('png')
                    # np.savetxt(output_dir + 'txt', detected_kps_np)
                    json_list.append(json_path)
                    detected_kps_list.append(detected_kps_np) 
        
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        dream.utilities.exists_or_mkdir(output_dir)
        
        analysis_info = dream.analysis.analyze_ndds_center_dream_dataset(
        json_list, # 在外面直接写一个dataset就好了，需要注意它的debug_node为LIGHT
        detected_kps_list,
        opt, 
        real_keypoint_names,
        [640, 480],
        output_dir,
        is_real=opt.is_real)
        return analysis_info
                
                
    
     

if __name__ == "__main__":
    opt = opts().init_infer(7, (480, 480))
    # inference(opt)
    inference_real(opt)
    
























