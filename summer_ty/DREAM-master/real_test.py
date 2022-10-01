# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:33:19 2022

@author: lenovo
"""

import os
from ruamel.yaml import YAML
import numpy as np
import json

def get_json_and_png(input_dir):
    # input_dir = "/mnt/data/Dream_data_all/data/real/panda-3cam_realsense/"
    dir_list = os.listdir(input_dir)
    found_jpgs = [os.path.join(input_dir, f) for f in dir_list if f.endswith(".jpg")]
    # print(found_jpgs)
    found_jsons = [os.path.join(input_dir, f) for f in dir_list if f.endswith(".json") and "camera" not in f]
    found_jpgs.sort()
    found_jsons.sort()
    
    print('length of found_jpgs', len(found_jpgs))
    print("length of found_jsons", len(found_jsons))
    assert len(found_jpgs) == len(found_jsons)
    object_name = "panda"
    keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    paths = [[found_jsons[0]]]
    imgs = [[found_jpgs[0]]]
    
    
    for i in range(len(found_jsons)-1):
        prev_json_path, next_json_path = found_jsons[i], found_jsons[i+1]
        parser = YAML(typ="safe")
        with open(prev_json_path, "r") as f:
            prev_data = parser.load(f.read().replace('\t', ' '))
        
        with open(next_json_path, "r") as f:
            next_data = parser.load(f.read().replace('\t', ' '))
            
        prev_keypoints = prev_data["objects"][0]["keypoints"]
        next_keypoints = next_data["objects"][0]["keypoints"]
        
        l2_err = []
        for j, kp_name in enumerate(keypoint_names):
            assert prev_keypoints[j]["name"] == kp_name
            assert next_keypoints[j]["name"] == kp_name
            
            prev_wrt_cam = prev_keypoints[j]["location"]
            next_wrt_cam = next_keypoints[j]["location"]
            
            l2_err.append(np.linalg.norm(np.array(prev_wrt_cam, dtype=np.float64) - np.array(next_wrt_cam, dtype=np.float64)))
        
        if max(l2_err) < 0.02:
            paths[-1].append(next_json_path)
            imgs[-1].append(found_jpgs[i+1])
        else:
            paths.append([next_json_path])
            imgs.append([found_jpgs[i+1]])
    
    return paths, imgs

input_dir = "/root/autodl-tmp/dream_data/data/real/panda-3cam_realsense/"
paths, imgs = get_json_and_png(input_dir)
path_meta = "/root/autodl-tmp/dream_data/data/real/realsense_split_info.json"
file_write_meta = open(path_meta, 'w')
meta_json = {}
meta_json['json_paths'] = paths
meta_json["img_paths"] = imgs
json_save = json.dumps(meta_json, indent=1)
file_write_meta.write(json_save)
file_write_meta.close()
    




















