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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import sys
import cv2
import json
import copy
import glob
import numpy as np
from lib.opts import opts
from lib.Dream_detector import DreamDetector
import torch
from tqdm import tqdm
import dream_geo as dream
from ruamel.yaml import YAML
from PIL import Image as PILImage
import time
import random



def find_dataset(opt):
    keypoint_names = opts().get_keypoint_names(opt)
    input_dir = opt.infer_dataset
    input_dir = os.path.expanduser(input_dir) # 输入的是../../franka_data_0825
    assert os.path.exists(input_dir),\
    'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir) # 现在变成List 从00000到02000了，目前生成了2000个视频序列了
    print(dirlist)
    dirlist.sort()
    
    found_videos = []
    for each_dir in dirlist:
        if each_dir.endswith(".json"):
            continue
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
    keypoint_names = opts().get_keypoint_names(opt)
    print("inference dataset", opt.infer_dataset)
    with torch.no_grad():
        found_videos = find_dataset(opt)
        json_list, detected_kps_list = [], []
        print("length of found_videos", len(found_videos))
        
        # sample_found_videos = random.sample(found_videos, 150)
        # print("length of sample found videos", len(sample_found_videos))

        # for video_idx, found_video_0 in tqdm(enumerate(sample_found_videos)): # 全部Inference！
        cou = 0
        for video_idx, found_video_0 in tqdm(enumerate(found_videos)):
            # print(found_video_0) 
            length = len(found_video_0[0])
            if video_idx >= 10000:
                continue
            
            index_list = ['00120',
                          '00152',
                          '00166',
                          '00192',
                          '00225',
                          '00258',
                          '00285',
                          '00342',
                          '00348']
            if (index_list[cou] not in found_video_0[0][0]):
                    continue


            detector = DreamDetector(opt, keypoint_names, is_real=False, is_ct=True, idx=int(index_list[cou]))
            for i, img_path in enumerate(found_video_0[0]):
                if i == 0:
                    continue
                json_path = found_video_0[1][i]
#                print('img_path', img_path)
#                print('json_path', json_path)
                img = cv2.imread(img_path)
                t1 = time.time()
                ret, detected_kps_np, _ = detector.run(img, i, json_path, is_final=True, teaser_flag=True)  
                t2 = time.time()
#                print("消耗一次时间", t2 - t1)          
                output_dir = img_path.rstrip('png')
                # np.savetxt(output_dir + 'txt', detected_kps_np)
                json_list.append(json_path)
                detected_kps_list.append(detected_kps_np) 
                    # print(detected_kps)
            cou += 1
            if cou == 9:
                break
    
    exp_dir = opt.exp_dir
    pth_order = opt.load_model.split('/')[-1]
    exp_id = opt.load_model.split('/')[-3]
    pth_order = pth_order.rstrip('.pth')
    output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
    dream.utilities.exists_or_mkdir(output_dir)
    
    save_name = opt.infer_dataset.split('/')[-2]
    path_meta = os.path.join(output_dir, f"{save_name}_dt_and_json.json")
    if not os.path.exists(path_meta):
        file_write_meta = open(path_meta, 'w')
        meta_json = dict()
        meta_json["dt"] = np.array(detected_kps_list).tolist()
        meta_json["json"] = json_list

        json_save = json.dumps(meta_json, indent=1)
        file_write_meta.write(json_save)
        file_write_meta.close()

    parser = YAML(typ="safe")
    with open(path_meta, "r") as f:
        real_data = parser.load(f.read().replace('\t', ' '))
        detected_kps_list = real_data["dt"]
        json_list = real_data["json"] 
    
    
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
    json_list, detected_kps_list, d_lst = [], [], []
    json_lists, detected_kps_lists = [], []
    count = 0
    with torch.no_grad():
        for idx, (video_images, video_jsons) in tqdm(enumerate((zip(real_images, real_jsons)))):
            this_json_list, this_detected_kp_proj_list = [], []
            if idx >= 100:  
                count = idx
                detector = DreamDetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=True, idx=idx)
                assert len(video_images) == len(video_jsons)
                length = len(video_images)
                
                for j, (img_path, json_path) in enumerate(zip(video_images, video_jsons)):
                    if j == 0:
                        continue 
                    
                    img = cv2.imread(img_path)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 双边高斯平滑
                    # save_dir = "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/ct_infer_img/gaussian_k3"
                    # dream.utilities.exists_or_mkdir(save_dir)
                    # img = cv2.bilateralFilter(img,5,10,10)
                    # img = cv2.GaussianBlur(img,(3, 3),10)
                    # cv2.imwrite(os.path.join(save_dir, f"{idx}_{str(j).zfill(5)}_gaussian_k3.png"), img)


#                    img_resized = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
#                    print("img_resized.shape",img_resized.shape)
#                    print("img1.shape", img1.shape)
#                    img_input = np.zeros((480, 480, 3))
#                    img_input[120:-120, 80:-80, :] = img_resized
#                    img = img_input
                
                    # print('img.shape', img.shape)
#                    img = PILImage.open(img_path).convert("RGB")
#                    print("#################### SIZE ####################")
#                    print('size', img.size)
#                    print("#################### SIZE ####################")
#                    img_shrink_and_crop = dream.image_proc.preprocess_image(
#                    img, (opt.input_w, opt.input_h), "shrink-and-crop"
#                    )
#                    img_shrink_and_crop = np.asarray(img_shrink_and_crop)
                    save_dir = f"/root/autodl-tmp/camera_to_robot_pose/topk_check/{idx}/"
                    dream.utilities.exists_or_mkdir(save_dir)
#                    ret, detected_kps_np = detector.run(img, j, json_path, is_final=True, save_dir=save_dir)
#                    detected_kps_np = dream.image_proc.convert_keypoints_to_raw_from_netin(
#                        detected_kps_netin,
#                        (opt.input_w, opt.input_h),
#                        img.size,
#                        "shrink-and-crop", 
#                    )

                    # t1 = time.time()
                    ret, detected_kps_np = detector.run(img, j, json_path, is_final=True, save_dir=save_dir)
                    
#                    detected_kps_np = (detected_kps_np - np.array([159, 119])) * 2
#                    img_PIL = PILImage.open(img_path)
                    
#                    print("total", ret["tot"])
#                    print("load time", ret["load"])
#                    print("pre_time", ret["pre"])
#                    print("net time", ret["net"])
#                    print("dec time", ret["dec"])
#                    print("post time", ret["post"])
#                    print("merge time", ret["merge"])
#                    print("display time", ret["display"])
#                    print("tracking time", ret["track"])
                    # t2 = time.time()
                    # print("t2 - t1", t2 - t1)
                    output_dir = img_path.rstrip('png')
                    np.savetxt(output_dir + 'txt', detected_kps_np)
                    json_list.append(json_path)
                    detected_kps_list.append(detected_kps_np) 
                    this_json_list.append(json_path)
                    this_detected_kp_proj_list.append(detected_kps_np.tolist())

            json_lists.append(this_json_list)
            detected_kps_lists.append(this_detected_kp_proj_list)
                    # print("shape", detected_kps_np.shape)
                    # d_lst.append(detected_kps_np.tolist())
        
        exp_dir = opt.exp_dir
        pth_order = opt.load_model.split('/')[-1]
        exp_id = opt.load_model.split('/')[-3]
        pth_order = pth_order.rstrip('.pth')
        output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        dream.utilities.exists_or_mkdir(output_dir)
        

        # path_meta = os.path.join(output_dir, f"dt_and_json_{count}.json")
        if opt.is_real == "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json.json")
        elif opt.is_real == "panda-3cam_realsense" and opt.multi_frame != 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_multi.json")
        elif opt.is_real != "panda-3cam_realsense" and opt.multi_frame == 0:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}.json")
        else:
            path_meta = os.path.join(output_dir, f"dt_and_json_{opt.is_real}_multi.json")
        if not os.path.exists(path_meta):
            file_write_meta = open(path_meta, 'w')
            meta_json = dict()
            meta_json["dt"] = np.array(detected_kps_list).tolist()
            meta_json["json"] = json_list
            if opt.multi_frame > 0:
                meta_json["dt_multi"] = detected_kps_lists
                meta_json["json_multi"] = json_lists

            json_save = json.dumps(meta_json, indent=1)
            file_write_meta.write(json_save)
            file_write_meta.close()

        parser = YAML(typ="safe")
        with open(path_meta, "r") as f:
            real_data = parser.load(f.read().replace('\t', ' '))
            detected_kps_list = real_data["dt"]
            json_list = real_data["json"] 
            if opt.multi_frame > 0:
                detected_kp_proj_lists = real_data["dt_multi"]
                json_lists = real_data["json_multi"]


        if opt.multi_frame == 0:
            analysis_info = dream.analysis.analyze_ndds_center_dream_dataset(
            json_list, # 在外面直接写一个dataset就好了，需要注意它的debug_node为LIGHT
            detected_kps_list,
            opt, 
            real_keypoint_names,
            [640, 480],
            output_dir,
            is_real=opt.is_real)
            return analysis_info
        else:
            dream.analysis.solve_multiframe_pnp(
            json_lists,
            detected_kp_proj_lists,
            opt,
            real_keypoint_names,
            [640,480],
            output_dir,
            multiframe=opt.multi_frame,
            is_real=opt.is_real,
            )


def inference_real_multiframe(opt):
    random.seed(opt.seed)
    print("opt.seed", opt.seed)
    real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    root_image_path = f"/root/autodl-tmp/camera_to_robot_pose/Dream_ty/{opt.note}/color/"
    root_json_path = f"/root/autodl-tmp/camera_to_robot_pose/Dream_ty/{opt.note}/meta_ty/"
    real_jsons = glob.glob(os.path.join(root_json_path, '*.json'))
    real_images = glob.glob(os.path.join(root_image_path, '*.png'))
    real_jsons.sort()
    real_images.sort()
    
    assert len(real_jsons) == len(real_images)
    
    meta_loc = {
    "real_1107" : [
    "000106", "000233", "000360", "000486",
    "000612", "000741", "000868", "000995",
    "001124", "001254", "001380", "001509",
    "001633", "001763", "001897", "002027",
    "002143", "002255", "002373", "002498",
    "002623", "002743", "002862", "002981",
    "003102", "003226", "003351", "003472",
    "003595", "003713", "003840"
    ],    
    "real_1107_with_background" : [
    "000108", "000229", "000353", "000483",
    "000611", "000736", "000861", "000984",
    "001106", "001227", "001352", "001479",
    "001606", "001729", "001857", "001984",
    "002108", "002232", "002351", "002474",
    "002596", "002722", "002850", "002975",
    "003100", "003225", "003351", "003474",
    "003598", "003716", "003836"
    ],
    "real_1108" : \
    ['000232', '000481', '001214', '001335', '001455', '001695', '002059', '002421', '002540', '002656', '002894', '003134', '003733', '003854', '003971', '004088', '004323', '004566', '004693', '004812'],
    "real_1107_with_background" : ["000111"]
    }
    compare_lst = meta_loc[opt.note]

    # json_list, detected_kps_list, d_lst = [], [], []
    json_list, dt_kps_projs_list = [], []
    count = 0
    with torch.no_grad():
        # detector = DreamDetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=True, idx=0)
        detector = DreamDetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=opt.is_ct, idx=0)
        for j, (img_path, json_path) in tqdm(enumerate(zip(real_images, real_jsons))):
            img_idx = img_path.split('/')[-1].replace(".png", "")
            json_idx = json_path.split('/')[-1].replace(".json", "")
            assert img_idx == json_idx
            # if j == 0 or j >= 30:
            if j == 0:
                continue

            img = cv2.imread(img_path) 
            if not opt.is_ct:
                img = PILImage.open(img_path).convert("RGB")
                # print("#################### SIZE ####################")
                # print('size', img.size)
                # print("#################### SIZE ####################")
                img_shrink_and_crop = dream.image_proc.preprocess_image(
                img, (opt.input_w, opt.input_h), "shrink-and-crop"
                )
                img = np.asarray(img_shrink_and_crop)
                # print("shrink and crop size", img)
            
                
            # print('img.shape', img.shape)  
 
            save_dir = f"/root/autodl-tmp/camera_to_robot_pose/topk_check/0/"
            dream.utilities.exists_or_mkdir(save_dir)
            # ret, detected_kps_np, camera_K = detector.run(img, j, json_path, is_final=True, save_dir=save_dir)
            # print("detected_kps_np", detected_kps_np)
            
            if compare_lst[count] in json_idx:
                json_list.append(json_path)
                dt_kps_projs_list.append(detected_kps_np.tolist()) 
                ret, detected_kps_np, camera_K = detector.run(img, j, json_path, is_final=True, save_dir=save_dir,teaser_flag=True,img_path=img_path)
                # count += 1
                # if count == 20:
                #     break
            else:
                ret, detected_kps_np, camera_K = detector.run(img, j, json_path, is_final=True, save_dir=save_dir,teaser_flag=True,img_path=img_path)
        
        
        # gt_kps_pos_list = [dream.utilities.load_x3d(json_path, real_keypoint_names) for json_path in json_list]
        
        # exp_dir = opt.exp_dir
        # pth_order = opt.load_model.split('/')[-1]
        # exp_id = opt.load_model.split('/')[-3]
        # pth_order = pth_order.rstrip('.pth')
        # output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
        # dream.utilities.exists_or_mkdir(output_dir)
        

        # # path_meta = os.path.join(output_dir, f"dt_and_json_{count}.json")
        # if opt.is_real == "panda-3cam_realsense":
        #     path_meta = os.path.join(output_dir, f"ros_{opt.note}_gt.json")

        # if not os.path.exists(path_meta):
        #     file_write_meta = open(path_meta, 'w')
        #     meta_json = dict()
        #     meta_json["dt"] = np.array(dt_kps_projs_list).tolist()
        #     meta_json["gt"] = gt_kps_pos_list
        #     meta_json["camera_K"] = camera_K.tolist()
            
        #     json_save = json.dumps(meta_json, indent=1)
        #     file_write_meta.write(json_save)
        #     file_write_meta.close()

        # parser = YAML(typ="safe")
        # with open(path_meta, "r") as f:
        #     real_data = parser.load(f.read().replace('\t', ' '))
        #     dt_kps_projs_list = real_data["dt"]
        #     gt_kps_pos_list = real_data["gt"] 
        #     camera_K = np.array(real_data["camera_K"])


        # print("dt_kps_projs_list", dt_kps_projs_list)
        # pnp_results = dream.analysis.solve_multiframe_pnp_real(
        #      gt_kps_pos_list,
        #      dt_kps_projs_list,
        #      opt,
        #      real_keypoint_names,
        #      [640,360],
        #      output_dir,
        #      camera_K,
        #      multi_frame=opt.multi_frame,
        #      is_real=opt.is_real,
        #      )
        
        # return pnp_results



        # return analysis_info
        # return True               
                
    
     

if __name__ == "__main__":
    opt = opts().init_infer(7, (480, 480))
    # opt = opts().init_infer(7, (400, 400))
    # inference(opt)
    # inference_real(opt)
    inference_real_multiframe(opt)
    # path_meta = os.path.join("/root/autodl-tmp/yangtian/summer_ty/DREAM-master/results/", f"ours_info_{opt.note}_gt.json")
    # file_write_meta = open(path_meta, 'w')
    # meta_json = []
    # meta_dic = {}
    # for idx in range(1, 2):
    #     opt.multi_frame = idx
    #     print("multi_frame", opt.multi_frame)
    #     pnp_results = inference_real_multiframe(opt)
    #     meta_dic[str(idx)] = pnp_results
    # meta_json.append(meta_dic)
    # json_save = json.dumps(meta_json, indent=1)
    # file_write_meta.write(json_save)
    # file_write_meta.close()
    
























