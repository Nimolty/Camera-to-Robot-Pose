# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:20:27 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
# import _init_paths  
from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
import numpy as np
from ruamel.yaml import YAML
import torch.utils.data
from lib.opts import opts
from lib.model.model import create_model, load_model, save_model, create_dream_hourglass
from lib.model.data_parallel import DataParallel
from lib.logger import Logger
from utilities import find_ndds_seq_data_in_dir, set_random_seed, exists_or_mkdir
from datasets import CenterTrackSeqDataset, ManipulatorNDDSSeqDataset
# import dream_geo as dream
# from lib.dataset.dataset_factory import get_dataset # 这里的dataset用我们自己的
from lib.Dream_trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dream_ct_inference import inference, inference_real
import json

def get_optimizer(opt, model):
    if opt.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == 'sgd':
      print('Using SGD')
      optimizer = torch.optim.SGD(
        model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
    else:
      assert 0, opt.optim
    return optimizer

def save_results(train_log, kp_metrics, pnp_results, mode, writer, epoch):
    train_log[mode] = {}
    train_log[mode]["kp_metrics"] = {}
    train_log[mode]["kp_metrics"]["correctness"] = []
    train_log[mode]["pnp_results"] = {}
    train_log[mode]["pnp_results"]["correctness"] = []
    
    # save kp_metrics results
    if kp_metrics["num_gt_outframe"] > 0:
        out_of_frame_not_found_rate = "Percentage out-of-frame gt keypoints not found (correct): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_missing_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0,
                        kp_metrics["num_missing_gt_outframe"],
                        kp_metrics["num_gt_outframe"],
                    )
        out_of_frame_found_rate = "Percentage out-of-frame gt keypoints found (incorrect): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_found_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0,
                        kp_metrics["num_found_gt_outframe"],
                        kp_metrics["num_gt_outframe"],
                    )
        writer.add_scalar(f"{mode}/out_of_frame_not_found_rate (correct)", round(float(kp_metrics["num_missing_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0, 3), epoch)
        writer.add_scalar(f"{mode}/out_of_frame_found_rate (incorrect)", round(float(kp_metrics["num_found_gt_outframe"])
                        / float(kp_metrics["num_gt_outframe"])
                        * 100.0, 3), epoch)
        
    else:
        out_of_frame_not_found_rate = None
        out_of_frame_found_rate = None
    
    train_log[mode]["kp_metrics"]["correctness"] += [out_of_frame_not_found_rate, out_of_frame_found_rate]
    if kp_metrics["num_gt_inframe"] > 0:
        in_frame_not_found_rate = "Percentage in-frame gt keypoints not found (incorrect): {:.3f}% ({}/{})".format(
                        float(kp_metrics["num_missing_gt_inframe"])
                        / float(kp_metrics["num_gt_inframe"])
                        * 100.0,
                        kp_metrics["num_missing_gt_inframe"],
                        kp_metrics["num_gt_inframe"],
                    )
        in_frame_found_rate = "Percentage in-frame gt keypoints found (correct): {:.3f}% ({}/{})".format(
                    float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0,
                    kp_metrics["num_found_gt_inframe"],
                    kp_metrics["num_gt_inframe"],
                )
                
        writer.add_scalar(f"{mode}/in_frame_not_found_rate (incorrect)", round(float(kp_metrics["num_missing_gt_inframe"])
                        / float(kp_metrics["num_gt_inframe"])
                        * 100.0, 3), epoch)
        writer.add_scalar(f"{mode}/in_frame_found_rate (correct)", round(float(kp_metrics["num_found_gt_inframe"])
                    / float(kp_metrics["num_gt_inframe"])
                    * 100.0, 3), epoch)
        
        train_log[mode]["kp_metrics"]["correctness"] += [in_frame_not_found_rate, in_frame_found_rate]
        if kp_metrics["num_found_gt_inframe"] > 0:
            L2_info = "L2 error (px) for in-frame keypoints (n = {}):".format(
                        kp_metrics["num_found_gt_inframe"]
                    )
            kp_AUC = "   AUC: {:.5f}".format(kp_metrics["l2_error_auc"])
            kp_AUC_threshold = " AUC threshold: {:.5f}".format(
                        kp_metrics["l2_error_auc_thresh_px"]
                    )
            Mean_pck = "   Mean: {:.5f}".format(kp_metrics["l2_error_mean_px"])
            Median_pck = "   Median: {:.5f}".format(kp_metrics["l2_error_median_px"])
            std_pck = "   Std Dev: {:.5f}".format(kp_metrics["l2_error_std_px"])
            train_log[mode]["kp_metrics"]["results"] = {}
            train_log[mode]["kp_metrics"]["results"]["L2_info"] = L2_info
            train_log[mode]["kp_metrics"]["results"]["kp_AUC"] = kp_AUC
            train_log[mode]["kp_metrics"]["results"]["kp_AUC_threshhold"] = kp_AUC_threshold
            train_log[mode]["kp_metrics"]["results"]["Mean_pck"] = Mean_pck
            train_log[mode]["kp_metrics"]["results"]["Median_pck"] = Median_pck
            train_log[mode]["kp_metrics"]["results"]["std_pck"] = std_pck
            
            writer.add_scalar(f"{mode}/kp_AUC", round(kp_metrics["l2_error_auc"], 5), epoch)
            writer.add_scalar(f"{mode}/Mean_pck", round(kp_metrics["l2_error_mean_px"], 5), epoch)
            writer.add_scalar(f"{mode}/Median_pck", round(kp_metrics["l2_error_median_px"], 5), epoch)
            writer.add_scalar(f"{mode}/std_pck", round(kp_metrics["l2_error_std_px"], 5), epoch)
            
        else:
            train_log[mode]["kp_metrics"]["results"] = ["No in-frame gt keypoints were detected."]
    else:
        train_log[mode]["kp_metrics"]["correctness"].append("No in-frame gt keypoints.")
    
    n_pnp_possible = pnp_results["num_pnp_possible"]
    if n_pnp_possible > 0:
        n_pnp_successful = pnp_results["num_pnp_found"]
        n_pnp_fails = pnp_results["num_pnp_not_found"]
        fail = "Percentage of frames where PNP failed when viable (incorrect): {:.3f}% ({}/{})".format(
                        float(n_pnp_fails) / float(n_pnp_possible) * 100.0,
                        n_pnp_fails,
                        n_pnp_possible,
                    )
        success = "Percentage of frames where PNP was successful when viable (correct): {:.3f}% ({}/{})".format(
                        float(n_pnp_successful) / float(n_pnp_possible) * 100.0,
                        n_pnp_successful,
                        n_pnp_possible,
                    )
        writer.add_scalar(f"{mode}/pnp_success", round(float(n_pnp_successful) / float(n_pnp_possible) * 100.0, 3), epoch)
        train_log[mode]["pnp_results"]["correctness"] += [fail, success]
        ADD = "ADD (m) for frames where PNP was successful when viable (n = {}):".format(
                        n_pnp_successful
                    )
        ADD_AUC = "   AUC: {:.5f}".format(pnp_results["add_auc"])
        ADD_AUC_threshold = "      AUC threshold: {:.5f}".format(pnp_results["add_auc_thresh"])
        Mean_ADD = "   Mean: {:.5f}".format(pnp_results["add_mean"])
        Median_ADD = "   Median: {:.5f}".format(pnp_results["add_median"])
        Std_ADD = "   Std Dev: {:.5f}".format(pnp_results["add_std"])
        train_log[mode]["pnp_results"]["results"] = {}
        train_log[mode]["pnp_results"]["results"]["ADD"] = ADD
        train_log[mode]["pnp_results"]["results"]["ADD_AUC"] = ADD_AUC
        train_log[mode]["pnp_results"]["results"]["ADD_AUC_threshold"] = ADD_AUC_threshold
        train_log[mode]["pnp_results"]["results"]["Mean_ADD"] = Mean_ADD
        train_log[mode]["pnp_results"]["results"]["Median_ADD"] = Median_ADD
        train_log[mode]["pnp_results"]["results"]["Std_ADD"] = Std_ADD
        writer.add_scalar(f"{mode}/ADD_AUC", round(pnp_results["add_auc"], 5), epoch)
        writer.add_scalar(f"{mode}/Mean_ADD", round(pnp_results["add_mean"], 5), epoch)
        writer.add_scalar(f"{mode}/Median_ADD", round(pnp_results["add_median"], 5), epoch)
        writer.add_scalar(f"{mode}/Std_ADD", round(pnp_results["add_std"], 5), epoch)
            
    
    

def main(opt):
    set_random_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    
    # 这些地方写tensorboard
    tb_path = os.path.join(opt.save_dir, 'tb')
    ckpt_path = os.path.join(opt.save_dir, 'ckpt')
    results_path = os.path.join(opt.save_dir, 'results')
    exists_or_mkdir(results_path)
    exists_or_mkdir(tb_path)
    exists_or_mkdir(ckpt_path)
    writer = SummaryWriter(tb_path)

    input_data_path = opt.dataset # 这里是dataset的路径
    val_data_path = opt.val_dataset
    found_data = find_ndds_seq_data_in_dir(input_data_path)
    if opt.add_dataset:
        add_data = find_ndds_seq_data_in_dir(opt.add_dataset)
        print("length of original found_data", len(found_data))
        found_data += add_data
        print("length of current found_data", len(found_data))
    
    val_data = find_ndds_seq_data_in_dir(val_data_path)
    keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]
    
    network_input_resolution = (480, 480) # 时刻需要注意这里是width x height
    network_output_resolution = (120, 120) # 时刻需要注意这里是width x height
    input_width, input_height = network_input_resolution
    network_input_resolution_transpose = (input_height, input_width) # centertrack的输入是HxW
    opt = opts().update_dataset_info_and_set_heads_dream(opt, 7, network_input_resolution_transpose)
    image_normalization = {"mean" : (0.5, 0.5, 0.5), "stdev" : (0.5, 0.5, 0.5)}

    Dataset = CenterTrackSeqDataset(
    found_data, 
    "Franka_Emika_Panda", 
    keypoint_names, 
    opt, 
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    include_ground_truth=True,
    include_belief_maps=True,
    seq_frame = 3
    ) 
    ValDataset = CenterTrackSeqDataset(
    val_data, 
    "Franka_Emika_Panda", 
    keypoint_names, 
    opt, 
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    include_ground_truth=True,
    include_belief_maps=True,
    seq_frame = 3
    ) 
    
    
#    Dataset = ManipulatorNDDSSeqDataset(
#    found_data, 
#    "Franka_Emika_Panda", 
#    keypoint_names, 
#    opt, 
#    network_input_resolution,
#    network_output_resolution,
#    image_normalization,
#    "shrink-and-crop",
#    augment_data=True,
#    include_ground_truth=True,
#    include_belief_maps=True
#    )

    
    print('length dataset', len(Dataset))
    
#    n_data = len(ValDataset)
#    n_train_data = int(round(n_data) * 0.01)
#    n_valid_data = n_data - n_train_data
#    ValDataset, valid_dataset = torch.utils.data.random_split(
#        ValDataset, [n_train_data, n_valid_data]
#    )
#    
#    n_data = len(Dataset)
#    n_train_data = int(round(n_data)*0.001)
#    n_valid_data = n_data - n_train_data
#    Dataset, valid_dataset = torch.utils.data.random_split(
#        Dataset, [n_train_data, n_valid_data]
#    )
    
    print(opt)
#    if not opt.not_set_cuda_env:
#      print(opt.not_set_cuda_env)
#      print('gpus_str', opt.gpus_str)
#      os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('device', opt.device)
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
#    model = create_dream_hourglass()
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)
    
    trainer = Trainer(opt, model, optimizer)
    print('opt.gpus', opt.gpus)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    train_loader = torch.utils.data.DataLoader(
        Dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )
    val_loader = torch.utils.data.DataLoader(
        ValDataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True 
        )
    
    for epoch in tqdm(range(start_epoch + 1, opt.num_epochs + 1)):
        trainer.train(epoch, train_loader, opt.device, writer, phase=opt.phase)
        this_path = os.path.join(ckpt_path, "model_{}.pth".format(epoch))
        save_model(this_path, epoch, model, optimizer)
        
        
#        # validation
        mean_valid_loss_per_batch, mean_valid_hm_loss_per_batch, mean_valid_reg_loss_per_batch = trainer.valid_epoch(val_loader, opt.device, phase=opt.phase)
        training_log = {}
        training_log["validation"] = {}
        training_log["validation"]["mean_valid_loss_all"] = mean_valid_loss_per_batch
        training_log["validation"]["mean_valid_loss_hm"] = mean_valid_hm_loss_per_batch
        training_log["validation"]["mean_valid_loss_reg"] = mean_valid_reg_loss_per_batch
        
        writer.add_scalar(f"validation/mean_valid_loss_all", mean_valid_loss_per_batch, epoch)
        writer.add_scalar(f"validation/mean_valid_loss_hm", mean_valid_hm_loss_per_batch, epoch)
        writer.add_scalar(f"validation/mean_valid_loss_reg", mean_valid_reg_loss_per_batch, epoch)
#        
        # inference in synthetic test set
        print("!!!!!!!!!!")
        print("opt.phase", opt.phase)
        print('load_model', opt.load_model)
        opt.load_model = this_path
        print('load_model', opt.load_model)
        print('infer_dataset', opt.infer_dataset)
        opt.infer_dataset = "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/synthetic_test_1005/"
        print('infer_dataset', opt.infer_dataset)
        syn_test_info = inference(opt)
        kp_metrics, pnp_results = syn_test_info[0], syn_test_info[1]
        save_results(training_log, kp_metrics, pnp_results, mode="synthetic", writer=writer, epoch=epoch)
#        
        # inference in pure test set
        print('infer_dataset', opt.infer_dataset)
        opt.infer_dataset = "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/pure_test/"
        print('infer_dataset', opt.infer_dataset)
        pure_test_info = inference(opt)
        kp_metrics_pure, pnp_results_pure = pure_test_info[0], pure_test_info[1]
        save_results(training_log, kp_metrics_pure, pnp_results_pure, mode="pure", writer=writer, epoch=epoch)
#        
        # inference in real
        real_test_info = inference_real(opt)
        kp_metrics_real, pnp_results_real = real_test_info[0], real_test_info[1]
        save_results(training_log, kp_metrics_real, pnp_results_real, mode="real", writer=writer, epoch=epoch)
        
        # save in json
        meta_path = os.path.join(results_path, "info_{}.json".format(epoch))
        if os.path.exists(meta_path):
            os.remove(meta_path)
        file_write_meta = open(meta_path, 'w')
        meta_json = [training_log]
        json_save = json.dumps(meta_json, indent=1)
        file_write_meta.write(json_save)
        file_write_meta.close()
        
        
                

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)


















