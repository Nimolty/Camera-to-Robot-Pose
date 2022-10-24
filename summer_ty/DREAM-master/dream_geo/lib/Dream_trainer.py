# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:42:02 2022

@author: lenovo
"""
# 设计我们自己的trainer, 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from .model.data_parallel import DataParallel
from .utils.utils import AverageMeter

from .model.losses import FastFocalLoss, RegWeightedL1Loss
from .model.losses import BinRotLoss, WeightedBCELoss
from .model.decode import generic_decode
from .model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from .utils.debugger import Debugger
from .utils.post_process import generic_post_process
import dream_geo as dream
from tqdm import tqdm

# 要定义loss_function

class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__() 
        self.loss = torch.nn.SmoothL1Loss()
    
    def forward(self, output, kp_projs_dis, cord):
        # output为bs x 2 x H X W
        # kp_projs为bs x 7 x 2 (先x, 后y)
        # cord为 bs x 7 x 2 (先x, 后y)
        # print('output.size', output.shape)
        output = output.permute(0, 2, 3, 1).contiguous()
        bs, n_kp, _ = kp_projs_dis.shape
        loss_data = torch.zeros(bs,n_kp, 2)
        for batch_idx in range(bs):
            for kp in range(n_kp):
                cor_x, cor_y = cord[batch_idx][kp].type(torch.long)
                out_x, out_y = output[batch_idx][cor_y][cor_x]
                loss_data[batch_idx][kp][0] = out_x
                loss_data[batch_idx][kp][1] = out_y
        
        loss = self.loss(loss_data, kp_projs_dis)
        return loss
class FocolLoss(torch.nn.Module):
    def __init__(self):
        super(FocolLoss, self).__init__()
    
    def forward(self, output, target, loc):
        # output为bs x 7 x H x W
        # belief_maps为bs x 7 x H x W
        # loc为bs x 7 x 2, 记录着每个class对应的gt点,未取整
        # 所以我们要生成一个mask: bs x 7 x H x W, 其中只在每个H x W下为1,或者不在画面内的点，也需要考虑
        # 现在有个问题，就是很有可能
        # trivial版本的和上述是一样的，但是也可以不trivial
        bs,n_kp, H, W = output.shape
        mask = torch.zeros_like(output)
        for batch_idx in range(bs):
            for kp in range(n_kp):
                cor_x, cor_y = loc[batch_idx][kp]
                if (cor_x < 0 ) or (cor_x > W - 1) or (cor_y < 0 ) or (cor_y > H - 1):
                    pass
                else:
                    cor_x = cor_x.int()
                    cor_y = cor_y.int()
                    mask[batch_idx][kp][cor_y][cor_x] = 1
        
        loss = 0
        alpha, beta = 2, 4
        loss1 = ((1 - output) ** alpha * torch.log(output) * mask).sum()
        loss2 = ((1 - target) ** beta * (output) ** alpha * torch.log(1 - output) * (1 - mask)).sum()
        if mask.sum():
            return -(loss1 + loss2) / mask.sum()
        return -loss2
 
class Loss(torch.nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.crit = torch.nn.MSELoss()
        # self.crit = FocolLoss()
        self.crit_reg = RegL1Loss()
        # self.crit_reg = torch.nn.SmoothL1Loss() 
        self.opt = opt
    
    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            otput['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1
        return output
    
    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}
        weights = {head : 1 for head in opt.heads}  
        weights["hm"] = 1
        weights['tracking'] = 0.0
        weights['reg'] = 0.01 
        
        
        for s in range(opt.num_stacks):
            output = outputs[s]
#            for key in output:
#                print(key, output[key].shape)
#            for key in batch:
#                if key != "config":
#                    print(key, batch[key].shape)
            
            
            output = self._sigmoid_output(output)
            
            if 'hm' in output:
#                losses['hm'] += self.crit(output['hm'], batch["next_belief_maps"].to(opt.device), batch["next_keypoint_projections_output"].to(opt.device)) / opt.num_stacks
                losses['hm'] += self.crit(output['hm'], batch["next_belief_maps"].to(opt.device)) / opt.num_stacks
            
            regression_heads = [
            'reg', 'tracking'] 
            # reg是点对点的offset, 
            # wh和ltrb_amodal是size之间的回归，
            # track表示两帧之间的差距，
            # 所以我们这里不需要wh和ltrb_amodal
            for head in regression_heads:
                losses[head] += self.crit_reg(
                    output[head], batch[head], batch["next_keypoint_projections_output_int"]
                    ) / opt.num_stacks
#                 losses[head] += self.crit_reg(
#                     output[head], batch[head].to(opt.device)
#                 ) / opt.num_stacks
        
        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += losses[head] * weights[head]
            
        return losses['tot'], losses

class Trainer(object):
    def __init__(
        self, opt, model, optimizer=None
            ):
        self.opt = opt
        self.optimizer = optimizer
        self.model = model
        self.loss = Loss(self.opt)
        self.base_lr = opt.lr
        self.total_epoch_nums = opt.num_epochs
        self.max_iters = opt.max_iters
    
    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
#            self.model = DataParallel(
#                self.model, device_ids=gpus, chunk_sizes=chunk_sizes
#                )
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            print('chunck_sizes', chunk_sizes)
            print('device', gpus)
            # print('self.model.cuda', self.model.device)
        else:
            self.model = self.model.cuda()
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)
    
    def valid_epoch(self, data_loader, device, phase):
        model = self.model
        if len(self.opt.gpus) > 1:
            model = self.model.module
        opt = self.opt
        with torch.no_grad():
            model.eval()
            valid_batch_losses = []
            valid_batch_hm_losses = []
            valid_batch_reg_losses = []
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                torch.cuda.empty_cache()
                pre_img = batch['prev_image_rgb_input'].to(device) # bs x 3 x H x W
                pre_hm = batch['prev_belief_maps'].to(device) # bs x H x W
                pre_hm = pre_hm.unsqueeze(1)
                repro_hm = batch["repro_belief_maps"].to(device)
                repro_hm = repro_hm.unsqueeze(1)
                next_img = batch['next_image_rgb_input'].to(device)
                if phase == "PlanA":
                    outputs = model(next_img, pre_img, pre_hm, repro_hm)
                elif phase == "CenterTrack+Repro":
                    outputs = model(next_img, pre_img, repro_hm)
                elif phase == "CenterTrack":
                    pre_origin_hm = batch["prev_origin_belief_maps"].to(device)
                    pre_origin_hm = pre_origin_hm.unsqueeze(1)
                    outputs = model(next_img, pre_img, pre_origin_hm)
                elif phase == "CenterTrack-Pre_hm":
                    outputs = model(next_img, pre_img)
                elif phase == "CenterNet":
                    outputs = model(next_img)
                elif phase == "PlanA_win":
                    pre_hm_cls = batch["prev_belief_maps_cls"].to(device)
                    repro_hm_cls = batch["repro_belief_maps_cls"].to(device)
                    # print("pre_hm_cls.shape", pre_hm_cls.shape)
                    outputs = model(next_img, pre_img, pre_hm, repro_hm, pre_hm_cls, repro_hm_cls)
                else:
                    raise ValueError
                # outputs = model(next_img, pre_img, pre_hm, repro_hm)
                loss, loss_stats = self.loss(outputs, batch)
                
                loss_all_this_batch = loss_stats["tot"].item()
                loss_hm_this_batch = loss_stats["hm"].item()
                loss_reg_this_batch = loss_stats["reg"].item() 
                valid_batch_losses.append(loss_all_this_batch)
                valid_batch_hm_losses.append(loss_hm_this_batch)
                valid_batch_reg_losses.append(loss_reg_this_batch)
            
            mean_valid_loss_per_batch = np.mean(valid_batch_losses)
            mean_valid_hm_loss_per_batch = np.mean(valid_batch_hm_losses)
            mean_valid_reg_loss_per_batch = np.mean(valid_batch_reg_losses)
        
        return float(mean_valid_loss_per_batch), float(mean_valid_hm_loss_per_batch), float(mean_valid_reg_loss_per_batch)
    
    def adapt_lr(self, epoch_num, batch_idx):
        cur_iters = (epoch_num - 1) * self.iter_per_epoch + batch_idx
        self.epoch_num = epoch_num
        warmup_iters = 3000
        warmup_ratio = 1e-06
        # print("self.max_iters", self.max_iters)
        # print('base_lr', self.base_lr)
        # print('all_epochs', self.total_epoch_nums)
        if epoch_num == 1 and cur_iters <= warmup_iters:
            k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
            lr_ = self.base_lr * (1 - k)
        else:
            lr_ = self.base_lr * (1.0 - (cur_iters - 1) / self.max_iters) ** 1.0
        
        for param_group in self.optimizer.param_groups:
            # print("past learning rate", param_group["lr"])
            param_group['lr'] = lr_
            # print("current learning rate", param_group["lr"])
          
    def run_epoch(self, phase, epoch, data_loader, device, writer):
        model = self.model
        model.train()
#        if len(self.opt.gpus) > 1:
#            model = self.model.module
#            print('model', model.device)
        opt = self.opt
        self.iter_per_epoch = len(data_loader)
        print("self.iter_per_epoch", self.iter_per_epoch)
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # batch = batch.to(device)
            torch.cuda.empty_cache()
            pre_img = batch['prev_image_rgb_input'].to(device) # bs x 3 x H x W
            pre_hm = batch['prev_belief_maps'].to(device) # bs x H x W
            pre_hm = pre_hm.unsqueeze(1)
            repro_hm = batch["repro_belief_maps"].to(device)
            repro_hm = repro_hm.unsqueeze(1)
            next_img = batch['next_image_rgb_input'].to(device)
#            print('pre_img.size', pre_img.shape)
#            print('pre_hm.size', pre_hm.shape)
#            print('next_img.size', next_img.shape)
            if phase == "PlanA":
                outputs = model(next_img, pre_img, pre_hm, repro_hm)
            elif phase == "CenterTrack+Repro":
                outputs = model(next_img, pre_img, repro_hm)
            elif phase == "CenterTrack":
                pre_origin_hm = batch["prev_origin_belief_maps"].to(device)
                pre_origin_hm = pre_origin_hm.unsqueeze(1)
                outputs = model(next_img, pre_img, pre_origin_hm)
            elif phase == "CenterTrack-Pre_hm":
                outputs = model(next_img, pre_img)
            elif phase == "CenterNet":
                outputs = model(next_img)
            elif phase == "PlanA_win":
                pre_hm_cls = batch["prev_belief_maps_cls"].to(device)
                repro_hm_cls = batch["repro_belief_maps_cls"].to(device)
                # print("pre_hm_cls.shape", pre_hm_cls.shape)
                
                outputs = model(next_img, pre_img, pre_hm, repro_hm, pre_hm_cls, repro_hm_cls)
                self.adapt_lr(epoch, batch_idx)
                
            else:
                raise ValueError 
#            outputs = model(next_img)
            loss, loss_stats = self.loss(outputs, batch)
#            print('loss', loss)
            
            if phase:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                with torch.no_grad(): 
                    if batch_idx % 10 == 0:
                        loss_all = loss_stats["tot"].item()
                        loss_hm = loss_stats["hm"].item()
                        loss_reg = loss_stats["reg"].item()
                        loss_tracking = loss_stats["tracking"].item()
                        print('loss_all', loss_all)
                        print('loss_hm', loss_hm)
                        print('loss_reg', loss_reg)
                        print('loss_tracking', loss_tracking)
                    
                    if batch_idx % 50 == 0:
                        writer.add_scalar(f"loss/training_loss", loss_all, batch_idx + (epoch-1) * len(data_loader))
                        writer.add_scalar(f"loss/heatmap_loss", loss_hm, batch_idx + (epoch-1) * len(data_loader))
                        writer.add_scalar(f"loss/reg_loss", loss_reg, batch_idx + (epoch-1) * len(data_loader))
                        writer.add_scalar(f"loss/tracking_loss", loss_tracking, batch_idx + (epoch-1) * len(data_loader))
                        # 我们一共可视化这些东西
                        # 上一帧的input图
                        # 上一帧的一张热力图
                        # 下一帧的input图
                        # 下一帧的预测热力图
                        # 下一帧的热力图gt
                        # 此处的R为4，即下采样倍数
                    
                    if batch_idx % 250 == 0:
                        output = outputs[0]
                        prev_rgb_net_inputs = dream.image_proc.images_from_tensor(batch["prev_image_rgb_input"]) # bs x 3 x H x W
                        next_rgb_net_inputs = dream.image_proc.images_from_tensor(batch["next_image_rgb_input"]) # bs x 3 x H x W
                        prev_belief_maps_wholes = batch["prev_belief_maps"] # bs x H x W
                        repro_belief_maps_wholes = batch["repro_belief_maps"] # bs x H x W
                        next_belief_maps = output["hm"] # bs x num_kp x (H/R) x (W/R)
                        next_gt_belief_maps = batch["next_belief_maps"] # bs x num_kp x (H/R) x (W/R)
                        if phase == "PlanA_win":
                            pre_hm_cls_wholes = batch["prev_belief_maps_cls"]
                            repro_hm_cls_wholes = batch["repro_belief_maps_cls"]
                        
                        for idx, sample in enumerate(zip(prev_rgb_net_inputs,next_rgb_net_inputs, prev_belief_maps_wholes, repro_belief_maps_wholes, next_belief_maps, next_gt_belief_maps)):
                            if idx < 4:
                                prev_rgb_net_input_img, next_rgb_net_input_img, prev_belief_map_whole, repro_belief_map_whole, next_belief_map, next_gt_belief_map = sample
                                # prev_rgb_net_input与next_rgb_net_input都已经是Image了
                                prev_belief_map_whole_img = dream.image_proc.image_from_belief_map(prev_belief_map_whole)
                                repro_belief_map_whole_img = dream.image_proc.image_from_belief_map(repro_belief_map_whole)
                                
                                next_belief_map_img = dream.image_proc.images_from_belief_maps(
                                next_belief_map, normalization_method=6
                                )
                                next_belief_maps_mosaic = dream.image_proc.mosaic_images(
                                next_belief_map_img, rows=3, cols=4, inner_padding_px=10
                                )
                                next_gt_belief_map_img = dream.image_proc.images_from_belief_maps(
                                next_gt_belief_map, normalization_method=6
                                )
                                next_gt_belief_maps_mosaic = dream.image_proc.mosaic_images(
                                next_gt_belief_map_img, rows=3, cols=4, inner_padding_px=10
                                )
                                
                                if phase == "PlanA_win":
                                    pre_hm_cls_img = dream.image_proc.images_from_belief_maps(
                                    pre_hm_cls_wholes[idx], normalization_method=6
                                    )
                                    pre_hm_cls_mosaic = dream.image_proc.mosaic_images(
                                    pre_hm_cls_img, rows=3, cols=4, inner_padding_px=10
                                    )
                                
                                    repro_hm_cls_img = dream.image_proc.images_from_belief_maps(
                                    repro_hm_cls_wholes[idx], normalization_method=6
                                    )
                                    repro_hm_cls_mosaic = dream.image_proc.mosaic_images(
                                    repro_hm_cls_img, rows=3, cols=4, inner_padding_px=10
                                    )
                                
                                writer.add_image(f'{idx} prev_rgb_net_input_img', np.array(prev_rgb_net_input_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                writer.add_image(f'{idx} next_rgb_net_input_img', np.array(next_rgb_net_input_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                writer.add_image(f'{idx} prev_belief_map_whole_img', np.array(prev_belief_map_whole_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                writer.add_image(f'{idx} repro_belief_map_whole_img', np.array(repro_belief_map_whole_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                writer.add_image(f'{idx} next_belief_maps_img', np.array(next_belief_maps_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                writer.add_image(f'{idx} next_gt_belief_map_img', np.array(next_gt_belief_maps_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                if phase == "PlanA_win":
                                    writer.add_image(f'{idx} pre_hm_cls', np.array(pre_hm_cls_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                                    writer.add_image(f'{idx} repro_hm_cls', np.array(repro_hm_cls_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
#                                
                
                         
    def train(self, epoch, train_loader,device, writer, phase='train'):
        return self.run_epoch(phase, epoch, train_loader, device, writer)
        
    





















