###########################################################################################
# python Dream_main.py tracking --exp_id 1  --pre_hm --same_aug --resume --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1,3 --model_last_pth model_40.pth
#
#python Dream_main.py tracking --exp_id 1  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1,3
#
#python Dream_main.py tracking --exp_id 2  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3

#python Dream_main.py tracking --exp_id 3  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_9.pth
#
#python Dream_main.py tracking --exp_id 4  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_14.pth
# 这里tracking loss乘以了0.01

#python Dream_ct_inference.py tracking --load_model /mnt/data/Dream_ty/Dream_model/center-dream/tracking/4/ckpt/model_12.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 3
#
#python Dream_ct_inference.py tracking --load_model /mnt/data/Dream_ty/Dream_model/center-dream/tracking/4/ckpt/model_6.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 0
#
#
#python Dream_main.py tracking --exp_id 4  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3
#
#python Dream_main.py tracking --exp_id 5  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 2


###########################################################################################
#python Dream_main.py tracking --exp_id 6  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 2
## tracking loss * 0.01, reg loss * 0.1
############################################################################################
#python Dream_main.py tracking --exp_id 7  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3
## tracking loss * 0.01, reg loss * 0.01
#
#python Dream_main.py tracking --exp_id 8  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0
############################################################################################
#python Dream_ct_inference.py tracking --load_model /mnt/data/Dream_ty/Dream_model/center-dream/tracking/11/ckpt/model_13.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 1

############################################################################################
#使用了dream预处理，但是用了centertrack的backbone
#python Dream_main.py tracking --exp_id 9  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0
## tracking loss * 0.01, reg loss * 0.01
#
#python Dream_main.py tracking --exp_id 10  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1
# tracking loss * 0.01, reg loss * 0.1
###########################################################################################
python Dream_main.py tracking --exp_id 11  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 2
# tracking loss * 0.01, reg loss * 0.01

# python Dream_main.py tracking --exp_id 12  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3
# tracking loss * 0.01, reg loss * 0.1

###########################################################################################
# 只有dream的loss1
#python Dream_main.py tracking --exp_id 13  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 1

#python Dream_main.py tracking --exp_id 14  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 0
## tracking loss *0.01 + reg loss * 0.01 / 使用了两个不同的decoder
############################################################################################
##单独测试是否dreamwork
#python Dream_main.py tracking --exp_id 15  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 1
## 引入refinement, 先用了dla backbone，用来dream的aug
## python Dream_main.py tracking --exp_id 16  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 3
#
## 引入refinement, 先用了dla backbone, 用了仿射变换的aug
#python Dream_main.py tracking --exp_id 17  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
## tracking监督为0
#python Dream_main.py tracking --exp_id 17  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 2 --resume --model_last_pth model_25.pth
## 引入refinement，用了dream backbone ； 一个output_decoder，3个mlp，但是之监督了两个，training中用了仿射变换的aug
#
#python Dream_main.py tracking --exp_id 18  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3
#
## 在新的数据集上跑，引入了refinement,使用了dla backbone，用了仿射变换的aug
#python Dream_main.py tracking --exp_id 19  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3
#
## 在新的数据集上跑，引入了refinement,使用了dla backbone，用了仿射变换的aug, CUDA为2
#python Dream_main.py tracking --exp_id 19  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_30.pth
#
## 在新的数据集上跑，使用了上一幀的heatmap,使用了dla backbone，用了仿射变换的aug CUDA为0
#python Dream_main.py tracking --exp_id 20  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_1.pth
#
## 在新的数据集上跑，单帧图片,使用了dla backbone，用了仿射变换的aug
#python Dream_main.py tracking --exp_id 21  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --resume --model_last_pth model_1.pth
#
## 在新的数据集上跑，单帧图片,只监督了亚像素，使用了dla backbone，用了仿射变换的aug
#python Dream_main.py tracking --exp_id 22  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
#
## planB
#python Dream_main.py tracking --exp_id 24  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1,3 --arch dlaca_34 --resume --model_last_pth model_1.pth

# planA !!!
#python Dream_main.py tracking --exp_id 25  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapa_34

## 在franka_data_1003上跑，时序，使用dla backbone，仿射变换的aug，使用repro hm （监督在亚像素了）
#python Dream_main.py tracking --exp_id 26  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase Origin --resume --model_last_pth model_20.pth
#
## 在franka_data_1003上跑，时序,仿射变换的aug, Plan A，监督在亚像素了（完蛋）
#python Dream_main.py tracking --exp_id 25  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapa_34 --phase PlanA --resume --model_last_pth model_47.pth --is_real panda-3cam_realsense
#
## 在franka_data_1003上跑，时序，训练时仿射变换的aug, origin, 但使用pre_hm而不是repro_hm, 使用dla的backbone, 监督在亚像素了（完蛋）
#python Dream_main.py tracking --exp_id 27  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase Origin_worepro
#
## 在franka_data_1003和franka_data_0909上混合跑，时序，训练时使用仿射变换的aug，PlanA
#python Dream_main.py tracking --exp_id 28  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 2,3 --arch dlapa_34 --phase PlanA --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --batch_size 32 --is_real panda-3cam_realsense --resume --model_last_pth model_11.pth
#
## 在franka_data_0909上跑，PLANa, 仿射变换的aug，监督在整像素上了
#python Dream_main.py tracking --exp_id 30  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapa_34 --phase PlanA --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense
#
## 在franka_data_0909上跑，CenterTrack - pre_hm，仿射变换的aug，监督在整像素了
#python Dream_main.py tracking --exp_id 31  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterTrack-Pre_hm --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense
#
## 在franka_data_1010上跑，plana直接cat，仿射变换的aug，监督在整像素了
#python Dream_main.py tracking --exp_id 32  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapa_341 --phase PlanA --is_real panda-3cam_realsense
#
#python Dream_main.py tracking --exp_id 33  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapa_34 --phase PlanA --is_real panda-3cam_realsense
#
## PlanA with window, 测试
#python Dream_main.py tracking --exp_id 34  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense
#
## PlanA with window ，在franka_data_0909上跑，仿射变化AUG，监督在整像素，学习率衰减,一共40个epoch --lr 1e-4， reg为0.01, hm 1
#python Dream_main.py tracking --exp_id 35  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense --num_epochs 40
#
#python Dream_main.py tracking --exp_id 35  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense --num_epochs 40 --resume --model_last_pth model_34.pth
#
## PlanA with window , 在混合数据上跑，仿射变化aug,监督在整像素，学习率按照iteration衰减，lr=1.25e-4， reg为0.01, hm 1
#python Dream_main.py tracking --exp_id 36  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3,4,5 --arch dlapawd_34 --phase PlanA_win --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1003/" --is_real panda-3cam_realsense --num_epochs 40 --batch_size 64  
#
## PLanA with window, 在franka data 0909跑，学习率按照iteration衰减，lr=1.25e-4, multi-head为4，一层transformer，reg为0.01, hm 1
#python Dream_main.py tracking --exp_id 37  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense --num_epochs 40 --max_iters 3e5 --resume --model_last_pth model_11.pth
#
## PlanA with window, 在franka data 0909跑，学习率按照iteration衰减，lr=1.25e-4, multi-head为8,3层transformer，但使用focol loss, reg loss为0.1, hm loss 0.1
#python Dream_main.py tracking --exp_id 38  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_0909/" --is_real panda-3cam_realsense --num_epochs 40 --max_iters 3e5 --resume --model_last_pth model_9.pth
#
## PlanA with window, 在franka data 1020跑，学习率按照Iteration衰减，lr=1.25e-4，multi-head为8， 3层trans，使用mseloss， hm loss1, reg loss 0.01
#python Dream_main.py tracking --exp_id 39 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40 --max_iters 3e5 
#
## CenterTrack, 在franka data 1020跑，lr=1.25e-4，hm loss1, reg loss 0.01
#python Dream_main.py tracking --exp_id 40 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterTrack --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40
#
## CenterNet,  在franka data 1020跑，lr=1.25e-4，hm loss1, reg loss 0.01
#python Dream_main.py tracking --exp_id 41 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterNet --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40

python Dream_main.py tracking --exp_id 39 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40 --max_iters 3e5 --resume --model_last_pth model_8.pth

python Dream_main.py tracking --exp_id 40 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterTrack --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40 --resume --model_last_pth model_11.pth

python Dream_main.py tracking --exp_id 41 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterNet --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40 --resume --model_last_pth model_11.pth

# 没有pos embedding，PlanA with window, 在franka data 1020跑，学习率按照Iteration衰减，lr=1.25e-4，multi-head为8， 3层trans，使用mseloss， hm loss1, reg loss 0.01
python Dream_main.py tracking --exp_id 45 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --is_real panda-3cam_realsense --num_epochs 40 --pos_embed --resume --model_last_pth model_10.pth

python Dream_main_ku.py tracking --exp_id 42 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1019_kuka/" --val_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_validation/" --syn_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_syn_test/" --pure_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_pure_test/" --is_real panda-3cam_realsense --num_epochs 40 --robot KUKA_LBR_Iiwa14 --num_classes 9

python Dream_main_ku.py tracking --exp_id 43 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterTrack --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1019_kuka/" --val_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_validation/" --syn_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_syn_test/" --pure_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_pure_test/" --is_real panda-3cam_realsense --num_epochs 40 --robot KUKA_LBR_Iiwa14 --num_classes 9

python Dream_main_ku.py tracking --exp_id 44 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterNet --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1019_kuka/" --val_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_validation/" --syn_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_syn_test/" --pure_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/kuka_pure_test/" --is_real panda-3cam_realsense --num_epochs 40 --robot KUKA_LBR_Iiwa14 --num_classes 9


# 没有Positional embedding 的planA with window
python Dream_main_ku.py tracking --exp_id 46 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1019_ur5e/" --val_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/ur5e_validation/" --syn_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/ur5e_syn_test/" --pure_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/ur5e_pure_test/" --is_real panda-3cam_realsense --num_epochs 40 --robot UR5e --num_classes 8 --pos_embed

# 混合数据，有pos_embed
python Dream_main.py tracking --exp_id 50 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense --num_epochs 35 --resume --model_last_pth model_6.pth
# 混合数据，没有pos_embed
python Dream_main.py tracking --exp_id 51 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapawd_34 --phase PlanA_win --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense --num_epochs 35 --pos_embed
# 混合数据 centertrack
python Dream_main.py tracking --exp_id 52 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterTrack --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense --num_epochs 35
# 混合数据 centernet
python Dream_main.py tracking --exp_id 53 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase CenterNet --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense --num_epochs 35


# 混合数据 alabtion_wo_shared
python Dream_main.py tracking --exp_id 54 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlaabla_34 --phase ablation_wo_shared --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense 
# 混合数据 ablation_shared
python Dream_main.py tracking --exp_id 55 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlaabla_34 --phase ablation_shared --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense 
# 混合数据 ablation_shared_repro
python Dream_main.py tracking --exp_id 56 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlaabla_34 --phase ablation_shared_repro --dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/franka_data_1020/" --add_dataset "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/test_1020/near_franka_data_1024/" --is_real panda-3cam_realsense 


python Results_save.py tracking --exp_id 39 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 40 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 41 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterNet --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 45 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense --pos_embed

python Results_save.py tracking --exp_id 50 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 51 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense --pos_embed

python Results_save.py tracking --exp_id 52 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 53 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterNet --is_real panda-3cam_realsense


python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/ckpt/model_30.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/ckpt/model_15.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/51/ckpt/model_13.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360 --pos_embed

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/ckpt/model_13.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-orb

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/52/ckpt/model_7.pth --phase CenterTrack --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/53/ckpt/model_8.pth --phase CenterNet --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360












############################################# INFERENCE #######################################################

#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/35/ckpt/model_33.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/35/ckpt/model_35.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/35/ckpt/model_35.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/35/ckpt/model_35.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-orb
#
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/36/ckpt/model_18.pth --arch dlapawd_34 --phase PlanA_win --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
#python Results_save.py tracking --exp_id 32 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapacat_341 --phase PlanACAT --is_real panda-3cam_realsense
#
#python Results_save.py tracking --exp_id 35 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense 
#
#python Results_save.py tracking --exp_id 36 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapawd_34 --phase PlanA_win --is_real panda-3cam_realsense 
#
#python Results_save.py tracking --exp_id 30 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapa_34 --phase PlanA --is_real panda-3cam_realsense




# 老数据集上synthetic和pure这里是不对的，直接看 real就好了




# 测试
python Dream_main.py tracking --exp_id 29  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0 --batch_size 1 --num_workers 0

#######################################################INFERENCE###########################################################
#python Results_save.py tracking --exp_id 19 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
#
#python Results_save.py tracking --exp_id 20 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
#
#python Results_save.py tracking --exp_id 21 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1

# python Results_save.py tracking --exp_id 21 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlaca_34

#python Results_save.py tracking --exp_id 25 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapa_34 --phase PlanA --is_real panda-3cam_realsense
#
#python Results_save.py tracking --exp_id 26 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack+Repro --is_real panda-3cam_realsense
#
#python Results_save.py tracking --exp_id 27 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack --is_real panda-3cam_realsense

python Results_save.py tracking --exp_id 19 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack+Repro 

python Results_save.py tracking --exp_id 20 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack 

python Results_save.py tracking --exp_id 21 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterNet 

python Results_save.py tracking --exp_id 25 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapa_34 --phase PlanA 

python Results_save.py tracking --exp_id 26 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack+Repro

python Results_save.py tracking --exp_id 27 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --phase CenterTrack 

#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_39.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4
##
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_26.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4
#
## tmux为0
python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_20.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.1 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
## tmux为5
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_42.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
## tmux为7
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_42.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
## tmux為8
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_42.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense 
#
#################################################################################################################
## tmux为4
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/19/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
## tmux为5
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/19/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
## tmux为6
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/19/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
## tmux为7
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/19/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-orb
#
## tmux为4
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_63.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-orb
#
## tmux为9
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_63.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
## tmux为5 
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_63.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
## tmux为7
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_63.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
## tmux为5
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
## tmux为7
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_20.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
## tmux为8
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_20.pth --phase CenterTrack --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense
#
## tmux为5
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_20.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_realsense 
#
## tmux为6
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_20.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_azure
#
## tmux为7
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_20.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-3cam_kinect360
#
## tmux为8
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_20.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4 --is_real panda-orb
#
# is_real \in { panda-orb, panda-3cam_realsense, panda-3cam_kinect360, panda-3cam_azure  }
#########################dataset###############################
"/root/autodl-tmp/camera_to_robot_pose/Dream_ty/synthetic_test_1018/" 新造数据集，朴素背景，30帧为一个视频，400个

###############################################################
36,37中段在real上的Inference有问题，需要重新弄
