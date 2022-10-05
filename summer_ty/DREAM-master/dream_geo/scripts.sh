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
python Dream_main.py tracking --exp_id 25  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlapa_34

# 在franka_data_1003上跑，时序，使用dla backbone，仿射变换的aug，使用repro hm
python Dream_main.py tracking --exp_id 26  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase Origin

python Dream_main.py tracking --exp_id 27  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --phase Origin

# 在franka_data_1003上跑，时序,仿射变换的aug, Plan A
python Dream_main.py tracking --exp_id 25  --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 3 --arch dlapa_34 --phase PlanA

#python Results_save.py tracking --exp_id 19 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
#
#python Results_save.py tracking --exp_id 20 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1
#
#python Results_save.py tracking --exp_id 21 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1

# python Results_save.py tracking --exp_id 21 --pre_hm --same_aug --hm_disturb 0.75 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 1 --arch dlaca_34




python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/26/ckpt/model_1.pth --phase Origin --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4

python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/25/ckpt/model_1.pth --phase PlanA --arch dlapa_34 --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4
#
#
#python Dream_ct_inference.py tracking --load_model /root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/21/ckpt/model_20.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 4