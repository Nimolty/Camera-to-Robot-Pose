# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

def keypoint_metrics(
    keypoints_detected, keypoints_gt, image_resolution, auc_pixel_threshold=20.0, real=False
):

    # TBD: input argument handling
    num_gt_outframe = 0
    num_gt_inframe = 0
    num_missing_gt_outframe = 0
    num_found_gt_outframe = 0
    num_found_gt_inframe = 0
    num_missing_gt_inframe = 0
    
    N, _ = keypoints_gt.shape
    
    if real:
        gap = 80
    else:
        gap = 140
    kp_errors = []
    for kp_proj_detect, kp_proj_gt in zip(keypoints_detected, keypoints_gt):

        if (
            # kp_proj_gt[0] <= 140.0
            kp_proj_gt[0] < 0.0
            # or kp_proj_gt[0] >= image_resolution[0] - 140.0
            or kp_proj_gt[0] > image_resolution[0]
            or kp_proj_gt[1] < 0.0
            or kp_proj_gt[1] > image_resolution[1]
        ):
            # GT keypoint is out of frame
            num_gt_outframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (correct)
                num_missing_gt_outframe += 1
            else:
                # Found a keypoint (wrong)
                num_found_gt_outframe += 1

        else:
            # GT keypoint is in frame
            num_gt_inframe += 1

            if kp_proj_detect[0] < -999.0 and kp_proj_detect[1] < -999.0:
                # Did not find a keypoint (wrong)
                num_missing_gt_inframe += 1
                # print('order', json_order)
                # print('kp_proj_gt', kp_proj_gt)
            else:
                # Found a keypoint (correct)
                num_found_gt_inframe += 1

                kp_errors.append((kp_proj_detect - kp_proj_gt).tolist())

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)

        # compute the auc
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_pixel_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel)
            / float(auc_pixel_threshold)
            / float(num_gt_inframe)
        )

    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None

    metrics = {
        "num_gt_outframe": num_gt_outframe,
        "num_missing_gt_outframe": num_missing_gt_outframe,
        "num_found_gt_outframe": num_found_gt_outframe,
        "num_gt_inframe": num_gt_inframe,
        "num_found_gt_inframe": num_found_gt_inframe,
        "num_missing_gt_inframe": num_missing_gt_inframe,
        "l2_error_mean_px": kp_l2_error_mean,
        "l2_error_median_px": kp_l2_error_median,
        "l2_error_std_px": kp_l2_error_std,
        "l2_error_auc": kp_auc,
        "l2_error_auc_thresh_px": auc_pixel_threshold,
    }
    return metrics


# Example of running the script
# python oks_plots.py --data all_dataset_keypoints.csv all_dataset_keypoints.csv --labels 1 2

# pythonw oks_plots.py --data deep-arm-cal-paper/data/dope/3cam_real_keypoints.csv deep-arm-cal-paper/data/dream_hg/3cam_real_keypoints.csv deep-arm-cal-paper/data/dream_hg_deconv/3cam_real_keypoints.csv deep-arm-cal-paper/data/resimple/3cam_real_keypoints.csv --labels DOPE DREAM AE resnet
parser = argparse.ArgumentParser(description="OKS for DREAM")
parser.add_argument(
    "--data", nargs="+", default="[all_dataset_keypoints.csv]", help="list of csv files"
)

parser.add_argument(
    "--labels",
    nargs="+",
    default=None,
    help="names for each dataset to be added as label",
)

parser.add_argument("--styles", nargs="+", default=None, help="")

parser.add_argument("--colours", nargs="+", default=None, help="")

parser.add_argument("--pixel", default=20)

parser.add_argument("--output", default="output_pck.pdf")

parser.add_argument("--show", default=False, action="store_true")

parser.add_argument("--title", default=None)

args = parser.parse_args()
args.colours = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
args.labels = ["Panda_Orb", "Ours", "CenterTrack", "CenterNet", "DREAM",
               "Panda_Azure", "Ours", "CenterTrack", "CenterNet", "DREAM",
               "Panda_RealSense", "Ours", "CenterTrack", "CenterNet", "DREAM"]
args.styles = ["3", "0", "1", "2", "3", 
               "3", "0", "1", "2", "3",
               "3", "0", "1", "2", "3"]
               
args.data = ["666",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/results/model_20/panda-orb_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/52/results/model_20/panda-orb_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/53/results/model_20/panda-orb_keypoints.csv",
             "DREAM",
             "666",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/results/model_20/panda-3cam_azure_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/52/results/model_20/panda-3cam_azure_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/53/results/model_20/panda-3cam_azure_keypoints.csv",
             "DREAM",
             "666",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/50/results/model_20/panda-3cam_realsense_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/52/results/model_20/panda-3cam_realsense_keypoints.csv",
             "/root/autodl-tmp/camera_to_robot_pose/Dream_ty/Dream_model/center-dream/tracking/53/results/model_20/panda-3cam_realsense_keypoints.csv",
             "DREAM"
            ]

print(args)


fig = plt.figure()
ax = plt.axes()

handles = []

for i_csv, csv_file in enumerate(args.data):
    print(csv_file)
    if csv_file == "666" or "DREAM" in csv_file:
        plt.plot([], [], " ", label=args.labels[i_csv].replace("_", " "))
        continue
    name_csv = csv_file.replace(".csv", "")

    df = pd.read_csv(csv_file)

    # PCK percentage of correct keypoints
    all_dist = []
    all_pred = []
    all_gt = []
    for i in range(7):
        # Compute all the distances between keypoints - Implementing them does not work well

        fpred = []
        fgt = []
        gt = df[[f"kp{i}x_gt", f"kp{i}y_gt"]].values.tolist()
        pred = df[[f"kp{i}x", f"kp{i}y"]].values.tolist()

        all_gt.append(gt)
        all_pred.append(pred)

        for i_entry, entry in enumerate(gt):
            if entry[0] > 0 and entry[0] < 640 and entry[1] > 0 and entry[1] < 480:
                fgt.append([float(entry[0]), float(entry[1])])
                fpred.append([float(pred[i_entry][0]), float(pred[i_entry][1])])

        pred = np.array(fpred)
        gt = np.array(fgt)

        values = np.linalg.norm(gt - pred, axis=1)

        # print(pair.shape)
        # add them to a single list

        all_dist += values.tolist()

    all_dist = np.array(all_dist)

    # print(len(all_dist))

    # all_dist = all_dist[np.where(all_dist<1000)]
    # all_dist = all_dist[np.where(all_dist<1000)]

    # print(all_dist[:10])
    # print(all_dist.shape)
    print("detected", len(all_dist))

    pck_values = np.arange(0, int(args.pixel), 0.01)

    y_values = []

    for value in pck_values:
        size_good = len(np.where(all_dist < value)[0]) / len(all_dist)
        y_values.append(size_good)
        # print(value,size_good)

    auc = np.trapz(y_values, dx=0.01) / float(args.pixel)

    print("auc", auc)
    all_dist = all_dist[np.where(all_dist < 1000)]
    print("mean", np.mean(all_dist))
    print("median", np.median(all_dist))
    print("std", np.std(all_dist))

    # TODO: consolidate above calculations with pnp_metrics

    dim = np.array(all_pred).shape
    temp_pred = np.reshape(np.array(all_pred), (dim[0] * dim[1], dim[2]))
    temp_gt = np.reshape(np.array(all_gt), (dim[0] * dim[1], dim[2]))
    kp_metrics = keypoint_metrics(temp_pred, temp_gt, (640, 480))
    assert kp_metrics["l2_error_mean_px"] == np.mean(all_dist)
    assert kp_metrics["l2_error_median_px"] == np.median(all_dist)
    assert kp_metrics["l2_error_std_px"] == np.std(all_dist)
    assert np.abs(auc - kp_metrics["l2_error_auc"]) < 1.0e-15

    # plot

    try:
        label = args.labels[i_csv]
    except:
        label = name_csv
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    try:

        colour = cycle[int(args.colours[i_csv])]
    except:
        colour = ""

    try:
        style = args.styles[i_csv]
        if style == "0":
            style = "-"
        elif style == "1":
            style = "--"
        elif style == "2":
            style = ":"
        elif style == "3":
            style = "-."

        else:
            style = "-"
    except:
        style = "-"

    label = f"{label} ({auc:.3f})"
    ax.plot(pck_values, y_values, style, color=colour, label=label)

# from matplotlib.patches import Rectangle

# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


plt.xlabel("PCK threshold distance (pixels)")
plt.ylabel("Accuracy")
plt.title(args.title)
# ax.legend([extra, handles[0], handles[1], handles[2], extra , handles[3], handles[4], handles[5] ], ("Non-DR",0,1,2 ,"DR",0,1,2),loc = "lower right")
ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.8)

legend = ax.get_legend()
for i, t in enumerate(legend.get_texts()):
    if args.data[i] == "666":
        t.set_ha("left")  # ha is alias for horizontalalignment
        t.set_position((-30, 0))

ax.set_ylim(0, 1)
ax.set_xlim(0, int(args.pixel))
plt.savefig(args.output)
if args.show:
    plt.show()
