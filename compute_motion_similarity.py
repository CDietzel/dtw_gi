import pickle
from itertools import chain, repeat, tee
from math import sqrt
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from tslearn.metrics import ctw

from dtw_gi import dtw_gi, softdtw_gi


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        return pickle.load(file)


def extract_human_keypoints(poses):
    keypoints_list = []
    for pose in poses:
        if pose is not None:
            keypoints_list.append(pose.landmarks_world)  # Only use this,
            # keypoints_list.append(pose.norm_landmarks)  # Not these.
            # keypoints_list.append(pose.landmarks)  # They are image-relative
    keypoints = np.array(keypoints_list)[:, 0:33, :]
    return keypoints


pre_swap = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
post_swap = [
    0,
    4,
    5,
    6,
    1,
    2,
    3,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
    24,
    23,
    26,
    25,
    28,
    27,
    30,
    29,
    32,
    31,
]


# UPDATE THESE PATHS
model_output_folder = Path("/home/dietzelcc/Documents/Repos/ibc/ibc/output")
human_motion_folder = Path("/home/dietzelcc/Documents/Repos/depthai_blazepose/outputs/")
loss_output = Path("/home/dietzelcc/Documents/Repos/dtw_gi/output")

robot_motion_prefix = "oracle_interbotix_"
robot_motion_postfix = "_n"
robot_motion_extension = ".modulated"
human_motion_extension = ".pickle"
motion_name_list = ["test1", "test2", "test3", "test4", "test5"]

tag_list = ["ibc_langevin_test", "mse_test"]
num_seeds = 3
num_hyperparameters = [18, 18]

sum_hyperparameters = sum(num_hyperparameters)
total_hyperparameters = num_seeds * sum_hyperparameters




full_ctw_scores = []
full_soft_dtw_gi_scores = []
full_dtw_gi_scores = []
full_nums = []
full_tags = []
full_names = []

avg_ctw_scores = []
avg_soft_dtw_gi_scores = []
avg_dtw_gi_scores = []
avg_nums = []
avg_tags = []
avg_names = []

for i, tag in enumerate(
    chain.from_iterable(
        tee(
            chain.from_iterable(
                repeat(x[1], x[0]) for x in zip(num_hyperparameters, tag_list)
            ),
            num_seeds,
        )
    )
):
    temp_ctw_scores = []
    temp_soft_dtw_gi_scores = []
    temp_dtw_gi_scores = []
    robot_motion_folder = model_output_folder / str(tag) / str(i)
    for name in motion_name_list:
        robot_motion_path = robot_motion_folder / (
            robot_motion_prefix + name + robot_motion_postfix + robot_motion_extension
        )
        human_motion_path = human_motion_folder / (name + human_motion_extension)

        human_poses = load_pickle(human_motion_path)
        robot_data = load_pickle(robot_motion_path)
        human_data = extract_human_keypoints(human_poses)

        # Only include arm keypoints (11-22)
        human_data = human_data[:, 11:23, :]

        # Reshape to flatten xyz coordinates for all keypoints for each frame
        human_data = human_data.reshape(-1, 36)

        # Ignore first few samples (noisy data)
        human_data = human_data[3:]

        ctw_loss = ctw(robot_data, human_data)



        
        _, _, dtw_gi_loss = dtw_gi(robot_data, human_data) # type: ignore
        soft_dtw_gi_loss = softdtw_gi(robot_data, human_data)

        full_ctw_scores.append(ctw_loss)
        full_dtw_gi_scores.append(dtw_gi_loss)
        full_soft_dtw_gi_scores.append(soft_dtw_gi_loss)
        full_nums.append(i)
        full_tags.append(tag)
        full_names.append(name)

        temp_ctw_scores.append(ctw_loss)
        temp_dtw_gi_scores.append(dtw_gi_loss)
        temp_soft_dtw_gi_scores.append(soft_dtw_gi_loss)

    avg_ctw_scores.append(mean(temp_ctw_scores))
    avg_dtw_gi_scores.append(mean(temp_dtw_gi_scores))
    avg_soft_dtw_gi_scores.append(mean(temp_soft_dtw_gi_scores))
    avg_nums.append(i)
    avg_tags.append(tag)
    print("score for num " + str(i) + " is: " + str(mean(temp_dtw_gi_scores)))

full_results = pd.DataFrame(
    {
        "Model ID": full_nums,
        "Model Type": full_tags,
        "Test File": full_names,
        "CTW Scores": full_ctw_scores,
        "DTW-GI Scores": full_dtw_gi_scores,
        "Soft DTW-GI Scores": full_soft_dtw_gi_scores,
    }
)
avg_results = pd.DataFrame(
    {
        "Model ID": avg_nums,
        "Model Type": avg_tags,
        "CTW Scores": avg_ctw_scores,
        "DTW-GI Scores": avg_dtw_gi_scores,
        "Soft DTW-GI Scores": avg_soft_dtw_gi_scores,
    }
)

loss_output.mkdir(parents=True, exist_ok=True)

full_results.to_csv(loss_output / "dtw_gi_full_results.csv", index=False)
avg_results.to_csv(loss_output / "avg_results.csv", index=False)

avg_results[
    ["CTW Scores", "DTW-GI Scores", "Soft DTW-GI Scores"]
] = avg_results.groupby(avg_results.index % sum_hyperparameters)[
    ["CTW Scores", "DTW-GI Scores", "Soft DTW-GI Scores"]
].sum(
    numeric_only=True
)

avg_results.drop(
    avg_results.tail(sum_hyperparameters * (num_seeds - 1)).index, inplace=True
)

avg_results[["CTW Scores", "DTW-GI Scores", "Soft DTW-GI Scores"]] /= 3

avg_results.to_csv(loss_output / "avg_avg_results.csv", index=False)

for tag_num, tag in enumerate(tag_list):
    filtered_avg = avg_results.loc[avg_results["Model Type"] == tag]

    best_ctw = filtered_avg[
        filtered_avg["CTW Scores"] == filtered_avg["CTW Scores"].min()
    ]
    best_dtw_gi = filtered_avg[
        filtered_avg["DTW-GI Scores"] == filtered_avg["DTW-GI Scores"].min()
    ]
    best_soft_dtw_gi = filtered_avg[
        filtered_avg["Soft DTW-GI Scores"] == filtered_avg["Soft DTW-GI Scores"].min()
    ]
    best_ctw.to_csv(loss_output / ("best_ctw_" + str(tag) + ".csv"), index=False)
    best_dtw_gi.to_csv(loss_output / ("best_dtw_gi_" + str(tag) + ".csv"), index=False)
    best_soft_dtw_gi.to_csv(
        loss_output / ("best_soft_dtw_gi_" + str(tag) + ".csv"), index=False
    )
