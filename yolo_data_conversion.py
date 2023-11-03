"""
This python file is retrieves the videos used in Yolo v4 tiny object detection and filters
the data within the pose_vides and pose_joints json files to only include the videos used
"""

import os
import json

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

path_train_obj1 = "D:\\Users\\SMH\\Tim_v2\\data\\train_object_videos_yolo.json"
path_val_obj1 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_object_videos_yolo.json"

path_train_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_videos.json"
path_train_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_joints.json"

path_val_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_videos.json"
path_val_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_joints.json"

train_obj1 = get_data(path_train_obj1)
val_obj1 = get_data(path_val_obj1)
train_pose1 = get_data(path_train_pose1)
train_pose2 = get_data(path_train_pose2)
val_pose1 = get_data(path_val_pose1)
val_pose2 = get_data(path_val_pose2)

train_videos = set()
val_videos = set()

for frame in train_obj1:
    train_videos.add(frame[0])

for frame in val_obj1:
    val_videos.add(frame[0])

def generate_filtered_frames(pose_videos, pose_detected, train_videos):
    new_pose_videos = []
    new_pose_detected = []
    for index, video_frame in enumerate(pose_videos):
        pose_frame = pose_detected[index]
        if video_frame[0] in train_videos:
            new_pose_videos.append(video_frame)
            new_pose_detected.append(pose_frame)
    return new_pose_videos, new_pose_detected

def save_json(data, file):
    with open(file, "w") as outfile:
        json.dump(data, outfile)

print("FILTERING VIDEOS")
new_val_pose1, new_val_pose2 =  generate_filtered_frames(val_pose1, val_pose2, val_videos)
new_train_pose1, new_train_pose2 =  generate_filtered_frames(train_pose1, train_pose2, train_videos)

print("STORING FILTERED VIDEOS")
new_path_train_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_videos_yolo.json"
new_path_train_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_joints_yolo.json"
new_path_val_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_videos_yolo.json"
new_path_val_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_joints_yolo.json"

save_json(new_train_pose1, new_path_train_pose1)
save_json(new_train_pose2, new_path_train_pose2)
save_json(new_val_pose1, new_path_val_pose1)
save_json(new_val_pose2, new_path_val_pose2)
print("COMPLETED")