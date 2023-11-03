import json
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Concatenate, Permute, multiply, Reshape, Input, concatenate
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from tensorflow.keras.layers import LayerNormalization, Attention
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# n = number of clips
# pose = [n x 10 x 19 x 2]
# pose_conf = [n x 10 x 19 x 1]
# objects_labels = [n x 10 x 91]
# object_bounding_boxes = [n x 10 x 4]
# action_labels = 25

# create object label map
objlabel_map_path = "C:\\Users\\mwil0091\\open_model_zoo\\demos\\human_pose_estimation_3d_demo\\python\\labels\\coco_80cl.txt"
objlabel_map_dict = dict()
objlabel_categories = []
with open(objlabel_map_path, 'r') as f:
    index = 0
    for line in f:
        line = line.strip('\n')
        objlabel_map_dict[line] = index
        objlabel_categories.append(line)
        index += 1
one_hot_encoder = OneHotEncoder(categories=[objlabel_categories], drop=None, sparse=False)

def get_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

val_filtered_dataset_path = "D:\\Users\\SMH\\Tim_v2\\data\\val_dataset_actionobject_filtered_yolo.json"
train_filtered_dataset_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_datase_actionobject_filtered_yolo.json"
val_filtered = get_json_data(val_filtered_dataset_path)
train_filtered = get_json_data(train_filtered_dataset_path)

# {video_name1: [clip_1, clip_2], video_name2: [clip_1, clip_2], ...}
# clip_1 = [frame_1, frame_2, ...]
# frame_1 = [action_label, object_label, object_coordinates, joint_coordinates]


def format_data(objlabel_map_dict, dataset):
    """
    This function takes in a dictionary of object labels and a dataset of video clips. It formats the data by extracting 
    2D coordinates, confidence scores, object labels, one-hot encoded object labels, object bounding boxes, action labels, 
    and clip count. It returns the formatted data as numpy arrays.
    
    Args:
    - objlabel_map_dict: A dictionary of object labels.
    - dataset: A dataset of video clips.
    
    Returns:
    - coord_2d: A numpy array of 2D coordinates.
    - coord_2d_conf: A numpy array of confidence scores.
    - object_labels: A numpy array of object labels.
    - object_bounding_boxes: A numpy array of object bounding boxes.
    - clip_action_labels: A numpy array of action labels.
    """
    coord_2d = []
    coord_2d_conf =[]
    object_labels = []
    one_hot_obj_labels = []
    object_bounding_boxes = []
    clip_action_labels = []
    clip_count = 0
    action_dict = dict()
    label_count = -1
    for video_name in dataset.keys():
        for clip_index, clip in enumerate(dataset[video_name]):
            coord_2d.append([])
            coord_2d_conf.append([])
            object_labels.append([])
            object_bounding_boxes.append([])
            one_hot_obj_labels.append([])
        # retrieve the action label from the first frame of the clip
            first_frame = clip[0]
            action_label = first_frame[0]
            try:
                action_dict[action_label]
            except Exception as e:
                label_count += 1
                action_dict[action_label] = label_count
            clip_action_labels.append(label_count)
        
            frame_count = 0
        # go through each frame in clip
            for frame_index, frame in enumerate(clip):
            # add frame list to clip
                coord_2d[clip_count].append([])
                coord_2d_conf[clip_count].append([])
                object_labels[clip_count].append([])
                object_bounding_boxes[clip_count].append([])
            # one_hot_obj_labels[clip_count].append([])
            # get object data
                frame_obj_label = frame[1]
                obj_label_dim = [0 if _ != 0 else 1 for _ in range(len(objlabel_map_dict.keys())) ]
                obj_bbx_dim = [0 for _ in range(4)]
                if frame_obj_label != "None":
                    obj_label_dim[objlabel_map_dict[frame_obj_label]] = 1
                # object_bounding_boxes[clip_count][frame_count].append(frame[2:6])
                    obj_bbx_dim = frame[2:6]
            # else:
                # frame_obj_label = "__background__"
                # object_bounding_boxes[clip_count][frame_count].append([0,0,0,0])
                # object_bounding_boxes[clip_count][frame_count] = [0,0,0,0]
            # object_labels[clip_count][frame_count].append(obj_label_dim)
                object_labels[clip_count][frame_count] = obj_label_dim
                object_bounding_boxes[clip_count][frame_count].append(obj_bbx_dim)
            # categorical_data = np.array(frame_obj_label)
            # categorical_data = categorical_data.reshape(-1, 1)
            # encoded_data = one_hot_encoder.fit_transform(categorical_data)
            # one_hot_obj_labels.append(encoded_data)

            # get joint data
                joint_data = first_frame[-58:]
            # remove overall confidence score
                joint_data = joint_data[:len(joint_data)-1]
            # get each joint data
                for joint_index in range(0,len(joint_data),3):
                    coord = joint_data[joint_index:joint_index+2]
                    conf = joint_data[joint_index+2]
                    coord_2d[clip_count][frame_count].append(coord)
                    coord_2d_conf[clip_count][frame_count].append(conf)
                frame_count += 1
            clip_count += 1
    
    coord_2d = np.array(coord_2d)
    coord_2d_conf = np.array(coord_2d_conf)
    object_labels = np.array(object_labels)
    object_bounding_boxes = np.array(object_bounding_boxes)
    clip_action_labels = np.array(clip_action_labels)
    return coord_2d,coord_2d_conf,object_labels,object_bounding_boxes,clip_action_labels

# coord_2d, coord_2d_conf, object_labels, object_bounding_boxes, clip_action_labels = format_data(objlabel_map_dict, val_filtered)


x_train_pose, x_train_confidence, x_train_object_classes, x_train_object_boxes, y_train_labels = format_data(objlabel_map_dict, train_filtered)
x_val_pose, x_val_confidence, x_val_object_classes, x_val_object_boxes, y_val_labels = format_data(objlabel_map_dict, val_filtered)

save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_pose_yolo.json", x_val_pose.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_confidence_yolo.json", x_val_confidence.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_object_classes_yolo.json", x_val_object_classes.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_object_boxes_yolo.json", x_val_object_boxes.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\y_val_labels_yolo.json", y_val_labels.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_pose_yolo.json", x_train_pose.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_confidence_yolo.json", x_train_confidence.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_object_classes_yolo.json", x_train_object_classes.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_object_boxes_yolo.json", x_train_object_boxes.tolist())
save_json_data("D:\\Users\\SMH\\Tim_v2\\data\\y_train_labels_yolo.json", y_train_labels.tolist())