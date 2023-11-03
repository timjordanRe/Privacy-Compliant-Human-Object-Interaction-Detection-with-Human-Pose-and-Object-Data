import json
import os

# pose videos = [video name, frame id, action label]
# pose joints = [joint coordinates]
# object videos = [video name, frame id, action label, num of detected objects]
# object recognition = [object label, object coordinates]


path_train_obj1 = "D:\\Users\\SMH\\Tim_v2\\data\\train_object_videos_yolo.json"
path_train_obj2 = "D:\\Users\\SMH\\Tim_v2\\data\\train_object_recognition_yolo.json"
path_train_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_videos_yolo.json"
path_train_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\train_pose_joints_yolo.json"
path_val_obj1 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_object_videos_yolo.json"
path_val_obj2 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_object_recognition_yolo.json"
path_val_pose1 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_videos_yolo.json"
path_val_pose2 = "D:\\Users\\SMH\\Tim_v2\\data\\validate_pose_joints_yolo.json"
action_object_path = "C:\\Users\\mwil0091\\open_model_zoo\\demos\\human_pose_estimation_3d_demo\\python\\labels\\action_object_mapping.json"
def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def count_vid_unique_frames(data):
    unique_vid_frames = dict()
    for el in data:
        vid_name = el[0]
        try:
            unique_vid_frames[vid_name] += 1
        except Exception as e:
            unique_vid_frames[vid_name] = 1
    return unique_vid_frames

def find_recognised_objects(obj_recognition_ls, associated_object):
    res = obj_recognition_ls[0]
    for object in enumerate(obj_recognition_ls):
        if object[0] == associated_object:
            res = object
    return res

# assumptions: only one person in frame, only one object in frame

# dataset format
# {vid_name1: [
#               [action label (nullable), object label (nullable), object bounding boxes (4 points), 58 joint coordinates], //frame1
#               [action label (nullable), object label (nullable), object bounding boxes (4 points), 58 joint coordinates], //frame2
#             ],
# vid_name2: [...}

def generate_dataset(obj1, obj2, pose1, pose2, action_object_mapping):
    dataset = dict()
    val_length = len(obj1)
    obj_counter = 0  # keeps track of how many detected objects have been processed
    crack = False
    for i in range(val_length):
        # check if retrieved video pose frame and object frame match
        pose_frame = pose1[i][1]
        object_frame = obj1[i][1]
        pose_vid_name = pose1[i][0]
        object_vid_name = obj1[i][0]
        frame_data = []
        if pose_vid_name != object_vid_name or pose_frame != object_frame:
            raise Exception(f'Frame mismatch: pose frame {pose_frame} != object frame {object_frame} between pose video {pose_vid_name} and object video {object_vid_name}')
        # retrieve data
        action_label = pose1[i][2]
        joint_coordinates = pose2[i]
        object_detect_num = obj1[i][3]
        object_label = "None"
        object_coordinates = []
        # check if video name is in dataset
        try:
            dataset[pose_vid_name]
        except Exception as e:
            dataset[pose_vid_name] = []
        if object_detect_num != 0 and action_label.lower() != "none":
            associated_obj = action_object_mapping[action_label]
            # find if correct object label is in object recognition data
            obj_data = find_recognised_objects(obj2[obj_counter:obj_counter+object_detect_num], associated_obj)
            # else retrieve first object detected
            object_label = obj_data[0]
            object_coordinates = obj_data[1:]
            # update obj_counter
            obj_counter += object_detect_num
        frame_data = [action_label] + [object_label] + object_coordinates + joint_coordinates
        dataset[pose_vid_name].append(frame_data)
    return dataset

def clean_dataset_action(dataset):
    """
    Removes video frames with no action label
    """
    new_dataset = dict()
    for video_name in dataset.keys():
        for i in range(len(dataset[video_name])):
            if dataset[video_name][i][0].lower() != "none":
                try:
                    new_dataset[video_name]
                except Exception as e:
                    new_dataset[video_name] = []
                new_dataset[video_name].append(dataset[video_name][i])
    return new_dataset

def clean_dataset_action_object(dataset):
    """
    Removes video frames with no action and object labels
    """
    new_dataset = dict()
    for video_name in dataset.keys():
        for i in range(len(dataset[video_name])):
            if dataset[video_name][i][0].lower() != "none" and dataset[video_name][i][1].lower() != "none":
                try:
                    new_dataset[video_name]
                except Exception as e:
                    new_dataset[video_name] = []
                new_dataset[video_name].append(dataset[video_name][i])
    return new_dataset


if __name__ == "__main__":
    print("GETTING RAW DATA")
    train_obj1 = get_data(path_train_obj1)
    train_obj2 = get_data(path_train_obj2)
    train_pose1 = get_data(path_train_pose1)
    train_pose2 = get_data(path_train_pose2)
    val_obj1 = get_data(path_val_obj1)
    val_obj2 = get_data(path_val_obj2)
    val_pose1 = get_data(path_val_pose1)
    val_pose2 = get_data(path_val_pose2)
    action_object_mapping = get_data(action_object_path)

    print("GENERATING DATASET")
    val_unclean_dataset = generate_dataset(val_obj1, val_obj2, val_pose1, val_pose2, action_object_mapping)
    train_unclean_dataset = generate_dataset(train_obj1, train_obj2, train_pose1, train_pose2, action_object_mapping)
    val_unclean_path = "D:\\Users\\SMH\\Tim_v2\\data\\validate_dataset_unclean_yolo.json"
    train_unclean_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_dataset_unclean_yolo.json"

    print("STORING UNCLEAN DATASET")
    with open(val_unclean_path, "w") as outfile:
        json.dump(val_unclean_dataset, outfile)
    with open(train_unclean_path, "w") as outfile:
        json.dump(train_unclean_dataset, outfile)
    
    print("CLEANING VALIDATION DATASET")
    val_unclean_dataset = get_data(val_unclean_path)
    val_dataset_action_clean = clean_dataset_action(val_unclean_dataset)
    val_dataset_actionobject_clean = clean_dataset_action_object(val_unclean_dataset)

    print("SAVING CLEANED VALIDATION DATASET")
    val_dataset_action_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\val_dataset_action_clean_yolo.json"
    val_dataset_actionobject_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\val_dataset_actionobject_clean_yolo.json"
    with open(val_dataset_action_clean_path, "w") as outfile:
        json.dump(val_dataset_action_clean, outfile)
    with open(val_dataset_actionobject_clean_path, "w") as outfile:
        json.dump(val_dataset_actionobject_clean, outfile)

    print("CLEANING TRAIN DATASET")
    train_dataset = get_data(train_unclean_path)
    train_dataset_action_clean = clean_dataset_action(train_dataset)
    train_dataset_actionobject_clean = clean_dataset_action_object(train_dataset)

    print("SAVING CLEANED TRAIN DATASET")
    train_dataset_action_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_dataset_action_clean_yolo.json"
    train_dataset_actionobject_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_dataset_actionobject_clean_yolo.json"
    with open(train_dataset_action_clean_path, "w") as outfile:
        json.dump(train_dataset_action_clean, outfile)
    with open(train_dataset_actionobject_clean_path, "w") as outfile:
        json.dump(train_dataset_actionobject_clean, outfile)

    # for video_name in val_dataset_clean.keys():
    #     print(video_name)
    #     print(len(val_dataset_clean[video_name]))
    #     # print(val_dataset_clean[video_name])

