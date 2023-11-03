import json
import os
import numpy as np

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def _get_train_clips(seed: int, num_clips: int, num_frames: int, clip_len: int) -> np.ndarray:
    """Uniformly sample indices for training clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.

    Returns:
        np.ndarray: The sampled indices for training clips.
    """
    np.random.seed(seed)
    all_inds = []
    for clip_idx in range(num_clips):
        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int32)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        all_inds.append(inds)

    return np.concatenate(all_inds)

def _get_test_clips(seed: int, num_clips: int, num_frames: int, clip_len: int) -> np.ndarray:
    """Uniformly sample indices for testing clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.

    Returns:
        np.ndarray: The sampled indices for testing clips.
    """

    np.random.seed(seed)
    all_inds = []
    for i in range(num_clips):
        if num_frames < clip_len:
            start_ind = i if num_frames < num_clips \
                else i * num_frames // num_clips
            inds = np.arange(start_ind, start_ind + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        all_inds.append(inds)

    return np.concatenate(all_inds)

def filter_data(seed: int, num_clips: int, clip_len: int, dataset: dict):
    new_dataset = dict()
    for video_name in dataset.keys():
        vid_frames = dataset[video_name]
        num_frames = len(vid_frames)
        # check if video has enough frames
        if num_frames < clip_len:
            continue
        new_dataset[video_name] = []
        inds = _get_train_clips(seed=seed, num_clips=num_clips, num_frames=num_frames, clip_len=clip_len)
        lower = 0
        upper = clip_len
        for i in range(num_clips):
            new_vid_frames = []
            for frame_id in inds[lower:upper]:
                new_vid_frames.append(vid_frames[frame_id])
            new_dataset[video_name].append(new_vid_frames)
            upper += clip_len
            lower += clip_len
    return new_dataset

def get_data_types(dataset: dict):
    data_types = dict()
    for vid_name in dataset.keys():
        for clip in dataset[vid_name]:
            for frame in clip:
                try:
                    data_types[len(frame)]
                except Exception as e:
                    data_types[len(frame)] = 0
                data_types[len(frame)] += 1
    return data_types

def output_data_types(data_types):
    for key in data_types.keys():
        if key == 60:
            print(f"frames with pose estimate only: {data_types[key]}")
        else:
            print(f"frames with pose estimate + object detection: {data_types[key]}")

if __name__ == "__main__":
    val_dataset_action_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\val_dataset_actionobject_clean_yolo.json"
    train_dataset_action_clean_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_dataset_actionobject_clean_yolo.json"
    val_filtered_dataset_path = "D:\\Users\\SMH\\Tim_v2\\data\\val_dataset_actionobject_filtered_yolo.json"
    train_filtered_dataset_path = "D:\\Users\\SMH\\Tim_v2\\data\\train_datase_actionobject_filtered_yolo.json"
    # generator seed
    seed = 1
    # number of test clips we want to create
    num_clips = 2
    # output number of frames we want
    clip_len = 10

    # filter validate dataset
    print("filtering validate dataset")
    val_dataset_clean = get_data(val_dataset_action_clean_path)
    val_filtered = filter_data(seed, num_clips, clip_len, val_dataset_clean)
    # output results
    print("outputting results")
    val_data_types = get_data_types(val_filtered)
    output_data_types(val_data_types)
    # save filtered validate dataset
    print("saving filtered validation results")
    save_data(val_filtered_dataset_path, val_filtered)

    # filter train dataset
    print("filtering train dataset")
    train_dataset_clean = get_data(train_dataset_action_clean_path)
    train_filtered = filter_data(seed, num_clips, clip_len, train_dataset_clean)
    # output results
    print("outputting results")
    train_data_types = get_data_types(train_filtered)
    output_data_types(train_data_types)
    # save filtered train dataset
    print("saving filtered train results")
    save_data(train_filtered_dataset_path, train_filtered)

    # try to access the data
    print("accessing data")
    val_filtered = get_data(val_filtered_dataset_path)
    train_filtered = get_data(train_filtered_dataset_path)
    val_data_types = get_data_types(val_filtered)
    output_data_types(val_data_types)
    train_data_types = get_data_types(train_filtered)
    output_data_types(train_data_types)

    print("printing val data")
    for video_name in val_filtered.keys():
        print(video_name)
        print(len(val_filtered[video_name]))
        for clip in val_filtered[video_name]:
            print(len(clip) == clip_len)
    
    print("printing train data")
    for video_name in train_filtered.keys():
        print(video_name)
        print(len(train_filtered[video_name]))
        for clip in train_filtered[video_name]:
            print(len(clip) == clip_len)
    print("SUCCESSFUL")


# {video_name1: [clip_1, clip_2], video_name2: [clip_1, clip_2], ...}
# clip_1 = [frame_1, frame_2, ...]
# frame_1 = [action_label, object_label, object_coordinates, joint_coordinates]