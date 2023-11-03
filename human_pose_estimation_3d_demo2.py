#!/usr/bin/env python3
"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import sys
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from modules.inference_engine import InferenceEngine
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

import pickle
import math
import os
import glob
from hmmlearn.hmm import GMMHMM

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(poses_3d.shape[0]):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3] = np.dot(R_inv, pose_3d[0:3] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


# GLOBAL VARIABLES
# directory_path = "/home/smh/Videos/object_detection_dataset/videos_train"
directory_path = "D:\\Users\\SMH\\Tim_v2\\validate\\bouncing ball (not juggling)"
pose_3D_list = []
pose_2D_list = {}


def get_video_paths_from_directory(directory_path, video_extensions=('mp4', 'avi', 'mkv')):
    """Get a list of video file paths from a directory.

    Args:
        directory_path (str): The path to the directory containing video files.
        video_extensions (tuple, optional): Tuple of video file extensions to look for.
            Default is ('.mp4', '.avi', '.mkv').

    Returns:
        list: A list containing the paths of all video files found in the directory.
    """
    video_paths = []
    for path, subdirs, files in os.walk(directory_path):
        for filename in files:
            if filename.split('.')[-1] in video_extensions:
                video_paths.append(os.path.join(path, filename))
    return video_paths


if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.',
                            add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', default = "C:\\Users\\mwil0091\\open_model_zoo\\demos\\human_pose_estimation_3d_demo\\python\\public\\human-pose-estimation-3d-0001\\FP16\\human-pose-estimation-3d-0001.xml",
                      help='Required. Path to an .xml file with a trained model.',
                      type=Path, required=False)
    args.add_argument('-i', '--input', required=False, default = 0,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU or GPU. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU).',
                      type=str, default='CPU')
    args.add_argument('--height_size', help='Optional. Network input layer height size.', type=int, default=256)
    args.add_argument('--extrinsics_path',
                      help='Optional. Path to file with camera extrinsics.',
                      type=Path, default=None)
    args.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args.add_argument('--no_show', help='Optional. Do not display output.', action='store_true')
    args.add_argument("-u", "--utilization_monitors", default='', type=str,
                      help="Optional. List of monitors to show initially.")
    args = parser.parse_args()

    cap = open_images_capture(args.input, args.loop)

    stride = 8
    inference_engine = InferenceEngine(args.model, args.device, stride)
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    if not args.no_show:
        cv2.namedWindow(canvas_3d_window_name)
        cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = Path(__file__).parent / 'data/extrinsics.json'
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    is_video = cap.get_type() in ('VIDEO', 'CAMERA')

    base_height = args.height_size
    fx = args.fx

    frames_processed = 0
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    presenter = monitors.Presenter(args.utilization_monitors, 0)
    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    start_time = perf_counter()
    frame = cap.read()

    video_paths = get_video_paths_from_directory(directory_path)
    print(video_paths)
    #exit()
    for video_path in video_paths:
        video_name = video_path.split('/')[-1].split('.')[0]
        pose_2D_list[video_name] = []
        cap = cv2.VideoCapture(video_path)
        ret, _ = cap.read()
        list_vid = []
        frame_id = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        if ret:
            for len_video in range(length):
                ret, frame = cap.read()
                if frame is None:

                    break
                    raise RuntimeError("CaSn't read an image from the input")
                else:
                    current_time = cv2.getTickCount()
                    input_scale = base_height / frame.shape[0]
                    scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
                    if fx < 0:  # Focal length is unknown
                        fx = np.float32(0.8 * frame.shape[1])

                    inference_result = inference_engine.infer(scaled_img)
                    poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
                    edges = []
                    count = 0
                    if len(poses_3d) > 0:
                        poses_3d = rotate_poses(poses_3d, R, t)
                        poses_3d_copy = poses_3d.copy()
                        x = poses_3d_copy[:, 0::4]
                        y = poses_3d_copy[:, 1::4]
                        z = poses_3d_copy[:, 2::4]
                        poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
                        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                        testing_data = (poses_3d.tolist())

                        data_2D = (poses_2d.tolist())
                        state = "none"

                        # Adding the pose to list
                        # -----------------------------------------------------------------
                        # MODIFY TO INCLUDE FRAME ID AND LABEL
                        # pose_3D_list.append(poses_3d.tolist(), )
                        print(video_path.split('/')[-1])
                        pose_2D_list[video_name].append([poses_2d.tolist()[0], frame_id, video_path.split('/')[-1]])
                        frame_id += 1

                        for i in range(len(poses_2d)):
                            # frame = cv2.blur(frame, (10,10))
                            # If camera can see their nose
                            if poses_2d[i][3] > 0 and poses_2d[i][4] > 0:
                                y = int(poses_2d[i][4] - (poses_2d[i][1] - poses_2d[i][4]) - 0.1 * (
                                    poses_2d[i][1] - poses_2d[i][55]) - 0.1 * (
                                            poses_2d[i][1] - poses_2d[i][52]))
                                if y < 0:
                                    y = 0
                            else:  # Not facing camer
                                y = int(poses_2d[i][1] - (poses_2d[i][1] - poses_2d[i][55]) - (
                                    poses_2d[i][1] - poses_2d[i][52]))

                            # Finds the largest and smallest x value
                            x = 1920
                            w = 0
                            for j in range(3, 55, 3):
                                if x > int(poses_2d[i][j]) > 0:
                                    x = int(poses_2d[i][j])
                                if int(poses_2d[i][j]) > w and int(poses_2d[i][j]) > 0:
                                    w = int(poses_2d[i][j])
                            xs = x
                            x = int(x - (0.1 * (w - x)))
                            w = int((w - x) + (xs - x))
                            if x < 0:
                                x = 0
                            # print(str(x) + "," + str(w))
                            # Find the smallest y, cause the person might bend over
                            if y < 0:
                                y = int(poses_2d[i][1])
                            for j in range(4, 56, 3):
                                if y > int(poses_2d[i][j]) > 0 or y < 0:
                                    y = int(poses_2d[i][j])

                            # if left foot is out of frame, height is the height of window minus y,
                            # to cover the whole screen from top (y) to bottom
                            if poses_2d[i][25] > 0 and poses_2d[i][25] > poses_2d[i][43]:
                                h = int(1.1 * (poses_2d[i][25] - y))
                            elif poses_2d[i][43] > 0:
                                h = int(1.1 * (poses_2d[i][43] - y))
                            else:
                                h = 720 - y
                            if w <= 10:  # fail safe
                                w = 10
                            if y > 400:
                                y = 400
                                h = 720 - y
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # Blurs the contents of the rectangle
                            blurred_part = frame[y:y + h, x:x + w]
                            blurred_part = cv2.GaussianBlur(blurred_part, (47, 47), 50)
                            frame[y:y + blurred_part.shape[0],
                            x:x + blurred_part.shape[1]] = blurred_part  # Blurs frame
                            cv2.putText(frame, str(state), org=(x, y),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2,
                                        lineType=cv2.LINE_AA)  # Prints text

                        edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape(
                            (-1, 1, 1))).reshape(
                            (-1, 2))
                        count = count + 1
                        poses_str = str(count)
                        vel_str = ""

                        #for i in range(len(poses_3d)):
                            # send[0] - x, send[1] - y, send[2] - z
                            #send = poses_3d[i][9].copy()
                            #send[1] = transform_y(send[1], send[2])
                            #send[0] = transform_x(send[0])
                            #send[2] = 0

                            #poses_str = poses_str + ";" + str(send.tolist())

                            #vel_str = vel_str + ";" + str([send[0] - prev[0], send[1] - prev[0], 0])
                            #prev = [send[0], send[1], 0]

                        #poses_str = poses_str + vel_str
                        # sys.stdout.write(str(poses_3d))
                        # client.publish("/value", poses_str) #send data using mqtt ("/topic", data as string)
                    plotter.plot(canvas_3d, poses_3d, edges)
                    presenter.drawGraphs(frame)
                    draw_poses(frame, poses_2d)
                    metrics.update(start_time, frame)
                    frames_processed += 1
                    if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
                        video_writer.write(frame)

                    if not args.no_show:
                        cv2.imshow(canvas_3d_window_name, canvas_3d)
                        cv2.imshow('3D Human Pose Estimation', frame)

                        key = cv2.waitKey(delay)
                        if key == esc_code:
                            break

                        if key == p_code:
                            if delay == 1:
                                delay = 0
                            else:
                                delay = 1
                        else:
                            presenter.handleKey(key)
                        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
                            key = 0
                            while (key != p_code
                                   and key != esc_code
                                   and key != space_code):
                                plotter.plot(canvas_3d, poses_3d, edges)
                                cv2.imshow(canvas_3d_window_name, canvas_3d)
                                key = cv2.waitKey(33)
                            if key == esc_code:
                                break
                            else:
                                delay = 1
                    start_time = perf_counter()
                    # frame = cap.read()

                metrics.log_total()
                for rep in presenter.reportMeans():
                    log.info(rep)
                if cv2.waitKey(1) == ord('q'):
                    # client.loop_stop()  # stop the loop
                    break


        # with open("test3D.json", "w") as out_file:
        #     json.dump(pose_3D_list, out_file)
        # formatted_pose_2D_list = []
        # for frame_data in pose_2D_list:
        #     poses, frame_id, label = frame_data
        #     formatted_pose_2D_list.append([frame_id ,label, poses[0]])
        with open(os.path.join(directory_path,"test2D.json"), "w") as out_file:
            json.dump(pose_2D_list, out_file)
