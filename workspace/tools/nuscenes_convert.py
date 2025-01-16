#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File     :   nuscenes_convert
@Date     :   2025/01/13 09:40:42
@Author   :   heshuang@tsingcloud.com
@Version  :   1.0
@Contact  :   heshuang@tsingcloud.com
@License  :   (C)Copyright 2025-2025, tsingcloud.com
@Desc     :   description
'''


from nuscenes.nuscenes import NuScenes, BoxVisibility
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import os
from tqdm import tqdm
import random


def show(img_path, labels):
    img_h, img_w = 900, 1600

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(Image.open(img_path))

    # Show boxes.
    for label in labels:
        xy = np.array([label[1], label[2]]) * np.array([img_w, img_h])
        wh = np.array([label[3], label[4]]) * np.array([img_w, img_h])
        lt = xy - wh / 2
        rb = xy + wh / 2
        plt.plot([lt[0], lt[0], rb[0], rb[0], lt[0]],
                 [lt[1], rb[1], rb[1], lt[1], lt[1]], 'r-')

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    plt.show()
    
    # nusc.render_sample_data(cam_front_data['token'], box_vis_level=BoxVisibility.ALL)

def calc_iou2d(bbox1, bbox2):
    x11, y11, x12, y12 = np.split(bbox1, 4, axis=-1)
    x21, y21, x22, y22 = np.split(bbox2, 4, axis=-1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    i_width = np.maximum((xB - xA), 0)
    i_height = np.maximum((yB - yA), 0)
    inter = i_width * i_height
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    union = boxAArea + np.transpose(boxBArea) - inter
    iou = inter / (1e-7 + union)
    
    return iou

def calc_coverage(bbox1, bbox2):
    x11, y11, x12, y12 = np.split(bbox1, 4, axis=-1)
    x21, y21, x22, y22 = np.split(bbox2, 4, axis=-1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    i_width = np.maximum((xB - xA), 0)
    i_height = np.maximum((yB - yA), 0)
    inter = i_width * i_height
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    min_area = np.minimum(boxAArea, np.transpose(boxBArea))
    coverage = inter / (1e-7 + min_area)
    
    return coverage, boxAArea, boxBArea

def filter(labels: list):
    """
    @pram labels: list of labels, each label is a list of [class_id, x, y, w, h, cx, cy, cz, cw, ch, cl, cyaw]
    """
    if len(labels) == 0:
        return []
    labels = np.array(labels)
    dist = np.linalg.norm(labels[:, 5:7], axis=1)
    labels = np.concatenate((labels, dist.reshape(-1, 1)), axis=1).tolist()
    labels = [x for x in labels if x[-1] < 60]
    if len(labels) == 0:
        return []
    labels.sort(key=lambda x: x[-1])
    box = np.array(labels).copy()[:, 1:5]
    left = box[:, 0] - box[:, 2] / 2
    right = box[:, 0] + box[:, 2] / 2
    top = box[:, 1] - box[:, 3] / 2
    bottom = box[:, 1] + box[:, 3] / 2
    box = np.stack([left, top, right, bottom], axis=1)
    coverage, area, _ = calc_coverage(box, box)

    selected = []
    passed = []
    for i in range(len(labels)):
        if i in passed:
            continue
        passed.append(i)
        selected.append(labels[i])
        for j in range(i + 1, len(labels)):
            if j in passed:
                continue
            if coverage[i, j] > 0.5 and area[j] < area[i]:
                passed.append(j)

    return selected


def get_labels(boxes, camera_intrinsic, cs_record):
    img_h, img_w = 900, 1600
    label_info = list()
    for box in boxes:
        if box.name.split('.')[0] != 'vehicle':
            continue
        if box.name.split('.')[1] not in ['car', 'truck', 'bus']:
            continue
        
        label = [0] # class id

        ############## 2D info ##############
        corners = np.matmul(camera_intrinsic, box.corners())
        corners = corners / corners[2, :]
        lt = np.min(corners[:2, :], axis=1)
        rb = np.max(corners[:2, :], axis=1)

        lt[lt < 0] = 0
        if rb[0] > 1600:
            rb[0] = 1600
        if rb[1] > 900:
            rb[1] = 900

        center = (lt + rb) / 2 # x, y
        size = rb - lt # w, h
        center[0] = center[0] / img_w
        center[1] = center[1] / img_h
        size[0] = size[0] / img_w
        size[1] = size[1] / img_h
        if np.min(center) < 0 or np.max(center) > 1 or np.min(size) < 0 or np.max(size) > 1:
            print(f"Warning: box is out of image: lt:{lt}, rb:{rb}, center:{center}, size:{size}")
            continue
        label.extend(center.tolist())
        label.extend(size.tolist())
        
        ############## 3D info ##############
        # Move box to ego vehicle coord system.
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))        
        label.extend(box.center.tolist())
        label.extend(box.wlh.tolist())
        label.append(box.orientation.degrees)

        label_info.append(label)

    return label_info


def main(nusc, samples, save_dir):
    only_show = True
    label_2d = True
    img_dir = os.path.join(save_dir, "images")
    label_dir = os.path.join(save_dir, "labels")
    if os.path.exists(save_dir):
        print(f"警告: 输出目录 {save_dir} 已存在，将覆盖其中的内容。")
        os.system(f"rm -rf {save_dir}")

    os.makedirs(img_dir)
    os.makedirs(label_dir)

    process_bar = tqdm(total=len(samples))

    for i, my_sample in enumerate(samples):
        sensor = 'CAM_FRONT'
        cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
        print(i, cam_front_data)

        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_data['token'])
        cs_record = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        if len(boxes) == 0:
            print(f"Warning: no boxes found in sample {i}")
            process_bar.update(1)
            continue

        label_info = get_labels(boxes, camera_intrinsic, cs_record)
        label_info = filter(label_info)
        if only_show and i % 5 == 0:
            show(data_path, label_info)
            continue

        img_path = os.path.join(img_dir, f"{i}.jpg")
        os.system(f"cp {data_path} {img_path}")

        label_path = os.path.join(label_dir, f"{i}.txt")
        with open(label_path, "w") as f:
            for label in label_info:
                if label_2d:
                    label = label[:5]
                f.write(" ".join([str(x) for x in label]) + "\n")
        
        process_bar.update(1)

if __name__ == '__main__':
    nusc_tool = NuScenes(version='v1.0-trainval', dataroot='/media/double/Data/datasets/nuScenese', verbose=True)
    nusc_tool.list_scenes()

    random.shuffle(nusc_tool.sample)
    harf = len(nusc_tool.sample) // 2

    train_num, val_num = 2000, 500
    train_samples = random.sample(nusc_tool.sample[:harf], train_num)
    val_samples = random.sample(nusc_tool.sample[harf:], val_num)

    save_dir_train = "/media/double/Data/datasets/nuScenese/yolo_dataset/train"
    save_dir_val = "/media/double/Data/datasets/nuScenese/yolo_dataset/val"
    main(nusc_tool, train_samples, save_dir_train)
    main(nusc_tool, val_samples, save_dir_val)

