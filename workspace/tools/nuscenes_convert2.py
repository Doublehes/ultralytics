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


import numpy as np
import pickle
import os
import random
from tqdm import tqdm
import shutil
import cv2



def show(img_path, labels):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]


    # Show boxes.
    for label in labels:
        xy = np.array([label[1], label[2]]) * np.array([img_w, img_h])
        wh = np.array([label[3], label[4]]) * np.array([img_w, img_h])
        lt = xy - wh / 2
        rb = xy + wh / 2
        cv2.rectangle(img, (int(lt[0]), int(lt[1])), (int(rb[0]), int(rb[1])), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(-1)


def get_car_3d_info(bbox_3d):
    """
    Coordinates in Camera:

    .. code-block:: none

                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1. The yaw is 0 at
    the positive direction of x axis, and decreases from the positive direction
    of x to the positive direction of z.

    @param bbox_3d: [x, y, z, l, w, h, yaw] in camera coordinate
    """
    x, y, z, l, w, h, yaw = bbox_3d
    
    cx, cy, cz = z, -x, -y
    
    cyaw = -yaw - np.pi / 2

    return (cx, cy, cz, w, h, l, cyaw)



def get_labels(cam_ins):
    img_w, img_h = 1600, 900
    label_info = []
    for instance in cam_ins:
        label = instance["bbox_label"]
        # nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        #                 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
        #                 'barrier')
        if label not in [0, 1, 2, 3, 4, 5, 6, 7]:
            continue
        if not instance["bbox_3d_isvalid"]:
            continue
        l, t, r, b = instance["bbox"]
        fx = lambda x: max(0, min(x, img_w))
        fy = lambda y: max(0, min(y, img_h))
        l, t, r, b = fx(l), fy(t), fx(r), fy(b)
        x, y, w, h = (l+r)/2, (t+b)/2, r-l, b-t
        x, y, w, h = x/img_w, y/img_h, w/img_w, h/img_h
        cx, cy, cz, cw, ch, cl, cyaw = get_car_3d_info(instance["bbox_3d"])
        label_info.append([label, x, y, w, h, cx, cy, cz, cw, ch, cl, cyaw])
        
    return label_info

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



def main(pkl_path, save_dir, sample_num=100, only_show=True, only_label_2d=True):

    root_path = "/media/double/Data1/datasets/nuScenese"
    cam = "CAM_FRONT"
    save_dir = os.path.join(root_path, save_dir)
    img_dir = os.path.join(save_dir, "images")
    label_dir = os.path.join(save_dir, "labels")
    if os.path.exists(save_dir):
        print(f"警告: 输出目录 {save_dir} 已存在，将覆盖其中的内容。")
        os.system(f"rm -rf {save_dir}")

    os.makedirs(img_dir)
    os.makedirs(label_dir)

    with open(pkl_path, "rb") as f:
        train_infos = pickle.load(f)
    metainfo = train_infos["metainfo"]
    datalist = train_infos["data_list"]
    print(metainfo)
    print(f"============ sample num: {len(datalist)} ============")

    datalist = random.sample(datalist, sample_num)
    process_bar = tqdm(total=len(datalist))
    for i, data in enumerate(datalist):
        img_path = data["images"][cam]["img_path"]
        img_path = os.path.join(root_path, "samples", cam, img_path)
        cam_ins = data["cam_instances"][cam]

        label_info = get_labels(cam_ins)
        label_info = filter(label_info)
        if only_show:
            show(img_path, label_info)
            continue

        img_new_path = os.path.join(img_dir, f"{i}.jpg")
        shutil.copy(img_path, img_new_path)

        label_path = os.path.join(label_dir, f"{i}.txt")
        with open(label_path, "w") as f:
            for label in label_info:
                if only_label_2d:
                    label = label[:5]
                f.write(" ".join([str(x) for x in label]) + "\n")
        
        process_bar.update(1)

if __name__ == '__main__':
    only_show = True
    train_pkl = "/media/double/Data1/datasets/nuScenese/nuscenes_infos_train.pkl"
    train_dir = "yolo_dataset/train3d_new"
    main(train_pkl, train_dir, 10000, only_show=only_show, only_label_2d=False)

    # val_pkl = "/media/double/Data/datasets/nuScenese/nuscenes_infos_val.pkl"
    # val_dir = "yolo_dataset/val3d"
    # main(val_pkl, val_dir, 500, only_show=only_show, only_label_2d=False)
    pass

