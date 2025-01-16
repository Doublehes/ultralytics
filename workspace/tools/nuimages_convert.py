#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File     :   nuimages_convert
@Date     :   2024/12/22 19:21:59
@Author   :   heshuang@tsingcloud.com
@Version  :   1.0
@Contact  :   heshuang@tsingcloud.com
@License  :   (C)Copyright 2024-2024, tsingcloud.com
@Desc     :   description
'''


from nuimages import NuImages

import os
import cv2
from nuimages.nuimages import NuImages
from tqdm import tqdm
import random


def convert_to_yolo(image_width, image_height, bbox):
    """
    将边界框转换为 YOLO 格式：归一化 [x_center, y_center, width, height]
    """
    l, t, r, b = bbox
    x_center = (l + r) / 2 / image_width
    y_center = (t + b) / 2 / image_height
    norm_width = (r - l) / image_width
    norm_height = (b - t) / image_height
    return x_center, y_center, norm_width, norm_height

if __name__ == '__main__':
    version = 'v1.0-train'
    num = 5000
    output_path = f'/media/double/Data/datasets/nuImages/yolo_dataset_{version.split("-")[-1]}'
    dataroot = '/media/double/Data/datasets/nuImages'


    # 创建输出目录
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    if os.path.exists(output_path):
        print(f"警告: 输出目录 {output_path} 已存在，将覆盖其中的内容。")
        os.system(f"rm -rf {output_path}")

    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    nuim = NuImages(version=version, dataroot=dataroot, verbose=True)
    for cat in nuim.category:
        print(cat["name"])
    category_map = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "bicycle": 3,
        "motorcycle": 4,
        "other_vehicle": 5,
        "pedestrian": 6,
    }

    for i, sample in enumerate(random.sample(nuim.sample, num)):
        # 打印进度
        if i % 100 == 0:
            print(f""">>> 进度: {i}/{num}""")

        # 渲染并保存图像
        # nuim.render_image(sample["key_camera_token"], out_path=f'./outputs/{i}.jpg', annotation_type="objects",
        #                   with_category=True, box_line_width=-1, render_scale=5)

        sample_data = nuim.get('sample_data', sample['key_camera_token'])
        # img_width = sample_data["width"]
        # img_height = sample_data["height"]
        # print(sample_data)

        # 图像文件路径
        image_path = os.path.join(dataroot, sample_data["filename"])
        image_output_path = os.path.join(images_dir, os.path.basename(sample_data["filename"]))
        label_output_path = os.path.join(labels_dir, os.path.basename(sample_data["filename"]).replace('.jpg', '.txt'))

        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法加载图像 {image_path}")
            continue
        img_height, img_width = img.shape[:2]
        cv2.imwrite(image_output_path, img)  # 复制图像到输出目录

        object_tokens, surface_tokens = nuim.list_anns(sample['token'], verbose=False)
        # 转换并保存 YOLO 格式的标注
        yolo_annotations = []
        for obj_token in object_tokens:
            obj = nuim.get('object_ann', obj_token)
            bbox = obj["bbox"]
            category = nuim.get('category', obj["category_token"])
            category_name = category["name"]
            category_first = category_name.split(".")[0]
            if category_first not in ["vehicle", "human"]:
                continue
            category_second = category_name.split(".")[1]
            if category_second not in ["car", "truck", "bus", "bicycle", "motorcycle", "pedestrian"]:
                category_second = "other_vehicle"
            x_center, y_center, norm_width, norm_height = convert_to_yolo(img_width, img_height, bbox)
            class_id = category_map[category_second]
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        with open(label_output_path, 'w') as f:
            f.write("\n".join(yolo_annotations))
        


        