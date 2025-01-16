#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File     :   generate_shape_dataset
@Date     :   2024/12/18 15:19:57
@Author   :   heshuang@tsingcloud.com
@Version  :   1.0
@Contact  :   heshuang@tsingcloud.com
@License  :   (C)Copyright 2024-2024, tsingcloud.com
@Desc     :   description
'''


import os
import cv2
import numpy as np
import json
import math
from tqdm import tqdm


def general_noise_image(height=224, width=224, channels=3, mean=200, stddev=20):
    # 生成高斯噪声
    mean = 200  # 高斯分布的均值
    stddev = 20  # 高斯分布的标准差
    noise = np.random.normal(mean, stddev, (height, width, channels))

    return noise.astype(np.uint8)

def generate_image(height=224, width=224, channels=3, fill_value=255):
    return np.full((height, width, channels), fill_value, np.uint8)


def generate_random_point(image):
    height, width = image.shape[:2]

    # 生成随机点的坐标
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)

    return x, y


# 定义一个函数来计算两点之间的距离
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# 定义一个函数来计算三角形的角
def triangle_angles(pointA, pointB, pointC):
    # 计算三边长度
    AB = distance(pointA, pointB)
    BC = distance(pointB, pointC)
    CA = distance(pointC, pointA)

    # 使用余弦定理计算各角的余弦值
    try:
        cosA = (BC**2 + CA**2 - AB**2) / (2 * BC * CA)
        cosB = (CA**2 + AB**2 - BC**2) / (2 * CA * AB)
        cosC = (AB**2 + BC**2 - CA**2) / (2 * AB * BC)
    except:
        # import pudb;pudb.set_trace()
        print("calc angles error:", pointA, pointB, pointC)
        return 0, 0, 0

    # 容错处理，确保余弦值在 [-1, 1] 范围内
    cosA = max(-1, min(1, cosA))
    cosB = max(-1, min(1, cosB))
    cosC = max(-1, min(1, cosC))

    # 使用反余弦函数来得到角度（以度为单位）
    angleA = math.acos(cosA) * (180 / math.pi)
    angleB = math.acos(cosB) * (180 / math.pi)
    angleC = math.acos(cosC) * (180 / math.pi)
  
    return angleA, angleB, angleC


def triangle_area(point1, point2, point3):
    return 0.5 * abs(point1[0] * (point2[1] - point3[1]) +
                     point2[0] * (point3[1] - point1[1]) +
                     point3[0] * (point1[1] - point2[1]))

def draw_triangle(image, color=(0, 255, 0)):
    while True:
        # 生成随机三角形的三个顶点
        triangle_points = []
        for _ in range(3):
            triangle_points.append(generate_random_point(image))
        # area = triangle_area(triangle_points[0], triangle_points[1], triangle_points[2])

        angles = triangle_angles(triangle_points[0], triangle_points[1], triangle_points[2])

        if all([angle > 30 for angle in angles]):
            break

    triangle_points = np.array(triangle_points)

    overlay = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(overlay, [triangle_points], color)
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1, 0)
    # cv2.fillPoly(image, [triangle_points], color)

    left_top = np.min(triangle_points, axis=0).tolist()
    right_bottom = np.max(triangle_points, axis=0).tolist()

    # 计算中心点坐标
    center_x = (left_top[0] + right_bottom[0]) // 2
    center_y = (left_top[1] + right_bottom[1]) // 2
    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    annotation = {
        "category": "triangle",
        "bbox": [left_top[0], left_top[1], right_bottom[0], right_bottom[1]],
        "bbox_xywh": [center_x, center_y, width, height],
    }

    return annotation, image

def draw_circle(image, color=(0, 0, 200)):
    # 生成随机圆的圆心和半径
    while True:
        x, y = generate_random_point(image)
        radius = np.random.randint(10, 75)
        if x - radius >= 0 and y - radius >= 0 and x + radius < image.shape[1] and y + radius < image.shape[0]:
            break
    
    overlay = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(overlay, (x, y), radius, color, -1)  # 使用红色填充圆
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1, 0)
    
    # cv2.circle(image, (x, y), radius, color, -1)  # 使用红色填充圆

    left_top = (x - radius, y - radius)
    right_bottom = (x + radius, y + radius)

    annotation = {
        "category": "circle",
        "bbox": [left_top[0], left_top[1], right_bottom[0], right_bottom[1]],
        "bbox_xywh": [x, y, radius*2, radius*2],
    }

    return annotation, image

def draw_square(image, color=(200, 0, 0)):
    
    while True:
        center = generate_random_point(image)
        length = np.random.randint(10, 150)
        if center[0] - length // 2 >= 0 and center[1] - length // 2 >= 0 and center[0] + length // 2 < image.shape[1] \
            and center[1] + length // 2 < image.shape[0]:
            break

    x1, y1 = center[0] - length // 2, center[1] - length // 2
    x2, y2 = center[0] + length // 2, center[1] + length // 2

    overlay = np.zeros_like(image, dtype=np.uint8)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # 使用蓝色填充矩形
    alpha = 0.5
    image = cv2.addWeighted(overlay, alpha, image, 1, 0)
    # cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # 使用蓝色填充矩形

    left_top = (min(x1, x2), min(y1, y2))
    right_bottom = (max(x1, x2), max(y1, y2))

    annotation = {
        "category": "square",
        "bbox": [left_top[0], left_top[1], right_bottom[0], right_bottom[1]],
        "bbox_xywh": [center[0], center[1], length, length],
    }
    return annotation, image


def calc_iou(box1, box2):
    """
    计算两个矩形框的交并比
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: 交并比
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集面积
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集面积
    union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area

    return inter_area / union_area


def calc_coverage(box1, box2):
    """
    计算两个矩形框的覆盖率
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: 覆盖率
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集面积
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)


    return inter_area / min(area1, area2)


def save_annos_yolo(annos, width, height, save_path):
    """
    将标注信息保存为YOLO格式的txt文件
    :param annos: 标注信息列表，每个元素为一个字典，包含类别、坐标等信息
    :param width: 图像宽度
    :param height: 图像高度
    :param save_path: 保存路径
    """
    name_to_id = {
        "triangle": 0,
        "circle": 1,
        "square": 2
    }
    with open(save_path, "w") as f:
        for anno in annos:
            category_id = name_to_id[anno["category"]]
            x, y, w, h = anno["bbox_xywh"]
            x /= width
            y /= height
            w /= width
            h /= height
            f.write(f"{category_id} {x} {y} {w} {h}\n")


if __name__ == "__main__":
    train = False
    generate_num = 1000
    img_id = 2001

    save_dir = "/home/double/Documents/datasets/shape/" + ("train" if train else "val")
    img_dir = os.path.join(save_dir, "images")
    anno_dir = os.path.join(save_dir, "annotations")
    anno_yolo_dir = os.path.join(save_dir, "labels")
    remove_dir = True
    if os.path.exists(save_dir) and remove_dir:
        print(f"remove {save_dir}")
        os.system(f"rm -rf {save_dir}")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(anno_yolo_dir, exist_ok=True)

    height, width = 320, 320

    process_bar = tqdm(total=generate_num)

    count = 0
    while True:
        img = generate_image(height=height, width=width, channels=3, fill_value=np.random.randint(50, 200))
        noise = general_noise_image(height=height, width=width, mean=0, stddev=np.random.randint(50, 200))

        draw_funcs = [draw_triangle, draw_circle, draw_square]

        annos = []
        for i in range(np.random.randint(1, 6)):
            draw_func = np.random.choice(draw_funcs)
            anno, img = draw_func(img, color=np.random.randint(0, 256, 3).tolist())
            annos.append(anno)

        
        coverages = []
        for i in range(len(annos)):
            for j in range(i+1, len(annos)):
                coverage = calc_coverage(annos[i]["bbox"], annos[j]["bbox"])
                coverages.append(coverage)
        
        if any([coverage > 0.1 for coverage in coverages]):
            # print(img_id, coverages)
            continue
            pass

        img = cv2.blur(img, (5, 5))
        img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)

        img_name = f"{img_id}.jpg"
        annotations = {
            "img_name": img_name,
            "width": width,
            "height": height,
            "instances": annos
        }
        
        cv2.imwrite(os.path.join(img_dir, f"{img_id}.jpg"), img)
        with open(os.path.join(anno_dir, f"{img_id}.json"), "w") as f:
            json.dump(annotations, f, indent=4)
        save_annos_yolo(annos, width, height, os.path.join(anno_yolo_dir, f"{img_id}.txt"))

        img_id += 1
        count += 1
        process_bar.update(1)

        if count >= generate_num:
            break

