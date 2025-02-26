#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   bev_visualizer.py
@Time    :   2024/06/27 17:21:15
@Author  :   shuang.he
@Version :   1.0
@Contact :   shuang.he@momenta.ai
@License :   Copyright 2024, Momenta/shuang.he
@Desc    :   None
'''


import os
from typing import List, Union, Tuple

import cv2
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.transform import Rotation


class Point:
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"


class PointTools:
    def __init__(self):
        pass

    def point_in_polygon(self, poly_points: List[Point], target_point: Point) -> bool:
        x, y = target_point.x, target_point.y
        poly_sides = len(poly_points)
        j = poly_sides - 1
        inside_polygon = False
        for i in range(poly_sides):
            p1 = poly_points[i]
            p2 = poly_points[j]
            if (p1.y < y <= p2.y or p2.y < y <= p1.y) and (p1.x < x or p2.x < x):
                if p1.x + (y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x) < x:
                    inside_polygon = not inside_polygon
            j = i
        return inside_polygon

    def trans_matrix(self, yaw) -> np.ndarray:
        trans = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
        return trans

    def box_corners(self, box_pos: Point, box_size: Point, yaw: float) -> np.ndarray:
        """
        @param box_pos: Point(x, y)
        @param box_size: Point(length, width)
        @param yaw: yaw
        """
        x0 = np.array([[0.5 * box_size.x, 0.5 * box_size.y],
                       [0.5 * box_size.x, -0.5 * box_size.y],
                       [-0.5 * box_size.x, -0.5 * box_size.y],
                       [-0.5 * box_size.x, 0.5 * box_size.y]])
        R = self.trans_matrix(yaw)
        t = np.array([box_pos.x, box_pos.y])
        x_trans = np.matmul(R, x0.T).T + t
        return x_trans


class Visualizer:
    def __init__(self, img_width=2600, img_height=1000, img_scale=10):
        """ 最大显示范围：
        img_scale: 像素位置与真实位置的映射比例。在像素的实际显示位置: position * scale
        |x| < img_width  / img_scale / 2, default: 130m
        |y| < img_height / img_scale / 2, default: 50m
        """
        self.img_width = img_width
        self.img_height = img_height
        self.img_scale = img_scale 
        self.white_color = (255, 255, 255)[::-1]
        self.green_color = (0, 255, 0)[::-1]
        self.blue_color = (0, 0, 255)[::-1]
        self.red_color = (255, 0, 0)[::-1]
        self.yellow_color = (255, 255, 0)[::-1]
        
        self.ptool = PointTools()

    def color_decay(self, color, ratio):
        return [int(c * ratio) for c in color]

    def real_pos2img_pos(self, pos: np.ndarray) -> np.ndarray:
        """ convert real position to image position 
        @param pos: np.ndarray, n rows, 2 columns, [x, y]
        """
        img_pos = pos * self.img_scale
        img_pos[:, 1] *= -1 # 图像的y轴是顺时针方向为正，而车体坐标系y轴是逆时针为正，因此需要矫正
        img_pos += (self.img_width / 2, self.img_height / 2)
        return img_pos.astype(int)

    def draw_bbox3d(self, img: cv2.Mat, box: Union[List, Tuple], color, thickness=2, id=None, tag=None):
        """ draw box on image
        @param box: List or Tuple, [x, y, length, width, yaw]
        """
        x, y, length, width, yaw = box
        box_corner = self.ptool.box_corners(box_pos=Point(x, y), box_size=Point(length, width), yaw=yaw)
        box_corner = self.real_pos2img_pos(box_corner)
        cv2.polylines(img, [box_corner], isClosed=True, color=color, thickness=thickness)
        if id is not None:
            if type(id) == float:
                id = int(id)
            cv2.putText(img, str(id), box_corner[0, :], 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=thickness)
        if tag is not None:
            cv2.putText(img, str(tag), box_corner[2, :], 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.yellow_color, thickness=thickness)

    def draw_line(self, img: cv2.Mat, start: List, end: List, color, thickness=2, arrow=False):
        """ draw line on image
        start: [x, y]
        end: [x, y]
        """
        start_end = np.array([start, end])
        start_end = self.real_pos2img_pos(start_end)
        if not arrow:
            cv2.line(img, tuple(start_end[0]), tuple(start_end[1]), color, thickness)
        else:
            cv2.arrowedLine(img, tuple(start_end[0]), tuple(start_end[1]), color, thickness)

    def draw_circle(self, img, center: list, radius=60, color=(255, 255, 255), thickness=2):
        """
        center: [x, y], dtype('float64')
        """
        center = self.real_pos2img_pos(np.array(center, dtype=float).reshape(1, 2)).reshape(2)
        radius = int(radius * self.img_scale)
        cv2.circle(img, tuple(center), radius, color, thickness)

    def draw_point(self, img: cv2.Mat, point: List, color):
        """ draw point on image
        point: [x, y]
        """
        radius = 3 / self.img_scale
        self.draw_circle(img, point, radius, color)

    def draw_3d_bboxes(self, img, boxes: List[List], color, tags=None):
        """
        @params boxes: List[List(id, x, y, length, width, yaw)]
        """
        for i, box in enumerate(boxes):
            if pd.isna(box[1:]).any():
                continue
            tag = tags[i] if tags is not None else None

            self.draw_bbox3d(img, box[1:], color=color, id=box[0], tag=tag)

        return img
    
    def draw_3d_box_ego(self, img):
        self.draw_bbox3d(img, np.array([1, 0, 5, 2, 0]), color=self.color_decay(self.red_color, 1), id='ego')  # draw ego

    def draw_bbox2d(self, img, box, color, thickness, id=None):
        """ draw box on image
        @param box: List[left, top, right, bottom]
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        cv2.rectangle(img, np.array(box[:2], dtype=int), np.array(box[2:4], dtype=int), color, thickness)
        cv2.putText(img, str(id), org=np.array(box[:2], dtype=int), fontFace=font, fontScale=fontScale, 
                        color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_2d_boxes(self, img, boxes, cam_name, color, thickness=2):
        for id, box in boxes:
            if type(box) == np.ndarray:
                box = box.tolist()
            if type(box) != list or len(box) < 4:
                continue
            if pd.isna(box + [id]).any():
                continue
            self.draw_bbox2d(img, box, color, thickness, id)
        cv2.putText(img, cam_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)

    def save_img(self, img, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"saved img: {output_path}")

    def generate_blank_img_3d(self, background_color=(0, 0, 0)):
        img_3d = np.full((self.img_height, self.img_width, 3), background_color, dtype=np.uint8)
        return img_3d
    
    def rotate_img(self, img, angle=0.0):
        """
        @param img: np.ndarray, shape=(h, w, 3)
        @param angle: float, degree
        """
    
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rotated = cv2.warpAffine(img, M, (w, h))

        return img_rotated