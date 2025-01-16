#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File     :   convert_coco
@Date     :   2024/12/17 10:14:55
@Author   :   heshuang@tsingcloud.com
@Version  :   1.0
@Contact  :   heshuang@tsingcloud.com
@License  :   (C)Copyright 2024-2024, tsingcloud.com
@Desc     :   description
'''


from ultralytics.data.converter import convert_coco

if __name__ == '__main__':
    original_label_dir = '/home/double/Documents/datasets/coco/annotations'
    converted_label_dir = '/home/double/Documents/datasets/coco/labels'
    convert_coco(original_label_dir, converted_label_dir)