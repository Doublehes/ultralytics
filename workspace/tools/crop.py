"""
从高分辨率图像中随机截取包含至少一个检测框的子图，并生成对应的YOLO标签
"""

import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict


def visualize_cropped_detections(
    cropped_img: np.ndarray,
    cropped_labels: List[List[float]],
    class_names: List[str] = None,
    show: bool = True,
    save_path: str = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),  # 绿色框
    text_color: Tuple[int, int, int] = (255, 255, 255),  # 白色文字
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    可视化截取后的子图和对应的YOLO格式标签
    :param cropped_img: 截取后的子图 (np.ndarray)
    :param cropped_labels: YOLO格式标签列表，每个元素为 [class_id, x_center, y_center, width, height]
    :param class_names: 类别名称列表（可选），如["car", "person"]，若为None则显示类别ID
    :param show: 是否显示图像
    :param save_path: 可视化图像保存路径（可选），如"vis_crop.jpg"
    :param box_color: 检测框颜色 (B, G, R)
    :param text_color: 文字颜色 (B, G, R)
    :param line_thickness: 检测框线宽
    :param font_scale: 文字大小缩放比例
    :return: 绘制了检测框的可视化图像
    """
    # 复制图像避免修改原数据
    vis_img = cropped_img.copy()
    h, w = vis_img.shape[:2]
    
    # 遍历所有标签并绘制
    for label in cropped_labels:
        # 解析YOLO标签
        class_id, x_c, y_c, box_w, box_h = label
        
        # 将YOLO归一化坐标转换为像素坐标 (x1, y1, x2, y2)
        x1 = int((x_c - box_w / 2) * w)
        y1 = int((y_c - box_h / 2) * h)
        x2 = int((x_c + box_w / 2) * w)
        y2 = int((y_c + box_h / 2) * h)
        
        # 绘制检测框
        cv2.rectangle(
            vis_img,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness=line_thickness
        )
        
        # 准备文字内容（类别名称/ID + 坐标）
        if class_names and int(class_id) < len(class_names):
            class_text = class_names[int(class_id)]
        else:
            class_text = f"Class {int(class_id)}"
        
        # 绘制文字背景（提升可读性）
        text = f"{class_text}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        text_w, text_h = text_size
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_h - 5
        text_bg_x2 = x1 + text_w
        text_bg_y2 = y1
        # 确保文字背景不超出图像边界
        text_bg_y1 = max(text_bg_y1, 0)
        
        cv2.rectangle(
            vis_img,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            box_color,
            thickness=-1  # 填充背景
        )
        
        # 绘制文字
        cv2.putText(
            vis_img,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness=line_thickness
        )
    
    # 显示图像
    if show:
        cv2.namedWindow("Cropped Image with Labels", cv2.WINDOW_NORMAL)
        cv2.imshow("Cropped Image with Labels", vis_img)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()
    
    # 保存图像
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"可视化图像已保存至: {save_path}")
    
    return vis_img

def load_yolo_labels(label_path: str) -> List[List[float]]:
    """
    加载YOLO格式的标签文件
    :param label_path: 标签文件路径 (.txt)
    :return: 标签列表，每个元素为 [class_id, x_center, y_center, width, height] (归一化坐标)
    """
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) != 5:
                continue
            labels.append(parts)
    return labels

def convert_yolo_to_abs(
    yolo_box: List[float], 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float, float]:
    """
    将YOLO归一化坐标转换为绝对像素坐标 (x1, y1, x2, y2, class_id)
    :param yolo_box: [class_id, x_center, y_center, w, h]
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: (x1, y1, x2, y2, class_id)
    """
    class_id, x_c, y_c, w, h = yolo_box
    x1 = (x_c - w/2) * img_width
    y1 = (y_c - h/2) * img_height
    x2 = (x_c + w/2) * img_width
    y2 = (y_c + h/2) * img_height
    return x1, y1, x2, y2, class_id

def convert_abs_to_yolo(
    abs_box: Tuple[float, float, float, float], 
    sub_img_width: int, 
    sub_img_height: int
) -> Tuple[float, float, float, float]:
    """
    将绝对像素坐标转换为子图的YOLO归一化坐标 (x_center, y_center, w, h)
    :param abs_box: (x1, y1, x2, y2) (相对于子图的绝对坐标)
    :param sub_img_width: 子图宽度
    :param sub_img_height: 子图高度
    :return: (x_center, y_center, w, h) (归一化)
    """
    x1, y1, x2, y2 = abs_box
    x_c = (x1 + x2) / 2 / sub_img_width
    y_c = (y1 + y2) / 2 / sub_img_height
    w = (x2 - x1) / sub_img_width
    h = (y2 - y1) / sub_img_height
    # 确保坐标在0-1范围内
    x_c = np.clip(x_c, 0, 1)
    y_c = np.clip(y_c, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)
    return x_c, y_c, w, h

def crop_by_gt_centers(
    img_path: str,
    label_path: str,
    crop_size: Tuple[int, int] = (640, 352),
    offset_range: Tuple[int, int] = (-100, 100),  # 中心点随机偏移范围（像素）
) -> List[Dict[str, any]]:
    """
    基于每个GT框的中心点随机偏移截取子图，k个GT框生成k张子图
    :param img_path: 输入图像路径
    :param label_path: YOLO格式标签文件路径
    :param crop_size: 截取子图尺寸 (width, height)，默认640x352
    :param offset_range: 中心点随机偏移范围 (min_offset, max_offset)，单位像素
    :return: 列表，每个元素为字典：
             {
                 "crop_img": 截取的子图 (np.ndarray),
                 "crop_labels": 子图对应的YOLO标签列表,
                 "crop_pos": (x1, y1, x2, y2) 截取区域在原图的坐标
             }
    """
    # 1. 加载图像和标签
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img_h, img_w = img.shape[:2]
    crop_w, crop_h = crop_size
    
    # 检查截取尺寸是否合法
    if crop_w > img_w or crop_h > img_h:
        raise ValueError(f"截取尺寸({crop_w}x{crop_h})大于图像尺寸({img_w}x{img_h})")
    
    # 加载并转换标签为绝对坐标
    yolo_labels = load_yolo_labels(label_path)
    if not yolo_labels:
        raise ValueError(f"标签文件为空或格式错误: {label_path}")
    
    # 转换所有GT框为绝对坐标，并计算每个框的中心点
    gt_boxes_abs = []  # 存储 (x1, y1, x2, y2, class_id, cx, cy)
    for label in yolo_labels:
        x1, y1, x2, y2, class_id = convert_yolo_to_abs(label, img_w, img_h)
        cx = (x1 + x2) / 2  # GT框中心点x
        cy = (y1 + y2) / 2  # GT框中心点y
        gt_boxes_abs.append((x1, y1, x2, y2, class_id, cx, cy))
    
    # 2. 为每个GT框生成子图
    crop_results = []
    min_offset, max_offset = offset_range
    for gt_box in gt_boxes_abs:
        _, _, _, _, _, gt_cx, gt_cy = gt_box
        
        # 2.1 对GT中心点进行随机偏移
        offset_x = random.randint(min_offset, max_offset)
        offset_y = random.randint(min_offset, max_offset)
        crop_cx = gt_cx + offset_x  # 截取区域的中心点x
        crop_cy = gt_cy + offset_y  # 截取区域的中心点y
        
        # 2.2 计算截取区域的左上角和右下角坐标（确保不越界）
        crop_x1 = int(crop_cx - crop_w / 2)
        crop_y1 = int(crop_cy - crop_h / 2)
        # 修正边界，避免截取区域超出图像
        crop_x1 = max(0, min(crop_x1, img_w - crop_w))
        crop_y1 = max(0, min(crop_y1, img_h - crop_h))
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        
        # 2.3 截取子图
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 2.4 筛选并转换该子图内的所有GT框标签
        crop_labels = []
        for box in gt_boxes_abs:
            bx1, by1, bx2, by2, cls_id, _, _ = box
            
            # 计算GT框与截取区域的交集（判断是否在子图内）
            inter_x1 = max(bx1, crop_x1)
            inter_y1 = max(by1, crop_y1)
            inter_x2 = min(bx2, crop_x2)
            inter_y2 = min(by2, crop_y2)
            
            # 有交集则保留该框，并转换为子图的YOLO坐标
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                # 转换为相对于子图的绝对坐标
                sub_bx1 = inter_x1 - crop_x1
                sub_by1 = inter_y1 - crop_y1
                sub_bx2 = inter_x2 - crop_x1
                sub_by2 = inter_y2 - crop_y1
                
                # 转换为YOLO归一化坐标
                yolo_xc, yolo_yc, yolo_w, yolo_h = convert_abs_to_yolo(
                    (sub_bx1, sub_by1, sub_bx2, sub_by2),
                    crop_w,
                    crop_h
                )
                crop_labels.append([cls_id, yolo_xc, yolo_yc, yolo_w, yolo_h])
        
        # 2.5 保存该子图的结果
        crop_results.append({
            "crop_img": crop_img,
            "crop_labels": crop_labels,
            "crop_pos": (crop_x1, crop_y1, crop_x2, crop_y2)
        })
    
    return crop_results

def save_crop_results(
    crop_results: List[Dict[str, any]],
    save_dir: str,
    img_prefix: str = "crop"
) -> None:
    """
    批量保存截取的子图和标签文件
    :param crop_results: crop_by_gt_centers的输出结果
    :param save_dir: 保存目录（会自动创建）
    :param img_prefix: 子图文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(crop_results):
        crop_img = result["crop_img"]
        crop_labels = result["crop_labels"]
        crop_pos = result["crop_pos"]
        
        # 生成保存路径
        img_save_path = os.path.join(save_dir, f"{img_prefix}_{idx:04d}.jpg")
        label_save_path = os.path.join(save_dir, f"{img_prefix}_{idx:04d}.txt")
        
        # 保存子图
        cv2.imwrite(img_save_path, crop_img)
        
        # 保存标签文件
        with open(label_save_path, 'w', encoding='utf-8') as f:
            for label in crop_labels:
                cls_id, xc, yc, w, h = label
                f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"保存子图 {idx+1}/{len(crop_results)}: {img_save_path}")
        print(f"保存标签 {idx+1}/{len(crop_results)}: {label_save_path}")

# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 输入路径
    save_dir = "./data/boxes/cropped"  # 截取结果保存目录
    img_dir = "/home/double/Documents/ultralytics/data/boxes/images/train"
    label_dir = "/home/double/Documents/ultralytics/data/boxes/labels/train"
    img_names = os.listdir(img_dir)
    img_names = [name for name in img_names if name.endswith(".jpg")]
    if len(img_names) == 0:
        raise ValueError(f"图像目录 {img_dir} 中没有找到jpg文件")

    good_label_quantity_prefixs = ["3_sampled", "4_sampled", "8_sampled", "9_sampled", "10_sampled", "11_sampled", "12_sampled", 
                                   "13_sampled", "17_sampled", "18_sampled", "23_sampled", "24_sampled", "25_sampled", "26_sampled", "27_sampled"]
    for img_name in img_names:
        if not any(img_name.startswith(prefix) for prefix in good_label_quantity_prefixs):
            continue
        
        input_img_path = os.path.join(img_dir, img_name)
        input_label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        
        # 执行截取
        print(f"正在处理图像: {input_img_path}")
        crop_results = crop_by_gt_centers(
            img_path=input_img_path,
            label_path=input_label_path,
            crop_size=(640, 352),
            offset_range=(-100, 100)  # 中心点随机偏移±100像素
        )

        # for i in range(len(crop_results)):
        #     first_crop = crop_results[i]
        #     img = visualize_cropped_detections(
        #         cropped_img=first_crop["crop_img"],
        #         cropped_labels=first_crop["crop_labels"],
        #         show=False,
        #     )
        #     cv2.imshow(f"Cropped Image", img)
        #     cv2.waitKey(0)

        save_crop_results(crop_results, save_dir=save_dir, img_prefix=img_name.replace(".jpg", "")+"_crop")
    
    