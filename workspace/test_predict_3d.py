from ultralytics import YOLO
import os
import cv2
import time
import numpy as np
from utils.bev_visualizer import Visualizer


def resize_with_padding(image, target_size=(256, 128)):
    """
    将图像保持比例地resize并填充到目标尺寸
    target_size: (width, height)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))

    # 创建目标画布，用均值颜色填充
    mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
    padded = np.full((target_h, target_w, 3), mean_color, dtype=np.uint8)

    # 将缩放后的图像放在中心
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


def data_iterator(data_dir):
    img_dir = os.path.join(data_dir, "images")
    img_names = os.listdir(img_dir)
    img_names.sort()
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        if not os.path.exists(label_path):
            yield img, None
            continue
        label = np.loadtxt(label_path)
        if len(label) == 0:
            label = np.array([])
        elif len(label.shape) == 1:
            label = label[np.newaxis, :]

        yield img, label


if __name__ == "__main__":
    pt = "runs/train_nuscenese-3d-new/yolo11n-3d_nuscenese-3d-new_bs8_ep50_sz960p_rect_wx0.2_wy0.54/weights/best.pt"
    model = YOLO(pt, task="detect3d")

    data_dir = "/home/double/Documents/BEVDet/data/nuScenese/yolo_dataset/nuscenes_data3d_all/val3d"
    # data_dir = "/home/double/Documents/BEVDet/data/nuScenese/yolo_dataset/data3d_new/val3d"
    # data_dir = "/home/double/Documents/data/bag_data/als_tms3/als_tracking/cross/test_mid_f"
    # data_dir = "/home/double/Documents/data/bag_data/als_tms3/als_tracking/same/test_mid_f"
    # data_dir = "/home/double/Documents/data/bag_data/als_tms3/als_tracking/opposite/test_mid_f"
    # data_dir = "/home/double/Documents/data/bag_data/als_tms3/extract_f600/test_mid_f"
    # data_dir = "/home/double/Documents/data/bag_data/als_tms3/extract_f300/test_mid_f"
    iterator = data_iterator(data_dir)
    infer_size = 960
    show_size = 960
    conf_thrs = 0.4
    visualizer = Visualizer(img_width=1600, img_height=900)
    cv2.namedWindow('imgshow', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgshow', show_size, int(show_size*1.2))
    while True:
        img, label = next(iterator)
        # img = resize_with_padding(img, target_size=(infer_size, infer_size))
        img_h, img_w = img.shape[0], img.shape[1]
        result = model.predict(img, verbose=True, imgsz=infer_size, conf=conf_thrs, show=False)[0]
        result_3d = result.result_3d.cpu().numpy() # (N, 8): l, t, r, b, conf, cls, x, y
        # result_3d[:, :4] *= img_w / infer_size

        boxes_3d = []
        for i, box in enumerate(result_3d):
            l, t, r, b, conf, cls, x, y, width, height, length, s_yaw, c_yaw = box
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
            cv2.putText(img, f"{i}", (int(l), int(t) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            yaw = np.arctan2(s_yaw, c_yaw)
            boxes_3d.append([i, x, y, length, width, yaw])

        img_bev = visualizer.generate_blank_img_3d()
        visualizer.draw_3d_box_ego(img_bev)
        visualizer.draw_3d_bboxes(img_bev, boxes_3d, color=visualizer.blue_color)
        visualizer.draw_circle(img_bev, [0, 0], 30)
        visualizer.draw_circle(img_bev, [0, 0], 60)

        if label is not None:
            boxes_3d_gt = []
            for i, box in enumerate(label):
                if len(box) > 12:
                    box = box[:12]
                cls, cx, cy, cw, ch, x, y, z, width, height, length, yaw = box
                cx, cy, cw, ch = cx * 1600, cy * 900, cw * 1600, ch * 900
                l, t, r, b = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
                cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                boxes_3d_gt.append([i, x, y, length, width, yaw])
            visualizer.draw_3d_bboxes(img_bev, boxes_3d_gt, color=visualizer.green_color)

        img = cv2.resize(img, (show_size, img_h * show_size // img_w))
        img_bev = cv2.resize(img_bev, (show_size, img_bev.shape[0] * show_size // img_bev.shape[1]))
        img_concat = np.concatenate([img, img_bev], axis=0)
        
        cv2.imshow('imgshow', img_concat)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    

    

        

    


    