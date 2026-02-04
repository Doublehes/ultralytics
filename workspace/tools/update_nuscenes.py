import cv2
import json
import os
import numpy as np
from tqdm import tqdm


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


if __name__ == "__main__":
    data_root = "/home/double/Documents/BEVDet/data/nuScenese/yolo_dataset/data3d_new/val3d"
    img_dir = f"{data_root}/images"
    label_dir = f"{data_root}/labels"
    refresh_dir = f"{data_root}/labels_refresh"
    new_label_dir = f"{data_root}/labels_new"
    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)

    imgs = os.listdir(img_dir)
    imgs.sort()
    for img_name in tqdm(imgs):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        refresh_path = os.path.join(refresh_dir, img_name.replace(".jpg", ".json"))

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[0], img.shape[1]

        # 原始标签
        obj_lines = open(label_path, "r").readlines()
        obj_ltrb_list = []
        for obj_line in obj_lines:
            obj_info = obj_line.strip().split(" ")
            cls_id = int(float(obj_info[0]))
            x = float(obj_info[1]) * img_w
            y = float(obj_info[2]) * img_h
            w = float(obj_info[3]) * img_w
            h = float(obj_info[4]) * img_h
            depth = float(obj_info[5])
            l, t, r, b = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            obj_ltrb_list.append([l, t, r, b])

            # cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 1)
            # cv2.putText(img, f"{cls_id}:{depth:.1f}m", (int(x - w / 2), int(y - h / 2) - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 刷新后的标签
        with open(refresh_path, "r") as f:
            objs_refresed = json.load(f)
        refreshed_ltrb_list = []
        for obj in objs_refresed:
            ltrb = obj["points"]
            refreshed_ltrb_list.append(ltrb)

            # cv2.rectangle(img, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)

        # 关联并生成新标签
        new_labels = []
        if len(refreshed_ltrb_list) > 0 and len(obj_ltrb_list) > 0:
            obj_ltrb_array = np.array(obj_ltrb_list)
            refreshed_ltrb_array = np.array(refreshed_ltrb_list)
            iou_matrix = calc_iou2d(refreshed_ltrb_array, obj_ltrb_array)
            matched_indices = np.argmax(iou_matrix, axis=1)
            for i, matched_idx in enumerate(matched_indices):
                iou = iou_matrix[i, matched_idx]
                refreshed_info = objs_refresed[i]
                assert refreshed_info["name"] in ["vehicle", "pedestrian"]
                cls_id = 0 if refreshed_info["name"] == "vehicle" else 1
                box2d_info = [cls_id] + refreshed_info["points"]
                if iou > 0.5:
                    original_info = obj_lines[matched_idx].strip().split(" ")
                    box3d_info =  original_info[5:12]
                else:
                    box3d_info = [1000, 1000, 1000, 0, 0, 0, 0]  # 默认一个远距离的3D框
                new_label = box2d_info + box3d_info
                new_labels.append(new_label)
        else:
            for refreshed_info in objs_refresed:
                ltrb = refreshed_info["points"]
                assert refreshed_info["name"] in ["vehicle", "pedestrian"]
                cls_id = 0 if refreshed_info["name"] == "vehicle" else 1
                box2d_info = [cls_id] + ltrb
                box3d_info = [1000, 1000, 1000, 0, 0, 0, 0]  # 默认一个远距离的3D框
                new_label = box2d_info + box3d_info
                new_labels.append(new_label)
        
        show = False
        if show:
            for new_label in new_labels:
                cls_id = int(new_label[0])
                l, t, r, b = new_label[1:5]
                depth = float(new_label[5])

                cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 2)
                str = f"{cls_id}: {depth:.1f}m" if depth < 1000 else f"{cls_id}: inf"
                cv2.putText(img, str, (int(l), int(t) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("img", img)
            cv2.waitKey(0)
        
        else:
            # TODO: convert 2d ltrb to xywh and normalize
            new_label_path = os.path.join(new_label_dir, img_name.replace(".jpg", ".txt"))
            with open(new_label_path, "w") as f:
                for new_label in new_labels:
                    l, t, r, b = new_label[1:5]
                    x = (l + r) / 2 / img_w
                    y = (t + b) / 2 / img_h
                    w = (r - l) / img_w
                    h = (b - t) / img_h
                    new_label[1:5] = [x, y, w, h]
                    str_line = " ".join([str(x) for x in new_label])
                    f.write(str_line + "\n")



