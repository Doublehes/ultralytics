import os
import cv2
from tqdm import tqdm


cls_name_to_id = {
    "box": 0,
}


if __name__ == "__main__":
    data_root = "/home/double/Documents/data/boxes/data_deeptouch"
    label_dir = "label"

    save_dir = "/home/double/Documents/ultralytics/data/boxes"
    if os.path.exists(save_dir):
        print(f"Save directory {save_dir} already exists. Remove it now.")
        os.system(f"rm -rf {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    label_txts = os.listdir(os.path.join(data_root, label_dir))

    dataset_pbar = tqdm(total=len(label_txts), desc="Dataset Processing")
    for label_txt in label_txts:
        txt_path = os.path.join(data_root, label_dir, label_txt)
        info_lines = open(txt_path, 'r').readlines()
        
        info_pbar = tqdm(total=len(info_lines), desc=f"Processing {label_txt}")
        for i, line in enumerate(info_lines):
            line = line.strip()
            line_dict = eval(line)
            img_path = line_dict["CAM_F"]["filepath"]
            dataset_name = img_path.split("/")[2]
            save_name = f"{dataset_name}_{i}"
            img = cv2.imread(os.path.join(data_root, img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                info_pbar.update(1)
                continue
            img_h, img_w = img.shape[0], img.shape[1]
            yolo_lines = []
            for box in line_dict["CAM_F"]["box2d"]:
                cls_name = box["name"]
                assert cls_name in cls_name_to_id, f"Unknown class name: {cls_name}"
                cls_id = cls_name_to_id[cls_name]
                x1, y1, x2, y2 = box["points"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                cx_norm = cx / img_w
                cy_norm = cy / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                yolo_line = f"{cls_id} {cx_norm} {cy_norm} {w_norm} {h_norm}\n"
                yolo_lines.append(yolo_line)
            
            if len(yolo_lines) == 0:
                info_pbar.update(1)
                continue
            
            train_val = "train"
            if "5_sampled" in img_path or "10_sampled" in img_path or "15_sampled" in img_path or "20_sampled" in img_path or "25_sampled" in img_path:
                train_val = "val"
            img_save_dir = os.path.join(save_dir, "images", train_val)
            img_save_path = os.path.join(img_save_dir, f"{save_name}.jpg")
            label_save_dir = os.path.join(save_dir, "labels", train_val)
            label_save_path = os.path.join(label_save_dir, f"{save_name}.txt")
            os.makedirs(img_save_dir, exist_ok=True)
            os.makedirs(label_save_dir, exist_ok=True)
            cv2.imwrite(img_save_path, img)
            with open(label_save_path, 'w') as f:
                f.writelines(yolo_lines)
            info_pbar.update(1)
        info_pbar.close()
        
        dataset_pbar.update(1)
    dataset_pbar.close()