import os
import random
from tqdm import tqdm

def sample_mini_dataset(data_dir, save_dir, trainval_name, sample_num=1000):
    images_dir = os.path.join(data_dir, trainval_name, "images")
    labels_dir = os.path.join(data_dir, trainval_name, "labels")

    save_images_dir = os.path.join(save_dir, trainval_name, "images")
    save_labels_dir = os.path.join(save_dir, trainval_name, "labels")
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)

    img_names = os.listdir(images_dir)
    sample_images = random.sample(img_names, sample_num)
    for img_name in tqdm(sample_images):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

        save_img_path = os.path.join(save_images_dir, img_name)
        save_label_path = os.path.join(save_labels_dir, img_name.replace(".jpg", ".txt"))

        os.system(f"cp {img_path} {save_img_path}")
        os.system(f"cp {label_path} {save_label_path}")
    


if __name__ == "__main__":
    root_dir = "/home/double/Documents/BEVDet/data/nuScenese/yolo_dataset"
    data_dir = os.path.join(root_dir, "nuscenes_data3d_all")
    save_dir = os.path.join(root_dir, "nuscenes_data3d_mini")

    train_name = "train3d"
    sample_mini_dataset(data_dir, save_dir, train_name, sample_num=10000)

    val_name = "val3d"
    sample_mini_dataset(data_dir, save_dir, val_name, sample_num=1000)

    pass