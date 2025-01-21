from ultralytics import YOLO
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def data_iterator(data_dir, with_label=True):
    img_dir = os.path.join(data_dir, "images")
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if not with_label:
            yield img, None
            continue
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        label = np.loadtxt(label_path)
        yield img, label


if __name__ == "__main__":
    pt = "/home/double/Documents/ultralytics/runs/train_nuscenese-3d_test/yolo11n-3d_nuscenese-3d_bs8_ep100_sz960p/weights/best.pt"
    model = YOLO(pt, task="detect3d")

    data_dir = "/media/double/Data/datasets/nuScenese/yolo_dataset/val3d"
    iterator = data_iterator(data_dir, with_label=True)
    orig_size = 1600
    infer_size = 960
    while True:
        img, label = next(iterator)

        result = model.predict(img, verbose=True, imgsz=infer_size)[0]
        
        result_3d = result.result_3d.cpu().numpy() # (N, 8): l, t, r, b, conf, cls, x, y
        result_3d[:, :4] *= orig_size / infer_size
        for box in result_3d:
            l, t, r, b, conf, cls, x, y = box
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        print(result_3d)
        print(label)
        plt.imshow(img[:, :, ::-1])
        plt.show()

    

    

        

    


    