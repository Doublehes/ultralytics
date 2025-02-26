from ultralytics import YOLO
import os
import cv2
import time
import pyautogui
import numpy as np

from mss import mss
from PIL import ImageGrab


sct = mss()

def screen_capture():
    # region = (0, 300, 1920, 1080) # 定义截图区域(left, top, right, bottom)
    # screenshot = ImageGrab.grab(all_screens=False)
    
    # screenshot = sct.grab(sct.monitors[1])  # sct.monitors[1] 指代第一个显示器
    # img = np.array(screenshot)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]  # BGRA to BGR

    return img


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
        if len(label) == 0:
            label = np.array([])
        elif len(label.shape) == 1:
            label = label[np.newaxis, :]

        yield img, label


if __name__ == "__main__":
    model = YOLO("runs/train_nuimage/yolo11n_nuimage_bs8_ep100_sz960p/weights/best.pt", task="detect")

    data_dir = "/media/double/Data/datasets/nuImages/yolo_dataset_train"
    img_iterator = data_iterator(data_dir)
    while True:
        t1 = time.time()

        # img = screen_capture()

        img, label = next(img_iterator)
        
        result = model.predict(img, verbose=True, imgsz=960)[0]
        img = result.plot()

        for box in label:
            cls, x, y, w, h = box
            x, y, w, h = x * 1600, y * 900, w * 1600, h * 900
            l, t, r, b = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.imshow("result", img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        cv2.waitKey(0)

        t2 = time.time()
        time.sleep(max(0, 2.0 - (t2 - t1)))

    # 释放资源
    cv2.destroyAllWindows()

    