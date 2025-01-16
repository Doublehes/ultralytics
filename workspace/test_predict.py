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


def image_iterator(img_dir):
    for img_path in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_path)
        img = cv2.imread(img_path)
        yield img


if __name__ == "__main__":
    model = YOLO("./models/yolo11x.pt", task="detect")

    img_dir = "/home/double/Documents/datasets/nuImages/samples/CAM_FRONT"
    img_iterator = image_iterator(img_dir)
    while True:
        t1 = time.time()

        # img = screen_capture()

        img = next(img_iterator)
        
        result = model.predict(img, verbose=True, imgsz=960)[0]
        img = result.plot()

        cv2.imshow("result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t2 = time.time()
        time.sleep(max(0, 1.0 - (t2 - t1)))

    # 释放资源
    cv2.destroyAllWindows()

    