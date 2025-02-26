from ultralytics import YOLO
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.bev_visualizer import Visualizer

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
    pt = "runs/train_nuscenese-3d_test/yolo11n-3d_nuscenese-3d_bs8_ep100_sz960p_rect_wx0.2_wy0.5_10000/weights/best.pt"
    model = YOLO(pt, task="detect3d")

    data_dir = "/media/double/Data/datasets/nuScenese/yolo_dataset/val3d"
    iterator = data_iterator(data_dir, with_label=True)
    orig_size = 1600
    infer_size = 960
    visualizer = Visualizer(img_width=1600, img_height=900)
    while True:
        img, label = next(iterator)
        # import pudb; pudb.set_trace()
        result = model.predict(img, verbose=True, imgsz=infer_size)[0]
        result_3d = result.result_3d.cpu().numpy() # (N, 8): l, t, r, b, conf, cls, x, y
        result_3d[:, :4] *= orig_size / infer_size

        print(result_3d)
        print(label)
        if len(label) == 0:
            continue

        boxes_3d = []
        for i, box in enumerate(result_3d):
            l, t, r, b, conf, cls, x, y = box
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
            boxes_3d.append([i, x, y, 4, 2, 0])
        boxes_3d_gt = []
        for i, box in enumerate(label):
            cls, cx, cy, cw, ch, x, y, z, width, height, length, yaw, dist = box
            cx, cy, cw, ch = cx * 1600, cy * 900, cw * 1600, ch * 900
            l, t, r, b = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            boxes_3d_gt.append([i, x, y, length, width, yaw])
        img_bev = visualizer.generate_blank_img_3d()
        visualizer.draw_3d_box_ego(img_bev)
        visualizer.draw_3d_bboxes(img_bev, boxes_3d, color=visualizer.blue_color)
        visualizer.draw_3d_bboxes(img_bev, boxes_3d_gt, color=visualizer.green_color)
        visualizer.draw_circle(img_bev, [0, 0], 30)
        visualizer.draw_circle(img_bev, [0, 0], 60)
        
        img_concat = np.concatenate([img, img_bev], axis=0)
        
        cv2.namedWindow('imgshow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imgshow', 1400, 1400) 
        cv2.imshow('imgshow', img_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # plt.imshow(img[:, :, ::-1])
        # plt.show()
        # plt.imshow(img_bev[:, :, ::-1])
        # plt.show()

    

    

        

    


    