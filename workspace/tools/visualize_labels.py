import glob
import os
import cv2

root_dir = "/home/double/Documents/ultralytics/data/boxes"
img_names = glob.glob(os.path.join(root_dir, "images/train", "*.jpg"))

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
for img_name in img_names:
    img = cv2.imread(img_name)
    h, w, _ = img.shape
    label_name = img_name.replace("images", "labels").replace(".jpg", ".txt")
    with open(label_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            cls_id = int(line[0])
            x_center = float(line[1]) * w
            y_center = float(line[2]) * h
            box_w = float(line[3]) * w
            box_h = float(line[4]) * h
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(img_name)
    cv2.imshow("img", img)
    cv2.waitKey(0)