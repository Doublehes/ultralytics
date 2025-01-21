from ultralytics import YOLO

# model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.pt", task="detect")
# res = model.val(data="VOC.yaml", imgsz=640, device="cpu")

# model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.onnx", task="detect")
# res = model.val(data="VOC.yaml", imgsz=640, device="cpu")

# model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.torchscript", task="detect")
# res = model.val(data="VOC.yaml", imgsz=640, device="cpu")
# res = model.benchmark(data="VOC.yaml", imgsz=640)


model = YOLO("runs/train_nuscenese-3d_test/yolo11n-3d_nuscenese-3d_bs8_ep100_sz960p/weights/last.pt", task="detect3d")
res = model.val(data="workspace/config/nuscenese-3d.yaml", imgsz=960)
