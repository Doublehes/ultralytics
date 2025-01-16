from ultralytics import YOLO

model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.pt", task="detect")
res = model.val(data="VOC.yaml", imgsz=640, device="cpu")

model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.onnx", task="detect")
res = model.val(data="VOC.yaml", imgsz=640, device="cpu")

model = YOLO("/home/double/Documents/ultralytics/runs/detect/train/weights/best.torchscript", task="detect")
res = model.val(data="VOC.yaml", imgsz=640, device="cpu")
# res = model.benchmark(data="VOC.yaml", imgsz=640)


