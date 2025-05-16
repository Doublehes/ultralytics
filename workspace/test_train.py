from ultralytics import YOLO, RTDETR, DETR


device = "0" # 0, cpu

# resume
resume = False

pretrained = False
pt_path = "/home/double/Documents/ultralytics/runs/train_shape/detr_shape_bs8_ep200_imgsz320_id/weights/last.pt"

data_yaml = "./workspace/config/shape.yaml"
# data_yaml = "./workspace/config/nuimage.yaml"
# data_yaml = "./workspace/config/nuscenese-2d.yaml"
# data_yaml = "./workspace/config/coco.yaml"
# data_yaml = "VOC.yaml"
# data_yaml = "coco8.yaml"

# cfg_yamf = "./workspace/config/dense5.yaml"
# cfg_yamf = "yolo11n.yaml"
# cfg_yamf = "./workspace/config/rtdetr-s.yaml"
cfg_yamf = "./workspace/config/detr.yaml"


if "rtdetr" in cfg_yamf:
    MODEL_CLASS = RTDETR
elif "detr" in cfg_yamf:
    MODEL_CLASS = DETR
else:
    MODEL_CLASS = YOLO

test = False
imgsz = 320
batch = 16
epochs = 100

data_name = data_yaml.split('/')[-1].split('.')[0]
model_name = cfg_yamf.split('/')[-1].split('.')[0]
project = f"./runs/train_{data_name}"
if test:
    project = project + "_test"

run_name = f"{model_name}_{data_name}_bs{batch}_ep{epochs}_imgsz{imgsz}_id"

if resume:
    model_pt = f"./{project}/{run_name}/weights/last.pt"
    model = MODEL_CLASS(model_pt)
    model.train(resume=True, device=device)  # resume training
elif pretrained:
    model = MODEL_CLASS(pt_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                pretrained=True, project=project, workers=4, close_mosaic=10,
                warmup_epochs=50,
                name=run_name, plots=True, device=device)
else:
    model = MODEL_CLASS(cfg_yamf)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                        pretrained=True, project=project, workers=4, close_mosaic=10,
                        name=run_name, plots=True, device=device)

