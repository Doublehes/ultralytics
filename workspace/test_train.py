from ultralytics import YOLO, RTDETR


# resume
resume = False

# data_yaml = "./workspace/config/shape.yaml"
# data_yaml = "./workspace/config/nuimage.yaml"
data_yaml = "./workspace/config/nuscenese-2d.yaml"
# data_yaml = "VOC.yaml"
# data_yaml = "dota8.yaml"

# cfg_yamf = "./workspace/config/dense5.yaml"
cfg_yamf = "yolo11n.yaml"
# cfg_yamf = "./workspace/config/rtdetr-s.yaml"
# cfg_yamf = "yolo11n-obb.yaml"

MODEL_CLASS = YOLO
test = True

data_name = data_yaml.split('/')[-1].split('.')[0]
model_name = cfg_yamf.split('/')[-1].split('.')[0]
project = f"../runs/train_{data_name}"
if test:
    project = project + "_test"
imgsz = 960
batch = 8
epochs = 100
run_name = f"{model_name}_{data_name}_bs{batch}_ep{epochs}_imgsz{imgsz}_id"

if resume:
    model_pt = f"./{project}/{run_name}/weights/last.pt"
    model = MODEL_CLASS(model_pt)
    model.train(resume=True)  # resume training
else:
    model = MODEL_CLASS(cfg_yamf)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                        pretrained=True, project=project, workers=4,
                        name=run_name, plots=True)

