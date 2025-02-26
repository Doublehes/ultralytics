from ultralytics import YOLO, RTDETR


# resume
resume = True

# data_yaml = "./workspace/config/nuimage.yaml"
# data_yaml = "./workspace/config/nuscenese-2d.yaml"
data_yaml = "./workspace/config/nuscenese-3d.yaml"

cfg_yamf = "./workspace/config/yolo11n-3d.yaml"
# cfg_yamf = "yolo11n.yaml"

name_suffix = "_wx0.2_wy0.5_10000"

pretrained = None
MODEL_CLASS = YOLO
test = True
argumentation = False
rect_train = True

data_name = data_yaml.split('/')[-1].split('.')[0]
model_name = cfg_yamf.split('/')[-1].split('.')[0]
project = f"runs/train_{data_name}"
if test:
    project = project + "_test"
imgsz = 960
batch = 8
epochs = 100
run_name = f"{model_name}_{data_name}_bs{batch}_ep{epochs}_sz{imgsz}p"

args = {
    "translate": 0.0,
    "scale": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "erasing": 0.0
}
if argumentation:
    args = dict()
    run_name = run_name + "_aug"

if rect_train:
    run_name = run_name + "_rect"

if name_suffix:
    run_name = run_name + name_suffix

if resume:
    model_pt = f"./{project}/{run_name}/weights/last.pt"
    model = MODEL_CLASS(model_pt)
    model.train(resume=True)  # resume training
else:
    if pretrained:
        model = MODEL_CLASS(pretrained, task='detect3d')
    else:
        model = MODEL_CLASS(cfg_yamf, task='detect3d')
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                        pretrained=True, project=project, workers=4, rect=rect_train,
                        name=run_name, plots=True, **args)

