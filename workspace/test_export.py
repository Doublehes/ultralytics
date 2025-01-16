from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolov6s.yaml", task="detect")


# Export the model to ONNX format
onnx_path = model.export(format="onnx", simplify=True)  # creates 'yolo11n.onnx'
# Load the exported ONNX model
onnx_model = YOLO(onnx_path, task="detect")

torchscript_path = model.export(format="torchscript")
torchscript_model = YOLO(torchscript_path, task="detect")

inputs = "bus.jpg"

result1 = model.predict(inputs, stream=False)
result1[0].save("1.jpg")
result2 = onnx_model.predict(inputs, stream=False, device="cpu")
result2[0].save("2.jpg")
result3 = torchscript_model.predict(inputs, stream=False)
result3[0].save("3.jpg")
pass