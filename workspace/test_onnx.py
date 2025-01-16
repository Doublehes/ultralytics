import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics.utils.ops import non_max_suppression

# 加载 ONNX 模型
onnx_model_path = './yolo11n.onnx'
session = ort.InferenceSession(onnx_model_path)

# 加载输入图像
image_path = './bus.jpg'
image = cv2.imread(image_path)

# 预处理：调整图片大小并转为适合模型输入的格式
input_height, input_width = 640, 640  # 根据模型的输入要求调整
image_resized = cv2.resize(image, (input_width, input_height))

# 转换为 NCHW 格式，并归一化
image_input = image_resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
image_input = np.expand_dims(image_input, axis=0)  # 增加批量维度
image_input /= 255.0  # 根据模型需求进行归一化

# 进行推理
inputs = {session.get_inputs()[0].name: image_input}
outputs = session.run(None, inputs)

# 解析输出结果（目标检测模型输出为 [batch, 84, num_detections]，其中1维包含 [x1, y1, x2, y2, score, classes]）
detections = outputs[0]  # 取出 batch 中的第一个结果

# 转换成 PyTorch 张量
detections = torch.from_numpy(detections)
# 应用非极大值抑制（NMS）以去除重复的检测框, 输出List[Tensor[N, 6]], 6个值分别为[x1, y1, x2, y2, score, class_id]
detections = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.5)
# 取出第一个结果
detections = detections[0]
# 将结果转换为 numpy 数组
detections = detections.cpu().numpy()

# 绘制检测框
for detection in detections:
    x1, y1, x2, y2, score, class_id = detection
    if score > 0.5:  # 过滤掉低置信度的框
        # 还原坐标到原始图像的尺度
        x1 *= image.shape[1] / input_width
        y1 *= image.shape[0] / input_height
        x2 *= image.shape[1] / input_width
        y2 *= image.shape[0] / input_height
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'Class {int(class_id)}: {score:.2f}', (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow('Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
