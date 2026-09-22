import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def run_cv_onnx_demo():
    print("\n[1/4] 正在加载轻量级 YOLOv8 视觉模型...")
    # 第一次运行会自动下载一个只有几兆字节的预训练微型模型
    model = YOLO("yolov8n.pt") 

    print("[2/4] 正在把模型导出为通用的 .onnx 格式...")
    model.export(format="onnx")  # 在当前文件夹下自动生成 yolov8n.onnx

    print("[3/4] 使用 ONNX Runtime 部署引擎加载本地 yolov8n.onnx 模型...")
    session = ort.InferenceSession("yolov8n.onnx")

    # 获取模型的输入和输出节点名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"     ✅ 模型节点解析成功！输入节点名: {input_name}")

    print("[4/4] 构造模拟图像数据并运行本地 ONNX 推理...")
    # 模拟一张 640x640 像素的 RGB 图像数据
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # 在本地用 ONNX 引擎执行推理（不依赖 PyTorch）
    outputs = session.run([output_name], {input_name: dummy_input})

    print(f"\n🎉 祝贺！CV 本地 ONNX 部署推理成功！输出 Tensor 形状: {outputs[0].shape}\n")
    return {"status": "Success", "output_shape": str(outputs[0].shape)}

if __name__ == "__main__":
    run_cv_onnx_demo()