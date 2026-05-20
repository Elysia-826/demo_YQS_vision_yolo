from ultralytics import YOLO
import os

# 可替换为训练得到的 best.pt 实际路径
PT_PATH = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v3\weights\best.pt"

def export_and_verify():
    # 1. 检查权重文件是否存在
    if not os.path.exists(PT_PATH):
        raise FileNotFoundError(f"❌ 找不到模型文件: {PT_PATH}，请检查路径")

    # 2. 加载训练好的最佳权重
    print(f"📦 正在加载 PyTorch 模型: {PT_PATH}")
    model = YOLO(PT_PATH)

    # 3. 导出为 ONNX 格式
    # opset=12    : 兼容 OpenCV DNN / X-AnyLabeling / TensorRT 的标准算子版本
    # dynamic=False: 固定 640x640 输入尺寸，C++部署与标注软件更稳定
    # half=False  : 强制 FP32 精度，避免 CPU/OpenCV 下的 FP16 兼容报错
    # simplify=True: 清理冗余计算节点，减小体积并提升推理速度
    print("⚙️ 正在导出 ONNX 模型...")
    onnx_path = model.export(
        format="onnx",
        imgsz = 640,
        opset=12,
        dynamic=False,
        half=False,
        simplify=True,
        nms = True
    )
    print(f"✅ 模型已成功导出至: {onnx_path}")

if __name__ == "__main__":
    export_and_verify()