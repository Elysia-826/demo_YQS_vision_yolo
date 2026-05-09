from ultralytics import YOLO
import os

# 可替换为训练得到的 best.pt 实际路径
PT_PATH = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best.pt"
# 可替换为用于验证的本地测试图片路径
TEST_IMG_PATH = r"C:\Users\YQS\Desktop\yolo\dataset\images\val"

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
        opset=12,
        dynamic=False,
        half=False,
        simplify=True
    )
    print(f"✅ 模型已成功导出至: {onnx_path}")

    # 4. 加载导出的 ONNX 模型进行验证推理
    print("🔍 正在加载 ONNX 模型并测试推理...")
    onnx_model = YOLO(onnx_path)

    # 5. 运行推理验证（使用本地测试图片）
    if not os.path.exists(TEST_IMG_PATH):
        print(f"⚠️ 测试图片 {TEST_IMG_PATH} 不存在，跳过推理验证。")
        return

    # conf=0.25 适用于辅助标注阶段（保召回）；最终部署可改为 0.4~0.5
    results = onnx_model(TEST_IMG_PATH, conf=0.25, verbose=False)

    # 6. 解析并打印推理结果
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"🎯 检测到 {len(boxes)} 个目标:")
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                cls_name = r.names[cls_id]
                print(f"   - 类别: {cls_name} (ID:{cls_id}) | 置信度: {conf:.2f} | 坐标: {xyxy}")
        else:
            print("🔍 未检测到目标。")

if __name__ == "__main__":
    export_and_verify()