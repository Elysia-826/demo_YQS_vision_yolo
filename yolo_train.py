from ultralytics import YOLO

# 1. 加载上一版修正后的模型（best_fixed.pt）
model = YOLO(r"C:\Users\YQS\Desktop\yolo\yolo11s.pt")

# 2. 训练 v2 模型
results = model.train(
    data="C:/Users/YQS/Desktop/yolo/dataset/data.yaml",   # 新数据集的 yaml，务必是正确顺序
    epochs=150,                 # 可以适当增加，观察收敛
    lr0=0.0005,        # 关键：学习率降低一个数量级，防止一步跨过最优点
    lrf=0.01,
    warmup_epochs=5,   # 更长的预热
    imgsz=640,
    batch=16,                    # 根据显存调整
    device=0,                   # GPU 0
    workers=4,                  # 根据 CPU 核心数调整
    patience=30,                # 早停，避免过度训练
    project="runs/temp_seed",   # 保存目录
    name="v4",                  # 本次训练名称
    exist_ok=False,             # 不覆盖已有同名目录
    # 以下为推荐的数据增强参数（可以根据需要开启）
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,                 # 左右翻转可能有帮助，但注意装甲板特征对称性
    mosaic=1.0,                 # 使用 mosaic 增强
    close_mosaic=15,            # 延后关闭，避免早期过拟合
    mixup=0.2,                  # 轻微 mixup
    copy_paste=0.3,             # 轻微复制粘贴增强
    multi_scale=True,
)