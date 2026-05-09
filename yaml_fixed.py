import torch
import os

# ================== 配置区 ==================
WRONG_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best.pt"
FIXED_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best_fixed.pt"

# 你的映射：错误索引 -> 正确索引
new_order = [1, 0, 2, 4, 3, 5]
# ===========================================

# 1. 检查文件存在
if not os.path.exists(WRONG_PT):
    raise FileNotFoundError(f"找不到模型文件：{WRONG_PT}")

# 2. 用 torch.load 直接加载 checkpoint，不经过 ultralytics 封装
print("正在加载模型 checkpoint（完全离线）...")
checkpoint = torch.load(WRONG_PT, map_location='cpu', weights_only=False)

# 3. 从 checkpoint 中提取模型结构（通常包含在 'model' 字段）
#    也可能是整个模型保存，根据 ultralytics 保存格式适配
if 'model' in checkpoint:
    model = checkpoint['model']          # 通常是完整的 nn.Module 对象
else:
    # 有些旧版本可能直接是 state_dict，需要先构建模型再加载，但我们这里只修改权重
    raise ValueError("未找到 'model' 字段，请确认这是一个完整的 Ultralytics checkpoint")

# 4. 定位检测头分类卷积层
#    Ultralytics 模型结构一般是 model.model[-1] 为 Detect 模块
try:
    detect = model.model[-1]  # 检测头
    cls_conv = detect.cv2[-1]  # 分类分支的最后一个卷积
except Exception as e:
    raise RuntimeError("无法定位分类卷积层，模型结构可能不标准") from e

# 5. 验证类别数
nc = cls_conv.out_channels
if len(new_order) != nc:
    raise ValueError(f"映射长度 ({len(new_order)}) 与类别数 ({nc}) 不匹配！")

print(f"类别数: {nc}, 映射: {new_order}")

# 6. 交换权重和偏置
cls_conv.weight.data = cls_conv.weight.data[new_order, :, :, :]
if cls_conv.bias is not None:
    cls_conv.bias.data = cls_conv.bias.data[new_order]

# 7. 更新 checkpoint 中的类别名（可选，不影响功能）
if hasattr(model, 'names'):
    model.names = {i: f'class{i}' for i in range(nc)}

# 8. 保存修正后的 checkpoint（保持与 ultralytics 兼容的格式）
torch.save(checkpoint, FIXED_PT)
print(f"✅ 修正完成！已保存至：{FIXED_PT}")
