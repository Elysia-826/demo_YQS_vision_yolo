import torch
import os

# ================== 配置 ==================
WRONG_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best.pt"
FIXED_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best_fixed.pt"

# 你的映射：旧索引 → 新索引（只涉及你实际用到的 6 个类别）
new_order_6 = [1, 0, 2, 4, 3, 5]   # 长度 6
# =========================================

# 1. 加载 checkpoint
if not os.path.exists(WRONG_PT):
    raise FileNotFoundError(f"模型不存在: {WRONG_PT}")

print("加载模型...")
ckpt = torch.load(WRONG_PT, map_location='cpu', weights_only=False)
model = ckpt['model']

# 2. 定位分类卷积层
detect = model.model[-1]
branch = detect.cv2[-1]
if isinstance(branch, torch.nn.Sequential):
    cls_conv = None
    for m in reversed(list(branch.children())):
        if isinstance(m, torch.nn.Conv2d):
            cls_conv = m
            break
else:
    cls_conv = branch

if cls_conv is None:
    raise RuntimeError("未找到分类卷积层")

out_channels = cls_conv.out_channels
print(f"模型输出通道数: {out_channels}")

# 3. 构造完整的交换索引 (前6个按映射表，后面保持不变)
if out_channels < len(new_order_6):
    raise ValueError(f"模型的输出通道数 ({out_channels}) 少于映射表长度 ({len(new_order_6)})")

full_order = list(range(out_channels))
for old_idx, new_idx in enumerate(new_order_6):
    full_order[old_idx] = new_idx

print(f"完整映射 (前6个交换): {full_order[:len(new_order_6)]}")

# 4. 交换权重和偏置
cls_conv.weight.data = cls_conv.weight.data[full_order, :, :, :]
if cls_conv.bias is not None:
    cls_conv.bias.data = cls_conv.bias.data[full_order]

# 5. 保存
torch.save(ckpt, FIXED_PT)
print(f"✅ 修正完成，保存至: {FIXED_PT}")

