import torch
import os

# ================== 配置区 ==================
WRONG_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best.pt"   # 按你实际路径修改
FIXED_PT = r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best_fixed.pt"

new_order = [1, 0, 2, 4, 3, 5]   # 错误索引 -> 正确索引
# ===========================================

# 检查文件
if not os.path.exists(WRONG_PT):
    raise FileNotFoundError(f"找不到模型文件：{WRONG_PT}")

print("正在加载模型（完全离线）...")
checkpoint = torch.load(WRONG_PT, map_location='cpu', weights_only=False)
model = checkpoint['model']          # Ultralytics 保存的完整模型

# 检测头
detect = model.model[-1]

# 获取分类卷积层
def get_cls_conv(detect, num_classes):
    if hasattr(detect, 'cv2'):
        branch = detect.cv2[-1]
        if isinstance(branch, torch.nn.Sequential):
            for module in reversed(list(branch.children())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
        elif isinstance(branch, torch.nn.Conv2d):
            return branch

    for module in detect.modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels == num_classes:
            return module
    return None

cls_conv = get_cls_conv(detect, len(new_order))
if cls_conv is None:
    raise RuntimeError("无法定位分类卷积层")

nc = cls_conv.out_channels
if len(new_order) != nc:
    raise ValueError(f"映射长度 ({len(new_order)}) 与类别数 ({nc}) 不匹配！")

print(f"类别数: {nc}, 映射: {new_order}")

# 交换权重与偏置
cls_conv.weight.data = cls_conv.weight.data[new_order, :, :, :]
if cls_conv.bias is not None:
    cls_conv.bias.data = cls_conv.bias.data[new_order]

# 保存
torch.save(checkpoint, FIXED_PT)
print(f"✅ 修正完成，已保存至：{FIXED_PT}")

old = torch.load(WRONG_PT, map_location='cpu', weights_only=False)['model']
new = torch.load(FIXED_PT, map_location='cpu', weights_only=False)['model']

old_weight = get_cls_conv(old.model[-1], 6).weight.data
new_weight = get_cls_conv(new.model[-1], 6).weight.data

print("通道0与1是否互换：", torch.allclose(old_weight[0], new_weight[1]))
print("通道1与0是否互换：", torch.allclose(old_weight[1], new_weight[0]))
print("通道3与4是否互换：", torch.allclose(old_weight[3], new_weight[4]))
print("通道4与3是否互换：", torch.allclose(old_weight[4], new_weight[3]))