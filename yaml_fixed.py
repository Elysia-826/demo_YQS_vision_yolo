import os

import torch
from ultralytics import YOLO

os.environ['ULTRALYTICS_OFFLINE'] = '1'
# ================== 修改这里 ==================
WRONG_PT = r"C:\Users\YQS\Desktop\yolo\runs\temp_seed\v1\weights\best.pt"     # 旧模型
FIXED_PT = r"C:\Users\YQS\Desktop\yolo\runs\temp_seed\v1\weights\best_fixed.pt" # 修正后模型

# 错误→正确的映射（错误索引对应的正确索引）
new_order = [1, 0, 2, 4, 3, 5]  # 已按你的要求配置好
# =============================================

print("加载旧模型...")
model = YOLO(WRONG_PT)

# 定位分类卷积层（YOLOv11/v8 检测头最后一个分类分支）
detect = model.model.model[-1]               # 检测头模块
cls_conv = detect.cv2[-1]                    # cv2 中最后一个卷积负责分类

# 验证通道数
nc = cls_conv.out_channels
if len(new_order) != nc:
    raise ValueError(f"new_order 长度 ({len(new_order)}) 与类别数 ({nc}) 不匹配！")

print(f"类别数: {nc}，映射: {new_order}")

# 交换权重与偏置
cls_conv.weight.data = cls_conv.weight.data[new_order, :, :, :]
if cls_conv.bias is not None:
    cls_conv.bias.data = cls_conv.bias.data[new_order]

# 保存
model.save(FIXED_PT)
print(f"✅ 修正完成，已保存至: {FIXED_PT}")

#验证
m_old = YOLO(WRONG_PT)
m_new = YOLO(FIXED_PT)
# 印出分类卷积权重的前几个通道，看是否是交换后的
w_old = m_old.model.model[-1].cv2[-1].weight.data
w_new = m_new.model.model[-1].cv2[-1].weight.data
print("通道0与通道1是否交换：", torch.allclose(w_old[0], w_new[1]) and torch.allclose(w_old[1], w_new[0]))
print("通道3与通道4是否交换：", torch.allclose(w_old[3], w_new[4]) and torch.allclose(w_old[4], w_new[3]))
