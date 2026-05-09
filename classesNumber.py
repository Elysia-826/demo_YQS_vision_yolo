import torch
ckpt = torch.load(r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v1\weights\best.pt", map_location='cpu', weights_only=False)
model = ckpt['model']
detect = model.model[-1]
# 提取分类卷积最后一个 Conv2d 的输出通道
branch = detect.cv2[-1]
if isinstance(branch, torch.nn.Sequential):
    cls_conv = [m for m in branch.modules() if isinstance(m, torch.nn.Conv2d)][-1]
else:
    cls_conv = branch
print(cls_conv.out_channels)  # 应该为 6