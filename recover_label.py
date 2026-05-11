# flip_classes_for_v2.py
import os
from pathlib import Path

LABELS_DIR = Path("C:/Users/YQS/Desktop/yolo/dataset/all_labels")   # 存放所有 txt 的文件夹
MAPPING = {
    0: 1,   # blue3 -> blue1
    1: 0,   # blue1 -> blue3
    3: 4,   # red3  -> red1
    4: 3,   # red1  -> red3
    # 2,5 不变
}

for txt_path in LABELS_DIR.glob("*.txt"):
    new_lines = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id in MAPPING:
                cls_id = MAPPING[cls_id]
            new_lines.append(f"{cls_id} " + " ".join(parts[1:]))
    with open(txt_path, 'w') as f:
        f.write("\n".join(new_lines) + "\n")

print("✅ 所有标签 class id 已反转完成！现在可与正确顺序的 data.yaml 配合使用。")