import os
from pathlib import Path

LABELS_DIR = Path(r"C:\你的路径\labels")
IMAGES_DIR = Path(r"C:\你的路径\images")

print(f"开始检查标签...\n")

empty_txts = []
missing_labels = []
wrong_ids = []

for img_path in IMAGES_DIR.glob("*"):
    # 跳过非图片文件
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        continue

    txt_path = LABELS_DIR / (img_path.stem + ".txt")

    if not txt_path.exists():
        missing_labels.append(img_path.name)
        continue

    # 读取并检查内容
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        empty_txts.append(img_path.name)
    else:
        for line in lines:
            parts = line.split()
            if not parts: continue
            cls_id = int(parts[0])
            # 你的类别是 0-5，如果超出这个范围就是有问题的 ID
            if cls_id < 0 or cls_id > 5:
                wrong_ids.append((img_path.name, cls_id))

# 输出结果
if missing_labels:
    print(f"❌ 缺失标签文件 ({len(missing_labels)} 个):")
    for f in missing_labels[:5]: print(f"   - {f}")
    if len(missing_labels) > 5: print(f"   ... 等 {len(missing_labels) - 5} 个")

if empty_txts:
    print(f"\n⚠️ 空标签文件 ({len(empty_txts)} 个):")
    for f in empty_txts[:5]: print(f"   - {f}")
    if len(empty_txts) > 5: print(f"   ... 等 {len(empty_txts) - 5} 个")

if wrong_ids:
    print(f"\n🚫 类别ID超出范围 ({len(wrong_ids)} 处):")
    for f, cid in wrong_ids[:5]: print(f"   - {f} (ID: {cid})")

if not missing_labels and not empty_txts and not wrong_ids:
    print("✅ 检查通过！所有标签文件非空且ID在0-5之间。")