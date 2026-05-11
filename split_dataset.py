import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ================= 配置区域 =================
# 所有图片和标签的公共父目录
SRC_IMAGES = r"C:\Users\YQS\Desktop\dataset\images"   # 存放所有图片
SRC_LABELS = r"C:\Users\YQS\Desktop\dataset\labels"   # 存放所有标签

# 输出目录（会自动在里面创建 images/train, labels/val 等）
DST_ROOT = r"C:\Users\YQS\Desktop\dataset\splitted"

# 划分比例：训练:验证:测试
RATIOS = (0.7, 0.2,0.1)      # 7:2:1，可根据需求调整
RANDOM_SEED = 42               # 固定随机种子，保证可复现
# ===========================================

def main():
    # 收集所有标签文件
    label_files = list(Path(SRC_LABELS).glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"在 {SRC_LABELS} 中没有找到标签文件！")

    print(f"找到 {len(label_files)} 个标签文件")

    # 读取每个标签文件，构建多标签矩阵
    # 首先获取所有类别 ID（假设和你的 data.yaml 一致：0~5）
    all_class_ids = set()
    sample_labels = []       # 每个样本对应的类别列表 (list of lists)
    sample_names = []        # 对应的文件名（不含后缀）

    for lb_path in label_files:
        labels = []
        with open(lb_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    labels.append(class_id)
                    all_class_ids.add(class_id)
        # 去重
        labels = list(set(labels))
        sample_labels.append(labels)
        sample_names.append(lb_path.stem)

    num_classes = len(all_class_ids)
    print(f"检测到类别数: {num_classes}，类别 ID: {sorted(all_class_ids)}")

    # 转换为多标签二值矩阵 (nsamples x nclasses)
    import numpy as np
    y = np.zeros((len(sample_labels), num_classes), dtype=int)
    for i, lbls in enumerate(sample_labels):
        for c in lbls:
            y[i, c] = 1

    # 第一步：分出临时训练+验证，和测试集
    test_size = RATIOS[2] / (RATIOS[0] + RATIOS[1] + RATIOS[2])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(msss.split(np.zeros(len(y)), y))

    # 第二步：从 train_val_idx 中再分出验证集
    val_ratio_from_train = RATIOS[1] / (RATIOS[0] + RATIOS[1])
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio_from_train, random_state=RANDOM_SEED)
    train_idx, val_idx = next(msss2.split(np.zeros(len(train_val_idx)), y[train_val_idx]))
    train_idx = train_val_idx[train_idx]   # 映射回原始索引
    val_idx = train_val_idx[val_idx]

    # 输出每个集合的样本数
    print(f"划分完成 → 训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    # 创建目录结构
    splits = [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx)
    ]
    for split_name, _ in splits:
        (Path(DST_ROOT) / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (Path(DST_ROOT) / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # 复制文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for split_name, indices in splits:
        for idx in indices:
            name = sample_names[idx]
            # 找到对应的图片（尝试不同扩展名）
            img_path = None
            for ext in image_extensions:
                possible = Path(SRC_IMAGES) / (name + ext)
                if possible.exists():
                    img_path = possible
                    break
            if img_path is None:
                print(f"⚠️ 警告：找不到图片 {name}，跳过")
                continue

            # 标签文件
            lbl_path = Path(SRC_LABELS) / (name + ".txt")
            if not lbl_path.exists():
                print(f"⚠️ 警告：找不到标签 {name}.txt，跳过")
                continue

            # 目标路径
            dst_img = Path(DST_ROOT) / "images" / split_name / (name + img_path.suffix)
            dst_lbl = Path(DST_ROOT) / "labels" / split_name / (name + ".txt")

            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

    print("✅ 文件复制完成！")

    # 打印每个集合的类别分布
    print("\n========== 类别分布统计 ==========")
    for split_name, indices in splits:
        counter = defaultdict(int)
        for idx in indices:
            for c in sample_labels[idx]:
                counter[c] += 1
        total = len(indices)
        print(f"\n[{split_name}] 图片数: {total}")
        for cls_id in sorted(counter.keys()):
            print(f"  类别 {cls_id}: {counter[cls_id]} 个实例")

    print("\n✅ 全部完成！请检查数据划分是否符合预期。")

if __name__ == "__main__":
    main()