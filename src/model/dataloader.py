import os
import shutil
import kagglehub
from collections import defaultdict

def download_dataset(dataset_name="msjahid/bean-crop-disease-diagnosis-and-spatial-analysis"):
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    path = kagglehub.dataset_download(dataset_name)
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"Dataset descargado correctamente en: {data_dir}")
    return data_dir

def organize_dataset(base_dir="data/raw", output_dir="data/processed", subset="Classification"):
    src_dir = os.path.join(base_dir, subset, subset) if subset == "Classification" else os.path.join(base_dir, subset)
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"No se encontró el directorio {src_dir}")

    os.makedirs(output_dir, exist_ok=True)
    splits = ["training", "validation", "test"]
    split_map = {"training": "train", "validation": "val", "test": "test"}
    total_count = defaultdict(int)
    per_class_count = {split_map[s]: defaultdict(int) for s in splits}

    for split in splits:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(output_dir, split_map[split])
        os.makedirs(dst_split, exist_ok=True)
        for cls in os.listdir(src_split):
            cls_src = os.path.join(src_split, cls)
            cls_dst = os.path.join(dst_split, cls)
            os.makedirs(cls_dst, exist_ok=True)
            imgs = [img for img in os.listdir(cls_src) if os.path.isfile(os.path.join(cls_src, img))]
            total_count[split_map[split]] += len(imgs)
            per_class_count[split_map[split]][cls] = len(imgs)
            for img_name in imgs:
                src_img_path = os.path.join(cls_src, img_name)
                dst_img_path = os.path.join(cls_dst, img_name)
                if not os.path.exists(dst_img_path):
                    shutil.copy2(src_img_path, dst_img_path)

    total_images = sum(total_count.values())
    print("\nDataset organizado correctamente en:", output_dir)
    print("Resumen de distribución:")
    for split, count in total_count.items():
        pct = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  - {split.upper():6}: {count:5d} images ({pct:.2f}%)")
        for cls, c in per_class_count[split].items():
            print(f"      • {cls}: {c} images")
    print(f"\nTotal general: {total_images} images distributed in {len(per_class_count['train'])} classes.")
    return {
        "total": total_images,
        "train": total_count["train"],
        "val": total_count["val"],
        "test": total_count["test"],
        "porcentajes": {
            "train": (total_count["train"] / total_images) * 100,
            "val": (total_count["val"] / total_images) * 100,
            "test": (total_count["test"] / total_images) * 100,
        }
    }


if __name__ == "__main__":
    dataset_path = download_dataset()
    stats = organize_dataset()
    print("\nEstadísticas globales:", stats)
