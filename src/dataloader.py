import os
import shutil
import kagglehub
from PIL import Image

def download_dataset(dataset_name="msjahid/bean-crop-disease-diagnosis-and-spatial-analysis"):
    """
    Descarga el dataset desde KaggleHub y lo copia al directorio data/raw.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    print("⬇️ Descargando dataset:", dataset_name)
    path = kagglehub.dataset_download(dataset_name)

    # Copia el contenido descargado al directorio destino
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(data_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"✅ Dataset descargado correctamente en: {data_dir}")
    return data_dir


def organize_dataset(base_dir="data/raw", output_dir="data/processed", subset="Classification"):
    """
    Organiza el dataset en carpetas train/val/test y convierte todas las imágenes a RGB.
    Si hay imágenes corruptas o con modos no válidos, las elimina.
    """
    # Detecta ruta correcta del dataset
    src_dir = os.path.join(base_dir, subset, subset) if subset == "Classification" else os.path.join(base_dir, subset)
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"No se encontró el directorio {src_dir}")

    os.makedirs(output_dir, exist_ok=True)
    splits = ["training", "validation", "test"]
    split_map = {"training": "train", "validation": "val", "test": "test"}

    adjusted_count = 0
    removed_count = 0
    total_images = 0

    for split in splits:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(output_dir, split_map[split])
        os.makedirs(dst_split, exist_ok=True)

        for cls in os.listdir(src_split):
            cls_src = os.path.join(src_split, cls)
            cls_dst = os.path.join(dst_split, cls)
            os.makedirs(cls_dst, exist_ok=True)

            for img_name in os.listdir(cls_src):
                src_img_path = os.path.join(cls_src, img_name)
                dst_img_path = os.path.join(cls_dst, img_name)
                total_images += 1

                try:
                    with Image.open(src_img_path) as img:
                        # Verificar canales
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                            adjusted_count += 1

                        # Asegurar tamaño válido
                        if img.width < 10 or img.height < 10:
                            raise ValueError("Dimensiones demasiado pequeñas")

                        img.save(dst_img_path)

                except Exception as e:
                    print(f"⚠️ Imagen inválida eliminada: {src_img_path} ({e})")
                    removed_count += 1
                    if os.path.exists(dst_img_path):
                        os.remove(dst_img_path)

    print("\n📂 Dataset organizado y limpiado correctamente")
    print(f"📊 Total de imágenes procesadas: {total_images}")
    print(f"🔧 Imágenes convertidas a RGB: {adjusted_count}")
    print(f"🗑️ Imágenes eliminadas por error: {removed_count}")


if __name__ == "__main__":
    download_dataset()
    organize_dataset()
