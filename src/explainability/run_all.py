import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

import cv2
import numpy as np
from PIL import Image
import asyncio
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.explainability.gradcam import generate_gradcam
from src.explainability.gradcam_pp import generate_gradcam_pp
from src.explainability.occlusion import occlusion_sensitivity

IMAGE_PATH = "data/processed/test/als/1619348033216.jpg"
MODEL_PATH = "outputs/models/best_model.keras"
CLASSES_PATH = "outputs/models/classes.txt"

SAVE_RESULTS = False   # guardar solo si es True
IMG_SIZE = (224, 224)


def load_class_names():
    with open(CLASSES_PATH, "r") as f:
        return [x.strip() for x in f.readlines()]


def predict_class(model, image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    return np.argmax(preds), preds[0]


def display_image(arr, title="Resultado"):
    try:
        cv2.imshow(title, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"\n[{title}] Vista ASCII aproximada:")
        small = cv2.resize(arr, (40, 20))
        for row in small:
            line = "".join(" .:-=+*#%@"[val // 25] for val in row[:,0])
            print(line)

async def run_explainers(model, image, class_idx, save_dir):

    tasks = [
        asyncio.to_thread(generate_gradcam, image, model, class_idx, save_dir, SAVE_RESULTS),
        asyncio.to_thread(generate_gradcam_pp, image, model, class_idx, save_dir, SAVE_RESULTS),
        asyncio.to_thread(occlusion_sensitivity, image, model, class_idx, 32, 16, save_dir, SAVE_RESULTS)
    ]

    gradcam_img, gradcam_pp_img, occlusion_img = await asyncio.gather(*tasks)

    print("\n✔ Procesamiento terminado. Mostrando imágenes...")

    display_image(gradcam_img, "Grad-CAM")
    display_image(gradcam_pp_img, "Grad-CAM++")
    display_image(occlusion_img, "Occlusion Sensitivity")


async def main():

    image_name = os.path.basename(IMAGE_PATH).split(".")[0]
    save_dir = f"outputs/explainability/{image_name}" if SAVE_RESULTS else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("🔄 Cargando modelo...")
    model = load_model(MODEL_PATH, compile=False)

    image = Image.open(IMAGE_PATH).convert("RGB")
    class_idx, probs = predict_class(model, image)
    classes = load_class_names()

    print("\nClase predicha:", classes[class_idx])
    print("Probabilidades:")
    for i, p in enumerate(probs):
        print(f" - {classes[i]}: {p:.4f}")

    print("\nEjecutando explicabilidad de forma asíncrona...")

    await run_explainers(model, image, class_idx, save_dir)

    print("\n🎉 Finalizado.")


if __name__ == "__main__":
    asyncio.run(main())
