import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


MODEL_PATH = "outputs/models/best_model.keras"
CLASSES_PATH = "outputs/models/classes.txt"

IMAGE_PATH = "data/processed/test/als/1619348033216.jpg"

IMG_SIZE = (224, 224)

def load_class_names(path):
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines()]


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)

    arr = np.array(img_resized).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def predict_image(model, image_array):
    preds = model.predict(image_array)[0]
    class_idx = int(np.argmax(preds))
    return class_idx, preds


def main():

    print(f"\n🔄 Cargando modelo desde: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)

    print(f"🖼️  Cargando imagen: {IMAGE_PATH}")
    pil_img, image_array = preprocess_image(IMAGE_PATH)

    class_names = load_class_names(CLASSES_PATH)
    class_idx, probs = predict_image(model, image_array)

    print("\n📌 RESULTADO")
    print(f"Clase predicha: {class_names[class_idx]}")
    print("\n📊 Probabilidades:")

    for i, p in enumerate(probs):
        print(f" - {class_names[i]}: {p:.4f}")

    print("\n✔ Test finalizado.\n")


if __name__ == "__main__":
    main()
