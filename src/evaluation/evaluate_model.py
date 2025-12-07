import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score
)

# CONFIG
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = "data/processed"
TEST_DIR = os.path.join(BASE_DIR, "test")

OUTPUT_ROOT = "outputs/evaluation"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

MODEL_PATH = "outputs/models/best_model.keras"
CLASSES_FILE = "outputs/models/classes.txt"

# CARGAR DATOS
def ensure_rgb(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label


def load_test_dataset():
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = ds_raw.class_names

    ds = ds_raw.map(ensure_rgb).prefetch(tf.data.AUTOTUNE)
    return ds_raw, ds, class_names

# EVALUACIÓN
def evaluate_model():
    print("\nCargando modelo...")
    model = load_model(MODEL_PATH)

    print("\nCargando dataset de test...")
    test_raw, test_ds, class_names = load_test_dataset()

    print("\nGenerando predicciones...")
    y_true, y_pred, y_prob = [], [], []
    images_list = []

    for batch_imgs, batch_labels in test_raw:
        batch_imgs_proc = preprocess_input(tf.cast(batch_imgs, tf.float32))
        pred = model.predict(batch_imgs_proc, verbose=0)

        y_true.extend(batch_labels.numpy())
        y_pred.extend(np.argmax(pred, axis=1))
        y_prob.extend(pred)
        images_list.extend(batch_imgs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    images = np.array(images_list)

    print("\nExtrayendo embeddings del penúltimo layer...")
    penultimate_layer = model.layers[-2]
    embed_model = tf.keras.Model(inputs=model.input, outputs=penultimate_layer.output)
    embeddings = embed_model.predict(preprocess_input(images.astype("float32")), verbose=0)

    # MÉTRICAS
    print("\nCalculando métricas...")
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro = recall_score(y_true, y_pred, average="macro")

    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        auc_macro = None

    # Reporte completo por clase
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    print("\nGuardando artefactos...")
    np.save(os.path.join(OUTPUT_ROOT, "y_true.npy"), y_true)
    np.save(os.path.join(OUTPUT_ROOT, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUTPUT_ROOT, "y_prob.npy"), y_prob)
    np.save(os.path.join(OUTPUT_ROOT, "images.npy"), images)
    np.save(os.path.join(OUTPUT_ROOT, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_ROOT, "confusion_matrix.npy"), cm)
    summary = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": MODEL_PATH,
        "num_test_images": len(y_true),
        "classes": class_names,

        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "kappa": float(kappa),
        "auc_macro": float(auc_macro) if auc_macro is not None else None,

        "classification_report": report,
        "paths": {
            "y_true": "y_true.npy",
            "y_pred": "y_pred.npy",
            "y_prob": "y_prob.npy",
            "images": "images.npy",
            "embeddings": "embeddings.npy",
            "confusion_matrix": "confusion_matrix.npy"
        }
    }

    with open(os.path.join(OUTPUT_ROOT, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\nEvaluación completada y guardada en outputs/models/evaluation/")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    evaluate_model()