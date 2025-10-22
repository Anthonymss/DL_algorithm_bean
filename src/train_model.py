# train_model_b0_improved.py
import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =========================
# CONFIG
# =========================
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(4)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 60

BASE_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(BASE_DIR, "train")
VAL_PATH = os.path.join(BASE_DIR, "val")
TEST_PATH = os.path.join(BASE_DIR, "test")

MODEL_LOCAL = os.path.join("src", "efficientnetb0_notop.h5")  # opcional
OUTPUT_DIR = os.path.join("outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Configuración: IMG_SIZE=", IMG_SIZE, " BATCH_SIZE=", BATCH_SIZE)

# =========================
# Utils: robust convert to RGB using PIL (used via tf.numpy_function)
# =========================
def _ensure_rgb_np(img_np):
    """
    img_np: numpy array HxWxC (uint8)
    returns: numpy array HxWx3 (uint8)
    """
    try:
        # If grayscale (H,W) or (H,W,1)
        if img_np.ndim == 2:
            img = Image.fromarray(img_np).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        if img_np.shape[-1] == 1:
            img = Image.fromarray(img_np.squeeze(-1)).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        # If >3 channels (CMYK, RGBA...), convert via PIL to RGB
        if img_np.shape[-1] != 3:
            img = Image.fromarray(img_np).convert("RGB")
            return np.asarray(img, dtype=np.uint8)

        # already RGB
        return img_np.astype(np.uint8)
    except Exception:
        # In case of any error, return a small black RGB image to avoid crashing
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

def ensure_rgb_tf(image, label):
    """
    Wrapper for tf.data pipeline. Receives image tensor HxWxC (uint8 or float),
    returns image shaped (IMG_SIZE[0], IMG_SIZE[1], 3), dtype uint8 -> cast later.
    """
    # Ensure dtype uint8 for PIL conversion
    # image may be float32 in [0,255] or uint8; convert to uint8
    img_uint8 = tf.cast(image, tf.uint8)

    # Use numpy_function to call PIL-based converter (robusto ante modos raros)
    img_rgb = tf.numpy_function(_ensure_rgb_np, [img_uint8], tf.uint8)

    # Set static shape so TensorFlow knows dimensions (important)
    img_rgb.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])

    # Convert back to float32 for model preprocess later
    img_rgb = tf.cast(img_rgb, tf.float32)
    return img_rgb, label

# =========================
# Dataset loading
# =========================
# Print counts (informativo)
for split in ["train", "val", "test"]:
    p = os.path.join(BASE_DIR, split)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Directorio no encontrado: {p}")
    print(f"\n📂 {split.upper()}:")
    for cls in sorted(os.listdir(p)):
        cls_p = os.path.join(p, cls)
        if os.path.isdir(cls_p):
            print(f"  {cls}: {len(os.listdir(cls_p))} imágenes")

# Force color_mode="rgb" to help but still sanitize afterwards
train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=True
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=False
)
test_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=False
)

class_names = train_raw.class_names
print("✅ Clases detectadas:", class_names)
with open("classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

# =========================
# Ensure RGB robustly (element-wise)
# We apply per-example ensure_rgb_tf. It is slightly slower but robust.
# =========================
# map over each example (not batch) -> use unbatch() -> map -> batch again
def make_dataset_safe(ds, shuffle_after=False):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_after:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset_safe(train_raw, shuffle_after=True)
val_ds = make_dataset_safe(val_raw, shuffle_after=False)
test_ds = make_dataset_safe(test_raw, shuffle_after=False)

# =========================
# Compute class weights (safe)
# Note: unbatch() may be costly in memory for huge datasets. If so, compute externally.
# =========================
labels_list = []
for _, batch_labels in train_raw.unbatch().map(lambda im, lab: (im, lab)):
    # this loop won't execute because map returns dataset: use unbatch() iteration
    break

# Simpler robust extraction:
labels = []
for _, lab in train_raw.unbatch():
    labels.append(int(lab.numpy()))
labels = np.array(labels)
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {int(i): float(w) for i, w in enumerate(class_weights)}
print("⚖️ Pesos de clase:", class_weights)

# =========================
# Data augmentation (applied inside model so augmentations are part of graph)
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
], name="data_augmentation")

# =========================
# Build EfficientNetB0 model
# =========================
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)

# Try to load base model with imagenet weights; fallback to weights=None if incompatible
print("\n✅ Construyendo EfficientNetB0 base...")
try:
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    print("✅ Cargados pesos ImageNet para EfficientNetB0")
except Exception as e:
    print(f"⚠️ No se pudieron cargar pesos imagenet ({e}). Usando pesos aleatorios (weights=None).")
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model.trainable = False  # fase 1: congelado
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =========================
# Callbacks
# =========================
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1)
checkpoint_best = ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1)
checkpoint_full = ModelCheckpoint(os.path.join(OUTPUT_DIR, "checkpoint_full.keras"), monitor="val_loss", save_best_only=False, verbose=0)

# =========================
# Train - Stage 1 (head)
# =========================
print("\n🚀 Entrenamiento fase 1 (solo cabeza)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1,
                    callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
                    class_weight=class_weights)

# =========================
# Fine-tune - Stage 2
# =========================
print("\n🔧 Fine-tuning: desbloqueando últimas capas del base_model...")
base_model.trainable = True
# desbloquea últimas N capas (ajustar si hace falta)
fine_tune_at = len(base_model.layers) - 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2,
                         callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
                         class_weight=class_weights)

# =========================
# Evaluate & Save
# =========================
print("\n🧪 Evaluando en test set...")
loss, acc = model.evaluate(test_ds)
print(f"Accuracy test: {acc:.4f}  loss: {loss:.4f}")

final_model_path = os.path.join(OUTPUT_DIR, "modelo_frijol_b0.keras")
model.save(final_model_path)
print("Modelo guardado en:", final_model_path)

# =========================
# Reports: confusion matrix, classification report
# =========================
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta")
plt.title("Matriz de Confusión")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "matriz_confusion.png")
plt.savefig(cm_path)
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_txt = classification_report(y_true, y_pred, target_names=class_names)

summary = {
    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": "EfficientNetB0 - mejorado",
    "accuracy_test": float(acc),
    "loss_test": float(loss),
    "best_val_accuracy_stage1": max(history.history.get("val_accuracy", [0.0])),
    "best_val_accuracy_stage2": max(history_fine.history.get("val_accuracy", [0.0])),
    "clases": class_names,
    "pesos_de_clase": class_weights,
    "ruta_modelo": final_model_path,
    "matriz_confusion_path": cm_path,
    "clasificacion": report
}

with open(os.path.join(OUTPUT_DIR, "reporte_entrenamiento.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "reporte_entrenamiento.txt"), "w", encoding="utf-8") as f:
    f.write("REPORTE ENTRENAMIENTO - EfficientNetB0 (mejorado)\n\n")
    f.write(json.dumps(summary, indent=4, ensure_ascii=False))
    f.write("\n\nClassification Report:\n")
    f.write(report_txt)

print("Reportes y artefactos guardados en:", OUTPUT_DIR)
