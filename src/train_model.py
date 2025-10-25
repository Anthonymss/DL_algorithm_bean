import os
import json
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(4)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 60

BASE_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(BASE_DIR, "train")
VAL_PATH   = os.path.join(BASE_DIR, "val")
TEST_PATH  = os.path.join(BASE_DIR, "test")

OUTPUT_DIR = os.path.join("outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_full.keras")

print(f"✅ Configuración: IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE}")

def count_images_in_split(path):
    total, classes = 0, []
    if not os.path.exists(path):
        return 0, classes
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            total += n
            classes.append((cls, n))
    return total, classes

train_count, train_classes = count_images_in_split(TRAIN_PATH)
val_count, val_classes = count_images_in_split(VAL_PATH)
test_count, test_classes = count_images_in_split(TEST_PATH)

print("\n📊 Conteo de imágenes:")
print(f"  Train: {train_count}")
for c,n in train_classes: print(f"    - {c}: {n}")
print(f"  Val: {val_count}")
for c,n in val_classes: print(f"    - {c}: {n}")
print(f"  Test: {test_count}")
for c,n in test_classes: print(f"    - {c}: {n}")

def _ensure_rgb_np(img_np):
    try:
        if img_np.ndim == 2:
            return np.stack([img_np]*3, axis=-1)
        if img_np.shape[-1] != 3:
            img_np = img_np[..., :3] if img_np.shape[-1] > 3 else np.repeat(img_np[..., np.newaxis], 3, axis=-1)
        return img_np.astype(np.uint8)
    except Exception:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

def ensure_rgb_tf(image, label):
    img_uint8 = tf.cast(image, tf.uint8)
    img_rgb = tf.numpy_function(_ensure_rgb_np, [img_uint8], tf.uint8)
    img_rgb.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return tf.cast(img_rgb, tf.float32), label


def load_split(path, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=shuffle
    )

print("\n📂 Cargando datasets...")
train_raw = load_split(TRAIN_PATH, shuffle=True)
val_raw   = load_split(VAL_PATH, shuffle=False)
test_raw  = load_split(TEST_PATH, shuffle=False)
class_names = train_raw.class_names
print("✅ Clases detectadas:", class_names)

def make_train_ds(ds):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_eval_ds(ds):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_train_ds(train_raw)
val_ds   = make_eval_ds(val_raw)
test_ds  = make_eval_ds(test_raw)


if train_count == 0:
    raise RuntimeError("No hay imágenes en TRAIN_PATH.")
steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = math.ceil(val_count / BATCH_SIZE)
test_steps = math.ceil(test_count / BATCH_SIZE)
print(f"\n🧮 steps_per_epoch={steps_per_epoch} | validation_steps={validation_steps}")

labels = [int(lab.numpy()) for _, lab in train_raw.unbatch()]
cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {int(i): float(w) for i, w in enumerate(cw)}
print("⚖️ Pesos de clase:", class_weights)


print("\n✅ Construyendo modelo EfficientNetB0...")
try:
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    print("✅ Pesos ImageNet cargados.")
except Exception as e:
    print(f"⚠️ Pesos ImageNet no cargados ({e}). Usando inicialización aleatoria.")
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(*IMG_SIZE, 3))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2)
])

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=METRICS)
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1)
checkpoint_best = ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"),
                                  monitor="val_loss", save_best_only=True, verbose=1)
checkpoint_full = ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=False, verbose=0)

initial_epoch = 0
epoch_tracker_path = os.path.join(OUTPUT_DIR, "last_epoch.txt")

if os.path.exists(CKPT_PATH):
    try:
        print(f"♻️ Reanudando desde checkpoint: {CKPT_PATH}")
        model = tf.keras.models.load_model(CKPT_PATH)
        if os.path.exists(epoch_tracker_path):
            with open(epoch_tracker_path, "r") as f:
                saved_epoch = int(f.read().strip())
            initial_epoch = saved_epoch + 1
            print(f"✅ Reanudando desde epoch {initial_epoch} (según last_epoch.txt)")
        else:
            print("⚠️ No se encontró registro de época previa, empezando desde 0.")
    except Exception as e:
        print(f"⚠️ Error al reanudar: {e}. Reiniciando desde cero.")
else:
    print("🆕 Entrenamiento nuevo (sin checkpoint previo).")

print("\n🚀 Entrenamiento Fase 1 (cabeza)...")
base_model.trainable = False

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
    class_weight=class_weights
)

with open(epoch_tracker_path, "w") as f:
    f.write(str(history.epoch[-1]))
print(f"💾 Epoch actual guardada: {history.epoch[-1]}")

print("\n🔧 Fine-tuning del modelo base...")
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    initial_epoch=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
    class_weight=class_weights
)

with open(epoch_tracker_path, "w") as f:
    f.write(str(history_fine.epoch[-1]))
print(f"💾 Epoch actual guardada: {history_fine.epoch[-1]}")

print("\n🧪 Evaluando modelo final...")
loss_all = model.evaluate(test_ds, steps=test_steps)
loss, acc = loss_all[:2]
print(f"✅ Accuracy final: {acc:.4f}")

y_true, y_pred = [], []
for images, labels in test_raw:
    x = preprocess_input(tf.cast(images, tf.float32))
    preds = model.predict(x, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "matriz_confusion.png")
plt.savefig(cm_path)
plt.close()

tflite_model_path = os.path.join(OUTPUT_DIR, "modelo_frijol_b0.tflite")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"📦 Modelo exportado: {tflite_model_path}")
except Exception as e:
    print(f"⚠️ Error exportando TFLite: {e}")

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_txt = classification_report(y_true, y_pred, target_names=class_names)

summary = {
    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": "EfficientNetB0 - CNN Frijol",
    "accuracy_test": float(acc),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted),
    "clases": class_names,
    "pesos_de_clase": class_weights,
    "ruta_checkpoint": CKPT_PATH,
    "ruta_tflite": tflite_model_path,
    "matriz_confusion_path": cm_path,
    "clasificacion": report_dict
}

with open(os.path.join(OUTPUT_DIR, "reporte_entrenamiento.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
print(f"\n📄 Reporte completo guardado en: {OUTPUT_DIR}")
