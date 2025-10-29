# src/train_model.py
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

np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(4)

#PARAMETROS
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

LOCAL_EFF_WEIGHTS = os.path.join("src", "efficientnetb0_notop.h5")
USE_MIXUP = False
MIXUP_ALPHA = 0.2

print(f"Configuración: IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE} | MIXUP={USE_MIXUP}")

#CAntidad de clases e imágenes
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

print("\nConteo de imágenes:")
print(f"  Train: {train_count}")
for c,n in train_classes: print(f"    - {c}: {n}")
print(f"  Val: {val_count}")
for c,n in val_classes: print(f"    - {c}: {n}")
print(f"  Test: {test_count}")
for c,n in test_classes: print(f"    - {c}: {n}")

if train_count == 0:
    raise RuntimeError("No hay imágenes en TRAIN_PATH. Comprueba rutas.")

def _ensure_rgb_np(img_np):
    try:
        if img_np.ndim == 2:
            return np.stack([img_np]*3, axis=-1)
        if img_np.shape[-1] == 1:
            return np.repeat(img_np, 3, axis=-1)
        if img_np.shape[-1] == 4:
            return img_np[..., :3]
        if img_np.shape[-1] == 3:
            return img_np
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    except Exception:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

def ensure_rgb_tf(image, label):
    img_uint8 = tf.cast(image, tf.uint8)
    img_rgb = tf.numpy_function(_ensure_rgb_np, [img_uint8], tf.uint8)
    img_rgb.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return tf.cast(img_rgb, tf.float32), label

#Carga de datasets
def load_split(path, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=shuffle
    )

print("\nCargando datasets...")
train_raw = load_split(TRAIN_PATH, shuffle=True)
val_raw   = load_split(VAL_PATH, shuffle=False)
test_raw  = load_split(TEST_PATH, shuffle=False)
class_names = train_raw.class_names
print("Clases detectadas:", class_names)
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

#Data Augmentation y parametrización de pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.35),
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomCrop(IMG_SIZE[0], IMG_SIZE[1]),
    tf.keras.layers.GaussianNoise(0.03),
], name="data_augmentation")

def preprocess_batch(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y
def make_train_ds(ds, use_mixup=False):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    if use_mixup:
        ds = ds.map(lambda x, y: mixup_map(x, y, alpha=MIXUP_ALPHA), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_eval_ds(ds):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def mixup_map(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    def no_mix():
        return x, tf.one_hot(y, depth=len(class_names))
    def do_mix():
        lam = np.random.beta(alpha, alpha)
        idx = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)
        mixed_x = lam * x + (1 - lam) * x2
        y_a = tf.one_hot(y, depth=len(class_names))
        y_b = tf.one_hot(y2, depth=len(class_names))
        mixed_y = lam * y_a + (1 - lam) * y_b
        return mixed_x, mixed_y
    return tf.cond(tf.less(batch_size, 2), lambda: no_mix(), lambda: do_mix())

# Preparar datasets finales
train_ds = make_train_ds(train_raw, use_mixup=USE_MIXUP)
val_ds   = make_eval_ds(val_raw)
test_ds  = make_eval_ds(test_raw)

steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = math.ceil(val_count / BATCH_SIZE)
test_steps = math.ceil(test_count / BATCH_SIZE)
print(f"\nsteps_per_epoch={steps_per_epoch} | validation_steps={validation_steps} | test_steps={test_steps}")

# Verificación rápida de un batch y rango de valores
for images, labels in train_ds.take(1):
    mn = tf.reduce_min(images).numpy()
    mx = tf.reduce_max(images).numpy()
    print(f"Check batch - shape: {images.shape} | range: {mn:.4f} .. {mx:.4f}")
    break

labels = [int(lab.numpy()) for _, lab in train_raw.unbatch()]
cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {int(i): float(w) for i, w in enumerate(cw)}
print("Pesos de clase:", class_weights)

#Construcción del modelo
print("\nConstruyendo modelo EfficientNetB0...")

try:
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(*IMG_SIZE, 3))
    if os.path.exists(LOCAL_EFF_WEIGHTS) and os.path.getsize(LOCAL_EFF_WEIGHTS) > 0:
        base_model.load_weights(LOCAL_EFF_WEIGHTS)
        print("Pesos locales EfficientNetB0 cargados correctamente")
    else:
        raise FileNotFoundError("Archivo local de pesos no encontrado o vacío.")
except Exception as e_local:
    print(f"No se pudieron cargar los pesos locales ({e_local}), intentando ImageNet...")
    try:
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        print("Pesos ImageNet cargados correctamente.")
    except Exception as e_img:
        print(f"No se pudieron cargar pesos de ImageNet ({e_img}). Se entrenará desde cero.")
        base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(*IMG_SIZE, 3))

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(class_names),
                                activation="softmax",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
model = tf.keras.Model(inputs, outputs)

METRICS = ["accuracy"]

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=METRICS)
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
checkpoint_best = ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"),
                                  monitor="val_loss", save_best_only=True, verbose=1)
checkpoint_full = ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=False, verbose=0)

# Reanudación desde checkpoint si existe
initial_epoch = 0
epoch_tracker_path = os.path.join(OUTPUT_DIR, "last_epoch.txt")
if os.path.exists(CKPT_PATH):
    try:
        print(f"Reanudando desde checkpoint: {CKPT_PATH}")
        model = tf.keras.models.load_model(CKPT_PATH)
        if os.path.exists(epoch_tracker_path):
            with open(epoch_tracker_path, "r") as f:
                saved_epoch = int(f.read().strip())
            initial_epoch = saved_epoch + 1
            print(f"Reanudando desde epoch {initial_epoch} (según last_epoch.txt)")
        else:
            print("No se encontró last_epoch.txt, empezando desde 0.")
    except Exception as e:
        print(f"Error al reanudar: {e}. Reiniciando desde cero.")

# Entrenamiento: Stage 1 (solo cabeza)
print("\nEntrenamiento Fase 1 (cabeza)...")
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
print(f"Epoch actual guardada: {history.epoch[-1]}")

# Fine-tuning, Stage 2 (descongelar parte del base model)
print("\nFine-tuning del modelo base...")
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch * 10,
    decay_rate=0.8,
    staircase=True
)
optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-4)


model.compile(optimizer=optimizer_fine,
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
print(f"Epoch actual guardada: {history_fine.epoch[-1]}")

# Evaluación final
print("\nEvaluando modelo final...")
loss_all = model.evaluate(test_ds, steps=test_steps)
loss, acc = loss_all[:2]
print(f"Accuracy final: {acc:.4f}")

# predicciones para reporte
y_true, y_pred = [], []
for images, labels in test_raw:
    x = tf.cast(images, tf.float32)
    x = preprocess_input(x)
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
    print(f"Modelo exportado: {tflite_model_path}")
except Exception as e:
    print(f"Error exportando TFLite: {e}")

# reporte final
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_txt = classification_report(y_true, y_pred, target_names=class_names)

summary = {
    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": "EfficientNetB0 - CNN Frijol (fine-tuned)",
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

print(f"\nReporte completo guardado en: {OUTPUT_DIR}")
