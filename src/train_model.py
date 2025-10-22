import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("✅ CPU limitada: intra=6 hilos, inter=4 hilos")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 60

# --- Rutas ---
BASE_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(BASE_DIR, "train")
VAL_PATH   = os.path.join(BASE_DIR, "val")
TEST_PATH  = os.path.join(BASE_DIR, "test")
MODEL_LOCAL = os.path.join("src", "efficientnetb0_notop.h5")
OUTPUT_DIR = os.path.join("outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CARGA DE DATOS
# ============================================================
for split in ["train", "val", "test"]:
    path = os.path.join(BASE_DIR, split)
    print(f"\n📂 {split.upper()}:")
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        print(f"  {cls}: {len(os.listdir(cls_path))} imágenes")

train_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_raw.class_names
print("✅ Clases detectadas:", class_names)

with open("classes.txt", "w") as f:
    f.write("\n".join(class_names))

# --- Asegurar RGB ---
def ensure_rgb(image, label):
    image = tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image
    return image, label

train_raw = train_raw.map(ensure_rgb)
val_raw = val_raw.map(ensure_rgb)
test_raw = test_raw.map(ensure_rgb)

# --- Preprocesamiento ---
def preprocess(image, label):
    image = preprocess_input(tf.cast(image, tf.float32))
    return image, label

train_ds = train_raw.map(preprocess).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds   = val_raw.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_raw.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)

# ============================================================
# MODELO
# ============================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
print("\n✅ Cargando modelo base EfficientNetB0 sin top...")

base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))

if os.path.exists(MODEL_LOCAL):
    try:
        base_model.load_weights(MODEL_LOCAL)
        print(f"✅ Pesos locales cargados desde {MODEL_LOCAL}")
    except Exception as e:
        print(f"⚠️ No se pudieron cargar los pesos locales ({e}), usando ImageNet.")
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
else:
    print("⚠️ Pesos locales no encontrados, usando ImageNet.")
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

base_model.trainable = False

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ============================================================
# CALLBACKS
# ============================================================
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1)

checkpoint_best = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

checkpoint_full = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "checkpoint_full.keras"),
    monitor="val_loss",
    save_best_only=False,
    verbose=1
)

# ============================================================
# PESOS DE CLASE
# ============================================================
labels = [int(y.numpy()) for _, y in train_raw.unbatch()]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("⚖️ Pesos de clase:", class_weights)

# ============================================================
# ENTRENAMIENTO - FASE 1
# ============================================================
print("\n🚀 Entrenando solo capa superior...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
    class_weight=class_weights
)

# ============================================================
# FINE TUNING
# ============================================================
print("\n🔧 Fine-tuning del modelo base...\n")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=[early_stop, reduce_lr, checkpoint_best, checkpoint_full],
    class_weight=class_weights
)

# ============================================================
# EVALUACIÓN FINAL
# ============================================================
print("\n🧪 Evaluando modelo en test...\n")
loss, acc = model.evaluate(test_ds)
print(f"✅ Accuracy final en test: {acc:.4f}")

model.save(os.path.join(OUTPUT_DIR, "modelo_frijol_final.keras"))
print("💾 Modelo completo guardado en outputs/models/modelo_frijol_final.keras")

# ============================================================
# GRÁFICAS Y REPORTES
# ============================================================
def plot_history(hist1, hist2):
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b-", label="Entrenamiento")
    plt.plot(epochs, val_acc, "r-", label="Validación")
    plt.title("Precisión (Accuracy)")
    plt.xlabel("Épocas")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-", label="Entrenamiento")
    plt.plot(epochs, val_loss, "r-", label="Validación")
    plt.title("Pérdida (Loss)")
    plt.xlabel("Épocas")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history, history_fine)

# --- Matriz de confusión ---
print("\n📊 Matriz de confusión:\n")
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de Confusión")
plt.show()

print("\n📋 Reporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
# --- 📄 Guardar reporte final ---
from datetime import datetime
import json

report_path = os.path.join(OUTPUT_DIR, "reporte_entrenamiento.txt")

# Obtener mejor accuracy de validación durante entrenamiento
best_val_acc_stage1 = max(history.history["val_accuracy"])
best_val_acc_stage2 = max(history_fine.history["val_accuracy"])
best_val_acc_total = max(best_val_acc_stage1, best_val_acc_stage2)

# Generar reporte de clasificación detallado
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Guardar matriz de confusión como imagen
cm_fig_path = os.path.join(OUTPUT_DIR, "matriz_confusion.png")
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig(cm_fig_path)
plt.close()

# Crear estructura de resumen
summary = {
    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": "EfficientNetB0 - Frijol CNN",
    "accuracy_test": float(acc),
    "loss_test": float(loss),
    "best_val_accuracy": float(best_val_acc_total),
    "epochs_totales": EPOCHS_STAGE1 + EPOCHS_STAGE2,
    "clases": class_names,
    "pesos_de_clase": class_weights,
    "ruta_modelo": os.path.join(OUTPUT_DIR, "modelo_frijol_v1.h5"),
    "matriz_confusion": cm.tolist(),
    "clasificacion": report_dict
}

# Guardar reporte como JSON
with open(os.path.join(OUTPUT_DIR, "reporte_entrenamiento.json"), "w", encoding="utf-8") as fjson:
    json.dump(summary, fjson, indent=4, ensure_ascii=False)

# Guardar resumen legible en TXT
with open(report_path, "w", encoding="utf-8") as f:
    f.write("📘 REPORTE DE ENTRENAMIENTO CNN - DETECCIÓN DE ENFERMEDADES DEL FRIJOL\n")
    f.write("="*70 + "\n\n")
    f.write(f"Fecha: {summary['fecha']}\n")
    f.write(f"Modelo: {summary['modelo']}\n\n")
    f.write(f"Mejor accuracy validación: {summary['best_val_accuracy']*100:.2f}%\n")
    f.write(f"Accuracy final test: {summary['accuracy_test']*100:.2f}%\n")
    f.write(f"Loss en test: {summary['loss_test']:.4f}\n\n")
    f.write("Clases detectadas:\n")
    for c in class_names:
        f.write(f" - {c}\n")
    f.write("\nMatriz de confusión (guardada como imagen): matriz_confusion.png\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))
    f.write("\n\nPesos de clase:\n")
    f.write(str(class_weights))
    f.write("\n\nModelo guardado en:\n")
    f.write(summary["ruta_modelo"])
    f.write("\n\n✅ Fin del reporte.\n")

print(f"\n📄 Reporte completo guardado en: {report_path}")
print(f"🧾 Versión JSON guardada en: {os.path.join(OUTPUT_DIR, 'reporte_entrenamiento.json')}")
