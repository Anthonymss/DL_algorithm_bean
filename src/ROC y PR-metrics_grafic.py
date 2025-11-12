import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, accuracy_score
from tensorflow.keras.applications.efficientnet import preprocess_input

# ========= CONFIG =========
BASE_DIR = r"C:\Users\Anthony_mss\Desktop\DL_Frejol"
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models-martes-v1", "best_model.keras")
TEST_DIR  = os.path.join(BASE_DIR, "data", "processed", "test")
IMG_SIZE  = (224, 224)
BATCH     = 32

# ========= CARGA MODELO =========
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado.")

# ========= DATASET (RGB + resize hecho por el loader) =========
test_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,      # <-- ya redimensiona a 224x224
    batch_size=BATCH,
    color_mode="rgb",         # <-- fuerza 3 canales
    shuffle=False
)

# Usa el orden REAL de clases detectado (coincidirá con el entrenamiento si así estaban las carpetas)
CLASS_NAMES = test_raw.class_names
print("Clases detectadas:", CLASS_NAMES)

# Preprocesamiento idéntico al entrenamiento: preprocess_input de EfficientNet
test_ds = (test_raw
           .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(tf.data.AUTOTUNE))

# ========= PREDICCIONES =========
y_true_list, y_prob_list = [], []
for x_batch, y_batch in test_ds:
    probs = model.predict(x_batch, verbose=0)
    y_prob_list.append(probs)
    y_true_list.append(y_batch.numpy())

y_true = np.concatenate(y_true_list, axis=0)    # (N,)
y_prob = np.concatenate(y_prob_list, axis=0)    # (N, num_classes)
y_pred = np.argmax(y_prob, axis=1)

# Chequeo rápido: debe estar ~0.91 si todo está bien
print("Accuracy (check):", accuracy_score(y_true, y_pred))

# ========= AUC-ROC macro (ovr) =========
auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
print("AUC-ROC macro:", auc_macro)

# ========= AUC-ROC y AUC-PR por clase =========
y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))

auc_roc_per_class = {}
auc_pr_per_class  = {}

for i, cls in enumerate(CLASS_NAMES):
    # ROC AUC (one-vs-rest)
    auc_roc_per_class[cls] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
    # PR AUC
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
    auc_pr_per_class[cls] = auc(recall, precision)

print("\n=== AUC ROC por clase ===")
for c in CLASS_NAMES:
    print(f"{c}: {auc_roc_per_class[c]:.3f}")

print("\n=== AUC PR por clase ===")
for c in CLASS_NAMES:
    print(f"{c}: {auc_pr_per_class[c]:{'.3f'}}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
