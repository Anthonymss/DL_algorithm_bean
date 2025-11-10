import os
import json
import math
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    accuracy_score, cohen_kappa_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# PARÁMETROS GLOBALES
duration = None
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

LOCAL_EFF_WEIGHTS = os.path.join("src", "efficientnetb0_notop.h5")
USE_MIXUP = False
MIXUP_ALPHA = 0.2


# === NUEVO: Callback personalizado para guardar métricas de cada época ===
class EpochMetricsLogger(Callback):
    def __init__(self, filepath, val_data=None):
        super().__init__()
        self.filepath = filepath
        self.val_data = val_data
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "accuracy", "val_accuracy", "loss", "val_loss", "val_f1"])
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # Calcular F1 sobre conjunto de validación
        val_f1 = None
        if self.val_data is not None:
            y_true, y_pred = [], []
            for x_batch, y_batch in self.val_data.take(5):  # Toma 5 lotes para estimar más rápido
                preds = self.model.predict(x_batch, verbose=0)
                y_true.extend(y_batch.numpy())
                y_pred.extend(np.argmax(preds, axis=1))
            val_f1 = f1_score(y_true, y_pred, average="macro")

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, lr, acc, val_acc, loss, val_loss, val_f1])

        logs["val_f1"] = val_f1
        self.history.append(logs)



# === FUNCIONES AUXILIARES ORIGINALES ===
def count_images_in_split(path):
    total, classes = 0, []
    if not os.path.exists(path):
        return 0, classes
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if os.path.isfile(os.path.join(cls_path, f))])
            total += n
            classes.append((cls, n))
    return total, classes


def _ensure_rgb_np(img_np):
    try:
        if img_np.ndim == 2:
            return np.stack([img_np] * 3, axis=-1)
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


def load_split(path, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=shuffle
    )


def preprocess_batch(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y


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


def make_train_ds(ds, use_mixup=False):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    if use_mixup:
        ds = ds.map(lambda x, y: mixup_map(x, y, alpha=MIXUP_ALPHA),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_ds(ds):
    ds = ds.unbatch()
    ds = ds.map(ensure_rgb_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# FUNCIÓN PRINCIPAL
def main():
    start_time = datetime.now()
    global class_names, data_augmentation

    print(f"Configuración: IMG_SIZE={IMG_SIZE} | BATCH_SIZE={BATCH_SIZE} | MIXUP={USE_MIXUP}")

    train_count, train_classes = count_images_in_split(TRAIN_PATH)
    val_count, val_classes = count_images_in_split(VAL_PATH)
    test_count, test_classes = count_images_in_split(TEST_PATH)

    print("\nConteo de imágenes:")
    print(f"  Train: {train_count}")
    for c, n in train_classes: print(f"    - {c}: {n}")
    print(f"  Val: {val_count}")
    for c, n in val_classes: print(f"    - {c}: {n}")
    print(f"  Test: {test_count}")
    for c, n in test_classes: print(f"    - {c}: {n}")

    if train_count == 0:
        raise RuntimeError("No hay imágenes en TRAIN_PATH. Comprueba rutas.")

    print("\nCargando datasets...")
    train_raw = load_split(TRAIN_PATH, shuffle=True)
    val_raw = load_split(VAL_PATH, shuffle=False)
    test_raw = load_split(TEST_PATH, shuffle=False)

    class_names = train_raw.class_names
    print("Clases detectadas:", class_names)
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3)),

        # Transformaciones geométricas
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.35),
        tf.keras.layers.RandomZoom(0.45),
        tf.keras.layers.RandomTranslation(0.25, 0.25),
        tf.keras.layers.RandomCrop(IMG_SIZE[0], IMG_SIZE[1]),

        # Transformaciones fotométricas
        tf.keras.layers.RandomContrast(0.6),
        tf.keras.layers.RandomBrightness(0.4),
        tf.keras.layers.RandomSaturation(0.5),

        # Ruido y perturbaciones 
        tf.keras.layers.GaussianNoise(0.08),
        tf.keras.layers.RandomHue(0.1),

        # Oclusiones o perturbaciones espaciales
        tf.keras.layers.RandomCrop(IMG_SIZE[0], IMG_SIZE[1]),
    ], name="data_augmentation_agresivo")


    train_ds = make_train_ds(train_raw, use_mixup=USE_MIXUP)
    val_ds = make_eval_ds(val_raw)
    test_ds = make_eval_ds(test_raw)

    steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
    validation_steps = math.ceil(val_count / BATCH_SIZE)
    test_steps = math.ceil(test_count / BATCH_SIZE)
    print(f"\nsteps_per_epoch={steps_per_epoch} | validation_steps={validation_steps} | test_steps={test_steps}")

    labels = [int(lab.numpy()) for _, lab in train_raw.unbatch()]
    cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = {int(i): float(w) for i, w in enumerate(cw)}
    print("Pesos de clase:", class_weights)

    model = build_model(class_names)
    callbacks, initial_epoch = setup_callbacks(model)

    # === NUEVO: CSV Logger ===
    csv_logger = EpochMetricsLogger(os.path.join(OUTPUT_DIR, "epoch_metrics.csv"))
    callbacks.append(csv_logger)

    # === Tiempo Fase 1 ===
    start_stage1 = datetime.now()
    history = train_stage(model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps, initial_epoch, callbacks)
    duration_stage1 = datetime.now() - start_stage1
    print(f"🕒 Tiempo Fase 1: {duration_stage1}")

    plot_training_curves(history, "training_curves")

    # === Tiempo Fase 2 (fine-tuning) ===
    start_stage2 = datetime.now()
    history_fine = fine_tune_model(model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps, callbacks)
    duration_stage2 = datetime.now() - start_stage2
    print(f"🕒 Tiempo Fine-Tuning: {duration_stage2}")

    plot_training_curves(history_fine, "training_curves_fine")

    # === Evaluaciones separadas ===
    print("\nEvaluando en conjunto de validación...")
    model.evaluate(val_ds)

    evaluate_and_export(model, test_ds, test_raw, class_names, class_weights, test_steps)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱ Tiempo total de entrenamiento: {duration}")


# ENTRENAMIENTO
def build_model(class_names):
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
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        print("Pesos ImageNet cargados correctamente.")

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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def setup_callbacks(model):
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    checkpoint_best = ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model.keras"),
                                      monitor="val_loss", save_best_only=True, verbose=1)
    checkpoint_full = ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=False, verbose=0)
    return [early_stop, reduce_lr, checkpoint_best, checkpoint_full], 0

def train_stage(model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps, initial_epoch, callbacks):
    print("\nEntrenamiento Fase 1 (cabeza)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history


def fine_tune_model(model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps, callbacks):
    print("\nFine-tuning del modelo base...")
    base_model = model.layers[2]
    base_model.trainable = True

    for layer in base_model.layers[:-100]:
        layer.trainable = False

    optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer_fine, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        initial_epoch=EPOCHS_STAGE1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history_fine

def plot_training_curves(history, filename_prefix="training_curves"):
    """
    Genera gráficos de Accuracy, Loss, F1-score (si existe) y Learning Rate por época.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))

    # === Accuracy ===
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.title('Evolución de Accuracy')
    plt.legend()
    plt.grid(True)

    # === Loss ===
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Evolución de Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}.png"))
    plt.close()

    # === Intentar leer CSV de métricas adicionales ===
    csv_path = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "f1_score" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["epoch"], df["f1_score"], marker='o', color='green')
            plt.xlabel("Época")
            plt.ylabel("F1-score (val)")
            plt.title("Evolución del F1-score por época")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}_f1.png"))
            plt.close()

        if "lr" in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(df["epoch"], df["lr"], marker='o', color='purple')
            plt.xlabel("Época")
            plt.ylabel("Learning Rate")
            plt.title("Evolución del Learning Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_prefix}_lr.png"))
            plt.close()

# EVALUACIÓN Y EXPORTACIÓN
def evaluate_and_export(model, test_ds, test_raw, class_names, class_weights, test_steps):
    print("\nEvaluando modelo final...")
    loss_all = model.evaluate(test_ds, steps=test_steps)
    acc = loss_all[1]
    print(f"Accuracy final (test): {acc:.4f}")

    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_raw:
        x = tf.cast(images, tf.float32)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    acc_test = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    prec_macro = precision_score(y_true, y_pred, average='macro')
    prec_weighted = precision_score(y_true, y_pred, average='weighted')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    rec_weighted = recall_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc_macro = None

    top2_acc = np.mean([label in np.argsort(pred)[-2:] for label, pred in zip(y_true, y_prob)])
    top3_acc = np.mean([label in np.argsort(pred)[-3:] for label, pred in zip(y_true, y_prob)])

    f1_per_class = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, f1_per_class, color='skyblue')
    plt.title("F1-score por clase")
    plt.ylabel("F1-score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"))
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "matriz_confusion.png")
    plt.savefig(cm_path)
    plt.close()

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        auc_score = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC - {cls}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{cls}.png"))
        plt.close()

        plt.figure()
        plt.plot(rec, prec)
        plt.title(f"Precision–Recall - {cls}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"pr_{cls}.png"))
        plt.close()

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    summary = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": "EfficientNetB0 - CNN Frijol (fine-tuned)",
        "accuracy_test": float(acc_test),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "kappa": float(kappa),
        "auc_macro": float(auc_macro) if auc_macro is not None else None,
        "top2_acc": float(top2_acc),
        "top3_acc": float(top3_acc),
        "f1_per_class": {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))},
        "class_report": report_dict,
        "confusion_matrix_path": cm_path,
        "best_model_path": os.path.join(OUTPUT_DIR, "best_model.keras"),
        "checkpoint_full_path": CKPT_PATH,
        "params": {
            "IMG_SIZE": IMG_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS_STAGE1": EPOCHS_STAGE1,
            "EPOCHS_STAGE2": EPOCHS_STAGE2,
            "MIXUP": USE_MIXUP,
            "CLASS_WEIGHTS": class_weights
        }
    }
    summary["training_duration"] = str(duration)

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\nResumen de métricas exportado a metrics_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
