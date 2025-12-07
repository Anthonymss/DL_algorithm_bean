import os
import json
import csv
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# CONFIGURACIÓN GLOBAL
np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 60

BASE_DIR = "data/processed"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

OUTPUT_DIR = "outputs/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
CLASSES_PATH = os.path.join(OUTPUT_DIR, "classes.txt")
TRAINING_TIMES = os.path.join(OUTPUT_DIR, "training_times.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")
CHECKPOINT_FULL = os.path.join(OUTPUT_DIR, "checkpoint_full.keras")

class EpochMetricsLogger(Callback):
    def __init__(self, filepath, val_data):
        super().__init__()
        self.filepath = filepath
        self.val_data = val_data

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "acc", "val_acc",
                "loss", "val_loss",
                "val_f1"
            ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        y_true, y_pred = [], []
        for x, y in self.val_data:
            preds = self.model.predict(x, verbose=0)
            y_true.extend(y.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
        val_f1 = f1_score(y_true, y_pred, average="macro")

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, lr, acc, val_acc, loss, val_loss, val_f1])

# PREPROCESAMIENTO
def ensure_rgb_tf(image, label):
    """Asegura RGB + resize + float32."""
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def load_split(path, shuffle=False):
    """Carga dataset en batches de tamaño correcto (32)."""
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=shuffle
    )


def preprocess_batch(x, y):
    x = preprocess_input(x)
    return x, y

# DATA AUGMENTATION
def create_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.35),
        tf.keras.layers.RandomContrast(0.40),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.GaussianNoise(0.03),
    ])

# MODELO
def build_model(class_names, data_aug):
    base_model = EfficientNetB0(include_top=False, weights="imagenet",
                                input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_aug(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ENTRENAMIENTO
def train():
    global_start = datetime.now()

    print("Cargando datos...")
    train_raw = load_split(TRAIN_DIR, shuffle=True)
    val_raw   = load_split(VAL_DIR, shuffle=False)

    class_names = train_raw.class_names
    print("Clases detectadas:", class_names)

    with open(CLASSES_PATH, "w") as f:
        f.write("\n".join(class_names))

    train_ds = train_raw.map(ensure_rgb_tf).map(preprocess_batch).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_raw.map(ensure_rgb_tf).map(preprocess_batch).prefetch(tf.data.AUTOTUNE)

    train_steps = math.ceil(len(train_raw) / 1) 
    val_steps   = math.ceil(len(val_raw) / 1)

    print("train_steps =", train_steps)
    print("val_steps   =", val_steps)

    labels = [int(y.numpy()) for _, y in train_raw.unbatch()]
    cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    data_aug = create_augmentation_layer()
    model = build_model(class_names, data_aug)

    logger = EpochMetricsLogger(CSV_LOG_PATH, val_ds)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, min_lr=1e-6),
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ModelCheckpoint(CHECKPOINT_FULL, monitor="val_loss", save_weights_only=False, verbose=0),
        logger
    ]

    # FASE 1 — Cabeza congelada
    print("\n=== Entrenando FASE 1 ===")
    start1 = datetime.now()

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=callbacks
    )

    t1 = datetime.now() - start1

    # FASE 2 — Fine-tuning
    print("\n=== Entrenando FASE 2 (Fine-Tuning) ===")

    base = model.layers[2]
    base.trainable = True

    for layer in base.layers[:-100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start2 = datetime.now()
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        initial_epoch=EPOCHS_STAGE1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=callbacks
    )
    t2 = datetime.now() - start2
    times = {
        "fase_1_tiempo": str(t1),
        "fase_2_tiempo": str(t2),
        "total": str(datetime.now() - global_start)
    }

    with open(TRAINING_TIMES, "w") as f:
        json.dump(times, f, indent=4)

    print("\nEntrenamiento finalizado.")
    print(json.dumps(times, indent=4))

    return history1, history2

if __name__ == "__main__":
    train()