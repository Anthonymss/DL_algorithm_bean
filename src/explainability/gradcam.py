import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2
import os
from datetime import datetime

IMG_SIZE = (224, 224)

def generate_gradcam(image: Image.Image, model, class_idx, save_dir=None, save=False):
    img_resized = image.resize(IMG_SIZE)
    img_arr = np.array(img_resized).astype(np.float32)
    img_input = preprocess_input(img_arr)
    img_input = np.expand_dims(img_input, axis=0)

    effnet = model.get_layer("efficientnetb0")

    cam_model = tf.keras.models.Model(
        inputs=effnet.input,
        outputs=[
            effnet.get_layer("top_conv").output,
            effnet.output
        ]
    )

    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, preds = cam_model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1).numpy()

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-7)
    cam = cv2.resize(cam, IMG_SIZE)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_arr = np.array(image.resize(IMG_SIZE)).astype(np.uint8)
    overlay = cv2.addWeighted(orig_arr, 0.6, heatmap, 0.4, 0)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "gradcam.png")
        Image.fromarray(overlay).save(filepath)

    return overlay
