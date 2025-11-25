import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)

def _preprocess(image: Image.Image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    return arr

def occlusion_sensitivity(image: Image.Image, model, class_idx,
                          patch=32, stride=16, save_dir=None, save=False):

    img_arr = _preprocess(image)
    img_arr = np.expand_dims(img_arr, axis=0)

    H, W = IMG_SIZE
    heatmap = np.zeros((H, W))

    base_prob = model.predict(img_arr)[0][class_idx]

    for y in range(0, H - patch, stride):
        for x in range(0, W - patch, stride):

            occluded = img_arr.copy()
            occluded[0, y:y+patch, x:x+patch, :] = 0

            p = model.predict(occluded)[0][class_idx]
            drop = base_prob - p

            heatmap[y:y+patch, x:x+patch] = drop

    heatmap_norm = heatmap - heatmap.min()
    heatmap_norm /= (heatmap_norm.max() + 1e-7)
    heatmap_norm = (heatmap_norm * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig = np.array(image.resize(IMG_SIZE)).astype(np.uint8)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "occlusion.png")
        Image.fromarray(overlay).save(filepath)

    return overlay
