# Clasificación de Enfermedades del Frijol con Deep Learning (DL_Frejol)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.44%25-brightgreen)
![F1](https://img.shields.io/badge/F1--Macro-0.951-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

Este proyecto implementa un **pipeline completo y reproducible** para la **clasificación automática de enfermedades en hojas de frijol** mediante **transfer learning con EfficientNetB0 (TensorFlow)**. Incluye descarga/preparación de datos, entrenamiento en dos fases, evaluación cuantitativa, generación de métricas y gráficos, explicabilidad (Grad-CAM) y ejecución automatizada mediante un orquestador `main.py`.

---

## 1. Estructura del Proyecto

```text
DL_algorithm_bean/
├── data/                         # Datos (raw, processed, splits)
├── outputs/                      # Resultados generados
│   ├── models/
│   ├── evaluation/
│   └── plots/
├── src/
│   ├── pipeline/                 # Orquestación del flujo completo
│   │   └── main.py
│   ├── training/                 # Preparación de datos y entrenamiento
│   │   ├── dataloader.py
│   │   └── train_model.py
│   ├── evaluation/               # Evaluación del modelo
│   │   └── evaluate_model.py
│   ├── visualization/            # Generación de gráficos
│   │   └── generate_plots.py
│   ├── inference/                # Pruebas con imágenes individuales
│   │   └── image_tester.py
│   └── explainability/           # Grad-CAM y XAI[Fuera del pipeline: ejecutar run_all.py]
│       ├── gradcam.py
│       ├── gradcam_pp.py
│       ├── occlusion.py
│       └── run_all.py
├── notebooks/
│   └── colab_test.ipynb
├── requirements.txt
├── classes.txt
└── README.md
```
## 2. Requisitos
Python ≥ 3.8 (Recomendado: 3.10 || 3.12)
Dependencias principales:
   tensorflow==2.17.0
   numpy==1.26.4
   scikit-learn==1.5.2
   pandas==2.3.3
   matplotlib==3.9.2
   json5==0.9.25
   kagglehub>=0.2.3
   fastapi
   uvicorn
   GPU con CUDA (opcional, recomendada)
## 3. Instalación del Entorno
Opción A — PowerShell (venv)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

Opción B — Conda
   conda create -n dlfrejol python=3.10 -y
   conda activate dlfrejol
   pip install -r requirements.txt
## 4. Ejecución Completa del Pipeline
Para ejecutar todo el flujo (datos → entrenamiento → evaluación → gráficos) con un solo comando:
   python src\pipeline\main.py
Este comando ejecuta secuencialmente:
   Preparación de datos (src/training/dataloader.py)
   Entrenamiento del modelo (src/training/train_model.py)
   Evaluación (src/evaluation/evaluate_model.py)
   Generación de gráficos (src/visualization/generate_plots.py)
## 5. Uso Manual de los Scripts

## 5.1 Preparación de Datos
   python src\training\dataloader.py
Estructura esperada de los datos procesados[Si tiene su data adecualo al siguiente formato]:
```text
   data/processed/
   ├── train/
   │   ├── als/
   │   ├── bean_rust/
   │   ├── healthy/
   │   └── unknown/
   ├── val/
   │   ├── als/
   │   ├── bean_rust/
   │   ├── healthy/
   │   └── unknown/
   └── test/
      ├── als/
      ├── bean_rust/
      ├── healthy/
      └── unknown/
```
## 5.3 Evaluación del Modelo
   python src\evaluation\evaluate_model.py
Este script carga outputs/models/best_model.keras, evalúa el desempeño sobre el conjunto de prueba y guarda artefactos de evaluación.
Artefactos generados:
   outputs/evaluation/
```text
   ├── metrics_summary.json    # Resumen global de métricas
   ├── y_true.npy              # Etiquetas verdaderas
   ├── y_pred.npy              # Predicciones discretas
   ├── y_prob.npy              # Probabilidades por clase
   ├── images.npy              # Referencias a las imágenes de test
   ├── embeddings.npy          # Embeddings del penúltimo layer
   └── confusion_matrix.npy    # Matriz de confusión
```
## 5.4 Generación de Gráficos
   python src\visualization\generate_plots.py
Salida:
   outputs/plots/
```text
   ├── training_curves.png / .pdf / .svg
   ├── confusion_matrix.png / .pdf / .svg
   └── (otros gráficos según configuración)
```
## 6. Resultados del Modelo[Del modelo reportado]
## 6.1 Métricas Globales
| Métrica         | Valor      |
| --------------- | ---------- |
| Accuracy        | **0.9444** |
| F1 Macro        | **0.9510** |
| F1 Weighted     | **0.9440** |
| Precision Macro | **0.9512** |
| Recall Macro    | **0.9519** |
| Cohen’s Kappa   | **0.9247** |
| AUC Macro       | **0.9941** |
| Nº de Imágenes  | **1296**   |
## 6.2 Métricas por Clase
| Clase            | Precisión  | Recall     | F1-Score   | Soporte  |
| ---------------- | ---------- | ---------- | ---------- | -------- |
| ALS              | 0.9143     | 0.9362     | 0.9251     | 376      |
| Bean Rust        | 0.9539     | 0.8803     | 0.9156     | 376      |
| Healthy          | 0.9366     | 0.9913     | 0.9632     | 343      |
| Unknown          | 1.0000     | 1.0000     | 1.0000     | 201      |
| **Macro Avg**    | **0.9512** | **0.9519** | **0.9510** | **1296** |
| **Weighted Avg** | **0.9450** | **0.9444** | **0.9440** | **1296** |


## 7.Dataset y Créditos
M. S. Jahid, “Bean Crop Disease Diagnosis and Spatial Analysis,” Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/msjahid/bean-crop-disease-diagnosis-and-spatial-analysis
