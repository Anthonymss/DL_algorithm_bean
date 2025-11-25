# Clasificación de Enfermedades del Frijol con DL

Este proyecto implementa un modelo de aprendizaje profundo para clasificar enfermedades del frijol utilizando transfer learning con EfficientNetB0. El modelo está entrenado para identificar diferentes tipos de enfermedades del frijol a partir de imágenes.

## Estructura del Proyecto

```
DL_Frejol/
├── data/                  # Directorio para los datos de entrenamiento y validación
├── outputs/               # Directorio para guardar modelos y registros de entrenamiento
├── src/                   # Código fuente
│   ├── model/            # Scripts de entrenamiento y evaluación
│   │   ├── dataloader.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── generate_plots.py
│   └── explainability/   # Scripts de explicabilidad (GradCAM, etc.)
├── requirements.txt      # Dependencias de Python
└── classes.txt           # Lista de clases o categorías
```

## Requisitos Previos

- Python 3.8 o superior
# Clasificación de Enfermedades del Frijol mediante Aprendizaje Profundo

Resumen: Este repositorio contiene la implementación de un flujo completo para clasificación de imágenes de hojas de frijol mediante transferencia de aprendizaje con EfficientNetB0. Incluye scripts para descarga y organización del dataset, entrenamiento (con fase de preentrenamiento y fine‑tuning), evaluación y generación de figuras y métricas reproducibles.

**Repositorio**: `DL_Frejol`

**2. Requisitos**

- Python 3.8+ (probado en Python 3.10 / 3.12).
- Dependencias listadas en `requirements.txt`. Versión fija principal:
   - `tensorflow==2.17.0`, `numpy==1.26.4`, `scikit-learn==1.5.2`, `pandas==2.3.3`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `json5==0.9.25`, `kagglehub>=0.2.3`.
- Recomendado: GPU con CUDA compatible para acelerar el entrenamiento (no obligatorio).

Instalación rápida (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. Preparación de datos**

- Descargar y organizar automáticamente:

```powershell
python src\model\dataloader.py
```
Si usa datos propios, coloque imágenes en `data/processed` con la misma estructura por clase.

**4. Entrenamiento**

- Ejecutar entrenamiento completo:

```powershell
python src\model\train_model.py
```

- Comportamiento clave:
   - Resolución de entrada: `224x224` (constante `IMG_SIZE`).
   - Batch size por defecto: `32` (constante `BATCH_SIZE`).
   - Entrenamiento en dos fases: fase 1 (cabeza entrenable, base congelada) y fase 2 (fine‑tuning de capas superiores).
   - Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` y logger personalizado que escribe `outputs/models/epoch_metrics.csv`.
   - Artefactos generados en `outputs/models/`: `best_model.keras`, `checkpoint_full.keras`, `classes.txt`, `epoch_metrics.csv`, `training_times.json`.

Parámetros comunes (modificar en `train_model.py` si se requiere): `BATCH_SIZE`, `IMG_SIZE`, `EPOCHS_STAGE1`, `EPOCHS_STAGE2`.

**5. Evaluación y generación de figuras**

- Evaluación sobre `data/processed/test`:

```powershell
python src\model\evaluate_model.py
```

- Generar figuras y resúmenes a partir de los ficheros de evaluación y del log de épocas:

```powershell
python src\model\generate_plots.py
```

- Artefactos de evaluación: `outputs/evaluation/metrics_summary.json`, `y_true.npy`, `y_pred.npy`, `y_prob.npy`, `embeddings.npy`, `confusion_matrix.npy`.

**6. Resultados reproducibles**

- Guardado de métricas por época: `outputs/models/epoch_metrics.csv`.
- Modelo guardado: `outputs/models/best_model.keras` (mejor según `val_loss`).
- Plots exportados en `outputs/plots/{png,svg,pdf}`.

Para reproducibilidad exacta: fije semillas y registre el entorno (ej. `pip freeze > requirements_freeze.txt`) y el archivo `outputs/models/training_times.json` que ya registra tiempos.

**7. Buenas prácticas y recomendaciones**

- Validar integridad y balance del dataset antes de entrenar.
- Comprobar `classes.txt` y que las etiquetas en `data/processed` coincidan con su orden.
- Para experimentos replicables, guarde una copia de `outputs/models/checkpoint_full.keras` y del CSV de épocas.

**8. Agradecimientos y licencia**
Este proyecto se basa en transfer learning con EfficientNetB0 (TensorFlow). Si corresponde, indique la fuente original del dataset (por ejemplo: `msjahid/bean-crop-disease-diagnosis-and-spatial-analysis` en Kaggle) al citar resultados.
--------------------------------------------------------------------------------

Autor(es), "Clasificación de enfermedades del frijol mediante aprendizaje profundo (DL_Frejol)", repositorio GitHub, Año. [En línea]. Disponible: <url-del-repositorio>
