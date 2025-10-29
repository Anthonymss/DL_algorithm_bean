# Clasificación de Enfermedades del Frijol con DL

Este proyecto implementa un modelo de aprendizaje profundo para clasificar enfermedades del frijol utilizando transfer learning con EfficientNetB0. El modelo está entrenado para identificar diferentes tipos de enfermedades del frijol a partir de imágenes.

## Estructura del Proyecto

```
DL_Frejol/
├── data/                  # Directorio para los datos de entrenamiento y validación
├── outputs/               # Directorio para guardar modelos y registros de entrenamiento
├── src/                   # Código fuente
│   ├── dataloader.py     # Utilidades para cargar y preprocesar datos
│   ├── train_model.py    # Script para entrenar el modelo
│   ├── evaluate_model.py # Script para evaluar el modelo
│   └── efficientnetb0_notop.h5  # Pesos pre-entrenados de EfficientNetB0
├── requirements.txt      # Dependencias de Python
└── classes.txt           # Lista de clases o categorías
```

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd DL_Frejol
   ```

2. Crea un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. Instala los paquetes requeridos:
   ```bash
   pip install -r requirements.txt
   ```

## Preparación de Datos

1. Ejecuta el script `dataloader.py` para descargar y organizar automáticamente el conjunto de datos:
   ```bash
   python src/dataloader.py
   ```
   
   Este script se encargará de:
   - Descargar el conjunto de datos de enfermedades del frijol
   - Organizar las imágenes en los directorios correspondientes
   - Crear las divisiones de entrenamiento y validación
   - Verificar la integridad de los datos

2. Si prefieres usar tus propias imágenes, colócalas en el directorio `data/` organizadas en subdirectorios por clase, asegurándote de que los nombres de las clases coincidan con los listados en `classes.txt`.

## Entrenamiento del Modelo

Para entrenar el modelo, ejecuta:

```bash
python src/train_model.py
```

Los parámetros de entrenamiento se pueden modificar en el archivo `train_model.py`, incluyendo:
- Tamaño del lote (batch size)
- Número de épocas
- Tasa de aprendizaje
- Dimensiones de las imágenes
- Configuración de aumento de datos

## Evaluación del Modelo

Para evaluar el modelo entrenado:

```bash
python src/evaluate_model.py
```

## Resultados

Las métricas y visualizaciones del rendimiento del modelo se guardarán en el directorio `outputs/`, incluyendo:
- Gráficas de precisión y pérdida (entrenamiento/validación)
- Matriz de confusión
- Reporte de clasificación

## Dependencias

- TensorFlow 2.17.0
- NumPy 1.26.4
- scikit-learn 1.5.2
- Matplotlib 3.9.2
- Seaborn 0.13.2
- Pillow 10.4.0
