import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(step_name, script_path):
    print(f"\n▶ {step_name}")
    print(f"→ Ejecutando: {script_path}")
    subprocess.run([PYTHON, str(script_path)], check=True)


def main():

    run(
        "Preparación de datos",
        ROOT / "src/training/dataloader.py"
    )

    run(
        "Entrenamiento del modelo",
        ROOT / "src/training/train_model.py"
    )

    run(
        "Evaluación del modelo",
        ROOT / "src/evaluation/evaluate_model.py"
    )

    run(
        "Generación de gráficos",
        ROOT / "src/visualization/generate_plots.py"
    )

    print("\n✅ PIPELINE COMPLETADO CON ÉXITO")


if __name__ == "__main__":
    main()
