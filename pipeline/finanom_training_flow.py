"""Flow Prefect para entrenar el modelo FINANOM sobre los datos de `training_data/`.

Depende de los artefactos que produce `pipeline/finanom_flow.py` (la fase de datos).
Ejecuta el modelo HIBRIDO (Isolation Forest + reglas de negocio) y genera el reporte
de revision, la evaluacion de calidad y la model card.

Cadena completa desde cero:
    uv run python pipeline/finanom_flow.py            # crudos -> training_data/
    uv run python pipeline/finanom_training_flow.py   # training_data/ -> modelo + reporte

Uso:
    uv run python pipeline/finanom_training_flow.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_Adrian.train_model import (
    CLEAN_FILE,
    MODEL_CARD_FILE,
    MODELED_FILE,
    OUTPUT_DIR,
    X_FILE,
    run_training,
)


@task(name="entrenar-modelo")
def entrenar_modelo(
    modeled_file: Path | str,
    x_file: Path | str,
    clean_file: Path | str,
    output_dir: Path | str,
    model_card_file: Path | str,
) -> dict[str, Path]:
    """Entrena el modelo hibrido y genera reporte, evaluacion y model card."""
    logger = get_run_logger()
    artifacts = run_training(
        modeled_file=modeled_file,
        x_file=x_file,
        clean_file=clean_file,
        output_dir=output_dir,
        model_card_file=model_card_file,
    )
    logger.info("Reporte de revision generado: %s", artifacts["report"])
    return artifacts


@flow(name="finanom-training-pipeline")
def finanom_training_pipeline(
    modeled_file: Path | str = MODELED_FILE,
    x_file: Path | str = X_FILE,
    clean_file: Path | str = CLEAN_FILE,
    output_dir: Path | str = OUTPUT_DIR,
    model_card_file: Path | str = MODEL_CARD_FILE,
) -> dict[str, Path]:
    """Entrena el modelo a partir de los datasets de `training_data/`."""
    return entrenar_modelo(modeled_file, x_file, clean_file, output_dir, model_card_file)


def main() -> None:
    artifacts = finanom_training_pipeline()
    print("\nArtefactos del modelo:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
