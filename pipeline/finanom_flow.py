"""Flow Prefect para ejecutar FINANOM de crudos a dataset modelado.

Uso:
    uv run python pipeline/finanom_flow.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_cleaning.clean import DICT_FILE as CLEAN_DICTIONARY_FILE
from data_cleaning.clean import OUTPUT_FILE as CLEAN_FILE
from data_cleaning.clean import run_cleaning
from data_consolidation.consolidate import OUTPUT_FILE as CONSOLIDATED_FILE
from data_consolidation.consolidate import PARQUET_DIR as RAW_DIR
from data_consolidation.consolidate import run_consolidation
from data_modeling.data_modeling import DICT_FILE as MODELING_DICTIONARY_FILE
from data_modeling.data_modeling import OUTPUT_DIR as MODELING_OUTPUT_DIR
from data_modeling.data_modeling import TRAINING_DATA_DIR
from data_modeling.data_modeling import run_modeling


@task(name="consolidar-datos")
def consolidar_datos(raw_dir: Path | str, output_file: Path | str) -> Path:
    """Genera la base consolidada desde las tablas parquet crudas."""
    logger = get_run_logger()
    path = run_consolidation(raw_dir=raw_dir, output_file=output_file)
    logger.info("Base consolidada generada: %s", path)
    return path


@task(name="limpiar-datos")
def limpiar_datos(
    input_file: Path | str,
    output_file: Path | str,
    dictionary_file: Path | str,
) -> Path:
    """Genera la base limpia y su diccionario."""
    logger = get_run_logger()
    path = run_cleaning(
        input_file=input_file,
        output_file=output_file,
        dictionary_file=dictionary_file,
    )
    logger.info("Base limpia generada: %s", path)
    return path


@task(name="modelar-datos")
def modelar_datos(
    input_file: Path | str,
    output_dir: Path | str,
    training_data_dir: Path | str,
    dictionary_file: Path | str,
) -> dict[str, Path]:
    """Genera el dataset modelado, matriz X y diagnosticos de calidad."""
    logger = get_run_logger()
    artifacts = run_modeling(
        input_file=input_file,
        output_dir=output_dir,
        training_data_dir=training_data_dir,
        dictionary_file=dictionary_file,
    )
    logger.info("Dataset modelado generado: %s", artifacts["dataset"])
    return artifacts


@flow(name="finanom-data-pipeline")
def finanom_data_pipeline(
    raw_dir: Path | str = RAW_DIR,
    consolidated_file: Path | str = CONSOLIDATED_FILE,
    clean_file: Path | str = CLEAN_FILE,
    modeling_output_dir: Path | str = MODELING_OUTPUT_DIR,
    training_data_dir: Path | str = TRAINING_DATA_DIR,
    clean_dictionary_file: Path | str = CLEAN_DICTIONARY_FILE,
    modeling_dictionary_file: Path | str = MODELING_DICTIONARY_FILE,
) -> dict[str, Path]:
    """Ejecuta el pipeline completo con artefactos en disco entre fases."""
    consolidated_path = consolidar_datos(raw_dir, consolidated_file)
    clean_path = limpiar_datos(consolidated_path, clean_file, clean_dictionary_file)
    modeling_artifacts = modelar_datos(clean_path, modeling_output_dir, training_data_dir, modeling_dictionary_file)

    return {
        "consolidated": Path(consolidated_path),
        "clean": Path(clean_path),
        **{name: Path(path) for name, path in modeling_artifacts.items()},
    }


def main() -> None:
    artifacts = finanom_data_pipeline()
    print("\nArtefactos generados:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
