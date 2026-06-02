# FINANOM

FINANOM detecta anomalías e inconsistencias financieras en transacciones hoteleras y las muestra en un dashboard para revisión operativa.

## Requisitos

- Python 3.12+
- `uv`

Si no tienes `uv`, instala siguiendo: <https://docs.astral.sh/uv/getting-started/installation/>

Instala dependencias:

```bash
uv sync
```

## Forma rápida: correr la demo

El repo ya incluye los artefactos necesarios dentro de `model_final/`, así que para probar el modelo final no necesitas reconstruir los datos.

```bash
uv run python model_final/app.py
```

Abre:

```text
http://127.0.0.1:5000
```

Flujo en el dashboard:

1. Revisa alertas.
2. Marca `Autorizado`, `Desestimado` o `Escalado`.
3. Presiona `Aplicar correcciones`.

El backend guarda el feedback, actualiza el estado aprendido y regenera la cola automáticamente.

## Arquitectura de `model_final`

`model_final/` es la carpeta autónoma del modelo final:

- `app.py`: backend Flask y API del dashboard.
- `dashboard/`: interfaz para revisar alertas.
- `model.py`: aplica el estado aprendido y genera la cola de revisión.
- `adapt.py` y `feedback.py`: convierten revisiones del auditor en ajustes de umbral y pesos.
- `reglas.py`: reglas de negocio explicables.
- `train_model.py`: reentrenamiento del modelo final.
- `data/`, `training_data/` y `output/`: datos y artefactos necesarios para correr la demo.

El modelo combina:

- Isolation Forest sobre features `feat_*`.
- Reglas de negocio para inconsistencias operativas.
- Feedback del auditor para bajar falsos positivos.

## Reconstruir desde datos crudos

Usa esto si quieres regenerar todo desde `tablas_parquet/`.

1. Genera datos consolidados, limpios y modelados:

```bash
uv run python pipeline/finanom_flow.py
```

Esto produce:

- `data_consolidation/output/transacciones_consolidado.parquet`
- `data_cleaning/output/transacciones_limpio.parquet`
- `training_data/transacciones_modelado.parquet`
- `training_data/X_modelo.parquet`

2. Copia los artefactos al modelo final autónomo:

```bash
mkdir -p model_final/data model_final/training_data
cp data_cleaning/output/transacciones_limpio.parquet model_final/data/
cp training_data/transacciones_modelado.parquet model_final/training_data/
cp training_data/X_modelo.parquet model_final/training_data/
```

3. Reentrena el modelo final:

```bash
uv run python model_final/train_model.py
```

4. Regenera datos del dashboard:

```bash
uv run python model_final/build_dashboard.py
```

5. Levanta la app:

```bash
uv run python model_final/app.py
```

## Comandos útiles

```bash
uv run python model_final/build_dashboard.py
uv run python model_final/train_model.py
uv run ruff check model_final pyproject.toml
```

## Duplicados

La regla `DUPLICADO` no marca como anomalía cualquier repetición en el mismo día. Solo suma score cuando hay duplicado de alta confianza: mismo folio, subfolio, cuarto, código, naturaleza contable, monto y minuto.

Las repeticiones por día quedan como contexto porque pueden ser cargos legítimos.
