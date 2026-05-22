# Pipeline FINANOM con Prefect

Este directorio orquesta las tres fases reproducibles del proyecto:

1. Consolidacion: `tablas_parquet/` -> `data_consolidation/output/transacciones_consolidado.parquet`
2. Limpieza: consolidado -> `data_cleaning/output/transacciones_limpio.parquet`
3. Modelado: limpio -> `data_modeling/output/transacciones_modelado.parquet` y `X_modelo.parquet`

Los notebooks y diccionarios siguen siendo documentacion explicativa. La fuente ejecutable del pipeline son los scripts Python.

## Ejecucion local

```bash
uv run python pipeline/finanom_flow.py
```

El flow usa rutas por defecto equivalentes a las fases existentes y sobrescribe los artefactos generados.

## UI opcional de Prefect

En una terminal:

```bash
uv run prefect server start
```

En otra terminal:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python pipeline/finanom_flow.py
```

La UI queda disponible en `http://127.0.0.1:4200`.

## Artefactos

- `data_consolidation/output/transacciones_consolidado.parquet`
- `data_cleaning/output/transacciones_limpio.parquet`
- `data_cleaning/diccionario_limpio.md`
- `data_modeling/output/transacciones_modelado.parquet`
- `data_modeling/output/X_modelo.parquet`
- `data_modeling/diccionario_modelado.md`
- `data_modeling/output/reporte_calidad_modelado.md`
- `data_modeling/output/proxy_feature_importance.csv`
- `data_modeling/output/proxy_anomaly_sample.csv`
- `data_modeling/output/proxy_feature_importance_top20.png`
- `data_modeling/output/proxy_anomaly_score_distribution.png`
