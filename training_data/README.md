# Training data

Esta carpeta contiene los datasets finales que se usan para entrenar modelos.

Se generan ejecutando:

```bash
uv run python pipeline/finanom_flow.py
```

Artefactos esperados:

- `transacciones_modelado.parquet`: dataset final con columnas `trace_*` para auditoria y `feat_*` para modelado.
- `X_modelo.parquet`: matriz numerica pura, lista para entrenar Isolation Forest.

Los archivos `.parquet` no se versionan en git porque son artefactos regenerables.
