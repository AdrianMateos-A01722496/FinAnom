# FINANOM

FINANOM detecta anomalías e inconsistencias financieras en transacciones hoteleras y las muestra en un dashboard para revisión operativa.

## Arquitectura del modelo final

Todo lo necesario para correr la demo vive en `model_final/`.

`model_final` combina cuatro piezas:

- **Datos y artefactos**: parquets limpios/modelados, modelo entrenado, reporte de revisión y archivos del dashboard.
- **Modelo no supervisado**: Isolation Forest sobre features `feat_*` para detectar patrones atípicos globales.
- **Reglas de negocio**: validaciones explicables para duplicados, montos atípicos, cancelaciones, cargos fuera de estancia, signos contables, método de pago y otros casos operativos.
- **Aprendizaje por feedback**: cuando el auditor desestima o confirma alertas, el sistema ajusta umbral y pesos de reglas para reducir falsos positivos en la siguiente cola.

La salida principal es una cola priorizada de revisión con severidad, motivo legible y evidencia para el auditor.

## Dashboard con aprendizaje

```bash
uv run python model_final/app.py
```

Abrir: <http://127.0.0.1:5000>

Flujo:

1. Revisar alertas.
2. Marcar `Autorizado`, `Desestimado` o `Escalado`.
3. Presionar `Aplicar correcciones`.

El backend aplica el feedback automáticamente: guarda revisiones, actualiza el estado aprendido y regenera los datos del dashboard sin CSV ni terminal.

## Comandos utiles

```bash
uv run python model_final/build_dashboard.py
uv run python model_final/train_model.py
```

## Duplicados

La regla `DUPLICADO` ya no marca como anomalía cualquier repetición en el mismo día. Ahora solo suma score cuando hay duplicado de alta confianza: mismo folio, subfolio, cuarto, código, naturaleza contable, monto y minuto. Las repeticiones por día quedan como contexto porque pueden ser cargos legítimos.

## Instalacion

```bash
uv sync
```
