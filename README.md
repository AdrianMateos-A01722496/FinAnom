# FINANOM

FINANOM es una demo de deteccion no supervisada de anomalías e inconsistencias financieras sobre datos de un PMS hotelero. El proyecto usa transacciones reales de un hotel en México, principalmente en MXN, y prepara un dataset listo para modelado con Isolation Forest.

## Problema de negocio

En hoteles, cargos duplicados, errores de posteo manual, montos fuera de contexto o cancelaciones mal registradas suelen detectarse hasta la auditoría nocturna. Esto retrasa el cierre contable y obliga al auditor a corregir incidencias cuando el día operativo ya terminó.

## User story 

> Como auditor, quiero que el sistema me alerte sobre cargos atípicos durante el día para no esperar a la auditoría nocturna para corregirlos.

## Objetivo

Construir un pipeline reproducible que transforme datos crudos del PMS en una matriz de features limpia, codificada y lista para detección no supervisada. La meta no es crear un modelo perfecto, sino una demo completa, explicable y robusta para identificar transacciones que ameritan corrección inmediata o revisión por auditoría.

## Datos

- Fuente cruda: `tablas_parquet/` con 24 tablas originales.
- Diccionarios fuente: `diccionario_datos/`.
- Tabla núcleo: `hottra`, con 1,145,526 transacciones.
- Contexto de reservación: `hothsp`, con 171,669 reservaciones.
- Grano analítico: una fila = una transacción = una posible predicción.

## Pipeline

El flujo principal está orquestado con Prefect:

```bash
uv run python pipeline/finanom_flow.py
```

Fases:

1. **Consolidación** (`data_consolidation/`): une transacciones (`hottra`) con contexto de reservación (`hothsp`) y genera una base analítica única.
2. **Limpieza** (`data_cleaning/`): elimina columnas redundantes, constantes, colineales o de bajo valor sin borrar transacciones.
3. **Modelado de datos** (`data_modeling/`): crea features, aplica encoding, escalado robusto, imputación, diagnósticos de calidad e Isolation Forest proxy.

Los notebooks y diccionarios documentan las decisiones de cada fase. Los archivos en `output/` son artefactos regenerables de cada etapa. Los datasets finales para entrenar modelos viven en `training_data/`.

## Artefactos principales

- `data_consolidation/output/transacciones_consolidado.parquet`
- `data_cleaning/output/transacciones_limpio.parquet`
- `training_data/transacciones_modelado.parquet`
- `training_data/X_modelo.parquet`
- `data_consolidation/diccionario_base_consolidada.md`
- `data_cleaning/diccionario_limpio.md`
- `data_modeling/diccionario_modelado.md`

## Estado actual

El pipeline completo genera:

- 1,145,526 transacciones conservadas en todas las fases.
- 63 features finales para modelado.
- Matriz `X_modelo.parquet` completamente numérica, sin nulos y sin features constantes.
- Reportes de calidad e importancias proxy para interpretar qué variables aportan señal.

## Instalación..

Este proyecto usa `uv`:

```bash
uv sync
```

Luego ejecutar:

```bash
uv run python pipeline/finanom_flow.py
```
