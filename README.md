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

Y la fase de **entrenamiento** del modelo, en un flow aparte que parte de `training_data/`:

```bash
uv run python pipeline/finanom_training_flow.py
```

4. **Entrenamiento** (`model_Adrian/`): modelo HÍBRIDO **consolidado** que une el trabajo de los tres integrantes y produce un reporte de revisión explicable para el auditor nocturno, su evaluación (inyección sintética, estabilidad, overlap) y la model card.

### Consolidación del modelo (lo mejor de los 3)

El pipeline de `model_Adrian/train_model.py` reutiliza:

- **Isolation Forest** con **muestreo estratificado** y **umbral adaptativo** — reutilizado de `model_Rogelio/train.py` (Rogelio).
- **Motor de reglas de negocio** tipadas (8 detectores con catálogos de códigos, scoring y mensajes legibles) — reutilizado de `model_Tony/reglas.py` (Tony).
- **Regla de método de pago** (Visa↔Amex) — de Adrian.
- **Explicabilidad SHAP** (TreeExplainer) sobre las filas que marca el IF — reutilizada de `model_Rogelio` (Rogelio).

La cola de revisión = `rule_score≥60` ∪ IF ∪ método_pago (severidad ALTO/CRÍTICO), rankeada por severidad.

Carpetas de cada integrante (preservadas): `model_Rogelio/` (IF + SHAP + findings), `model_Tony/` (reglas + demo) y `dashboard/` (dashboard Streamlit de Tony; fuera del alcance de esta consolidación, pero funcional con `uv run streamlit run dashboard/app.py`).

Los notebooks y diccionarios documentan las decisiones de cada fase. Los archivos en `output/` son artefactos regenerables de cada etapa. Los datasets finales para entrenar modelos viven en `training_data/`.

## Artefactos principales

- `data_consolidation/output/transacciones_consolidado.parquet`
- `data_cleaning/output/transacciones_limpio.parquet`
- `training_data/transacciones_modelado.parquet`
- `training_data/X_modelo.parquet`
- `data_consolidation/diccionario_base_consolidada.md`
- `data_cleaning/diccionario_limpio.md`
- `data_modeling/diccionario_modelado.md`
- `model_Adrian/output/reporte_revision.parquet` (cola de revisión explicable)
- `model_Adrian/output/reporte_evaluacion_modelo.md` y `model_Adrian/modelo_card.md`

## Estado actual

El pipeline completo genera:

- 1,145,526 transacciones conservadas en todas las fases.
- 63 features finales para modelado.
- Matriz `X_modelo.parquet` completamente numérica, sin nulos y sin features constantes.
- Reportes de calidad e importancias proxy para interpretar qué variables aportan señal.
- Modelo híbrido consolidado: cola de revisión de ~4% (presupuesto del auditor nocturno) con severidad, tipo de inconsistencia, motivo legible (reglas de Tony) y evidencia SHAP (Rogelio) por transacción.

## Modelo final y dashboard (`model_final/`)

`model_final/` es **el modelo que presentamos**: toma el modelo consolidado de `model_Adrian/` y le suma el **mecanismo de aprendizaje** (adaptación de umbral/pesos basada en el feedback del auditor), sin reentrenar el Isolation Forest. El dashboard HTML (diseño de Rogelio) muestra la cola y captura las decisiones del auditor.

**El modelo que aprende (lazo de feedback):**

1. **Genera los datos del dashboard** (cola de revisión actual):
   ```bash
   uv run python model_final/build_dashboard.py
   ```
2. **Levanta el dashboard** y ábrelo en el navegador:
   ```bash
   cd model_final/dashboard && uv run python -m http.server 8080
   #  → http://localhost:8080
   ```
3. **Revisa y marca** cada alerta como *Autorizado / Desestimado / Escalado* (se guardan en el navegador) y pulsa **⬇ Exportar revisiones** → descarga `revisiones.csv`.
4. **El modelo aprende**: ingiere las revisiones, adapta umbral/pesos y regenera la cola:
   ```bash
   uv run python model_final/adapt.py ~/Downloads/revisiones.csv
   ```
   Recarga el dashboard: las alertas de los tipos que el auditor desestima bajan de prioridad y la cola se reduce. Para volver al modelo base, borra `model_final/output/feedback_state.json` y regenera (paso 1).

> Requisito: haber generado antes la base y el modelo consolidado (`uv run python pipeline/finanom_flow.py` y `uv run python model_Adrian/train_model.py`).

Estructura de modelos por integrante: `model_Adrian/` (consolidado), `model_Tony/` (reglas), `model_Rogelio/` (IF + SHAP + dashboard original), `model_final/` (modelo presentable + aprendizaje).

## Instalación..

Este proyecto usa `uv`:

```bash
uv sync
```

Luego ejecutar:

```bash
uv run python pipeline/finanom_flow.py
```
