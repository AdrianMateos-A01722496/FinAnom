# FinAnom Dashboard

Dashboard local para visualizar alertas financieras hoteleras detectadas por el módulo de reglas de negocio.

---

## Inicio rápido (3 pasos)

**Paso 1 — Genera los datos del modelo** (si no existen aún):
```bash
uv run python Tony_anomaly_detection/run_demo.py
```

**Paso 2 — Genera la muestra para el dashboard:**
```bash
uv run python dashboard/create_sample.py
```

**Paso 3 — Corre el dashboard:**
```bash
uv run streamlit run dashboard/app.py
```

Se abrirá en `http://localhost:8501`.

---

## Estructura de archivos

```
dashboard/
├── app.py                    # Aplicación principal Streamlit
├── data_loader.py            # Carga, filtrado y cálculo de KPIs
├── state_manager.py          # Etiquetas manuales (feedback_manual.csv)
├── create_sample.py          # Script para generar muestras balanceadas
├── README.md                 # Este archivo
├── feedback_manual.csv       # Etiquetas guardadas (se crea al primera revisión)
└── data/
    ├── sample_alertas.csv        # Muestra de alertas (creada por create_sample.py)
    └── sample_transacciones.csv  # Muestra completa (alertas + señales)
```

---

## Fuentes de datos

| Modo (sidebar) | Archivo cargado | Tamaño |
|---|---|---|
| **Muestra de prueba** (default) | `dashboard/data/sample_alertas.csv` | ≤ 1,000 filas |
| **Archivos completos** | `anomaly_detection/output_alertas_operativas.csv` | ilimitado |

---

## Funcionalidades actuales

### KPIs (fila superior)
- Total transacciones · Alertas operativas · % con alerta
- Críticas · Altas · Medias
- Duplicados · Pagos sospechosos · Fuera de estancia · Montos atípicos · Cancelaciones

### Filtros
- **Fecha**: Hoy / Esta semana / Este mes / Este año / Todas / Rango personalizado
- **Nivel de riesgo**: CRITICO / ALTO / MEDIO / BAJO (multiselect)
- **Cluster de anomalía**: multiselect con todos los clusters presentes
- **Orden**: mayor score / más recientes / más antiguas

### Tabla de alertas
- Colores por nivel: 🔴 CRITICO · 🟠 ALTO · 🟡 MEDIO · ⬜ BAJO
- Muestra etiqueta manual si ya fue revisada
- Mensajes truncados en la tabla (texto completo en el detalle)

### Detalle y etiquetado
- Tarjeta visual con todos los campos de la alerta
- Expandibles: reglas activadas · señales de contexto
- Formulario de etiquetado con 3 opciones + comentario libre
- Historial de etiquetas con botón de descarga CSV

### Tabla de todas las transacciones
- Incluye tanto alertas como señales de contexto (si están disponibles)
- Misma codificación de color por nivel de riesgo

---

## Etiquetado manual

Las etiquetas se guardan en `dashboard/feedback_manual.csv`:

| Campo | Descripción |
|---|---|
| `id_transaccion` | ID de la transacción revisada |
| `etiqueta_manual` | `Es anomalía` / `No es anomalía` / `Pendiente de revisar` |
| `comentario` | Texto libre del revisor |
| `timestamp_revision` | Fecha/hora ISO-8601 de la revisión |

Este archivo se usará en el futuro como dataset de entrenamiento supervisado.

---

## Para conectar a datos en tiempo real

1. **Reemplazar `data_loader.py`** con una conexión directa a la BD TSA (SQLAlchemy / ODBC).
2. **Agregar botón "Actualizar"** que ejecute `detectar_anomalias(df)` sobre las transacciones del día.
3. **Automatizar con Prefect** (`pipeline/finanom_flow.py`) para detección nightly pre-auditoría.
4. **Migrar `feedback_manual.csv`** a una tabla SQL para revisiones multiusuario.
5. **Agregar autenticación** con `streamlit-authenticator` o despliegue en Streamlit Cloud.
