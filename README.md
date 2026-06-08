# FINANOM

FINANOM detecta anomalías e inconsistencias financieras en transacciones hoteleras **en el momento en que ocurren** y las presenta en un dashboard para revisión operativa. El objetivo es que el auditor no tenga que esperar a la auditoría nocturna para corregir errores como cargos duplicados, montos atípicos, modificaciones no autorizadas o métodos de pago mal aplicados.

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

En el primer arranque se crea automáticamente una base de datos local SQLite (`model_final/output/finanom_demo.sqlite`) y se llena con el último año de transacciones ya puntuadas a partir del reporte batch. No se requiere ningún paso manual.

## Qué hace el dashboard

El dashboard tiene tres vistas:

1. **Dashboard**: alertas priorizadas del día, con su severidad, tipo de inconsistencia y motivo.
2. **Transacciones históricas**: todas las transacciones (anómalas y normales) con filtros por anomalía, severidad, fecha, folio, cuarto, concepto, categoría y rango de monto, además de paginación y ordenamiento.
3. **Estadísticas**: tendencias y composición de las anomalías.

Acciones:

- **Simular transacción (`normal` / `anómala`)**: inserta una transacción nueva que se **puntúa en vivo** y aparece etiquetada al instante (ver "Detección en vivo").
- **Revisar y corregir**: marca cada alerta como `Autorizado`, `Desestimado` o `Escalado` y presiona `Aplicar correcciones`. El backend guarda el feedback, ajusta el estado aprendido (umbral del IF y pesos de las reglas) y actualiza la cola, sin reentrenar el modelo.

## Detección en vivo

Cuando llega una transacción nueva, **se reutiliza el mismo pipeline de features y el Isolation Forest ya entrenado** para puntuarla en el momento (no se reentrena el modelo). El proceso reconstruye las features sobre la **ventana del último año** (`WINDOW_START = 2025-03-11`), aplica las reglas de negocio y el estado aprendido, y guarda la transacción ya etiquetada en la base de datos. La latencia es de unos segundos, suficiente para auditoría intra-día.

El modelo combina:

- **Isolation Forest** (no supervisado) sobre las features `feat_*`, para detectar patrones atípicos desconocidos.
- **Reglas de negocio tipadas** (`reglas.py`) + una regla de método de pago, para las inconsistencias conocidas y explicables.
- **SHAP** para explicar, por cada alerta del IF, qué features la marcaron.
- **Feedback del auditor**, que baja la prioridad de los patrones que se desestiman con frecuencia.

## Arquitectura de `model_final`

`model_final/` es la carpeta autónoma del modelo final:

- `app.py`: backend Flask y API del dashboard (consulta, inserción y corrección de transacciones).
- `dashboard/`: interfaz web para revisar alertas, ver transacciones y estadísticas.
- `scoring.py`: scoring en vivo de transacciones nuevas (reutiliza el pipeline de features + IF guardado).
- `db.py`: capa de persistencia (SQLite local o Azure SQL); consultas, filtros y estadísticas.
- `migrate_to_sql.py`: siembra/inicializa la base de datos desde los artefactos locales.
- `model.py`: aplica el estado aprendido y genera la cola de revisión.
- `adapt.py` y `feedback.py`: convierten las revisiones del auditor en ajustes de umbral y pesos.
- `reglas.py`: reglas de negocio explicables.
- `train_model.py`: reentrenamiento batch del modelo final.
- `data/`, `training_data/` y `output/`: datos y artefactos necesarios para correr la demo.

## Base de datos

- **Local (por defecto)**: SQLite en `model_final/output/finanom_demo.sqlite`, creado y sembrado automáticamente. Para reiniciar el estado de la demo, borra ese archivo (se regenera en el siguiente arranque).
- **Azure SQL u otra base**: define la variable de entorno `DATABASE_URL` (o `SQLALCHEMY_DATABASE_URI`) con una URL compatible con SQLAlchemy y siembra la tabla:

  ```bash
  DATABASE_URL='mssql+pyodbc://...' uv run python model_final/migrate_to_sql.py
  ```

## Despliegue (Docker / Azure)

El repo incluye un `Dockerfile` listo para producción (sirve con `gunicorn`, lee el puerto de la variable `PORT` e incluye el driver `msodbcsql18` para Azure SQL).

```bash
docker build -t finanom .
docker run -p 8000:8000 finanom
```

La guía completa para publicar el dashboard en Azure (Container Registry + App Service o Container Apps, conexión a Azure SQL y configuración de puertos) está en **[`docs/azure_deploy.md`](docs/azure_deploy.md)**.

## Reconstruir desde datos crudos

Usa esto solo si quieres regenerar todo desde `tablas_parquet/`.

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

4. Regenera los datos del dashboard y levanta la app (vuelve a sembrar la base al arrancar):

   ```bash
   uv run python model_final/build_dashboard.py
   uv run python model_final/app.py
   ```

> El flujo de entrenamiento también está orquestado con Prefect en `pipeline/finanom_training_flow.py`.

## Comandos útiles

```bash
uv run python model_final/app.py                 # demo (dashboard + API)
uv run python model_final/migrate_to_sql.py      # (re)sembrar la base de datos
uv run python model_final/build_dashboard.py     # regenerar datos del dashboard
uv run python model_final/train_model.py         # reentrenar el modelo final
uv run ruff check model_final pyproject.toml     # lint
```
