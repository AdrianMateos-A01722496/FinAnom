# FINANOM model_final

Carpeta autonoma para demo y ejecucion del modelo final.

## Que hace

El modelo final prioriza transacciones hoteleras que ameritan revision. Combina un
modelo no supervisado, reglas de negocio explicables y un lazo de feedback que ajusta
la sensibilidad cuando el auditor confirma o desestima alertas.

## Arquitectura

`model_final` contiene todo lo necesario para correr la demo:

- `app.py`: backend Flask que sirve el dashboard y recibe correcciones.
- `dashboard/`: interfaz para revisar alertas y aplicar feedback.
- `model.py`: aplica el estado aprendido y genera la cola de revision.
- `adapt.py` y `feedback.py`: convierten decisiones del auditor en ajustes de umbral y pesos.
- `reglas.py`: reglas de negocio explicables.
- `train_model.py`: entrenamiento del modelo final.
- `data/`, `training_data/` y `output/`: datos y artefactos necesarios para ejecutar sin depender de otras carpetas.

La cola combina senales del modelo no supervisado con reglas de negocio. Cada alerta
tiene severidad, motivo legible y estado de revision.

## Correr dashboard con aprendizaje

Desde la raiz del repo:

```bash
uv run python model_final/app.py
```

Abrir: <http://127.0.0.1:5000>

El flujo principal ya no requiere CSV ni terminal:

1. Revisar alertas en el dashboard.
2. Marcar `Autorizado`, `Desestimado` o `Escalado`.
3. Presionar `Aplicar correcciones`.

El backend guarda las revisiones en `model_final/output/feedback_labels.csv`, actualiza
`model_final/output/feedback_state.json` y regenera
`model_final/dashboard/data/anomalies.json`.

## Regenerar datos del dashboard

```bash
uv run python model_final/build_dashboard.py
```

## Reentrenar modelo final

```bash
uv run python model_final/train_model.py
```

Este script usa solo archivos dentro de `model_final`.

## Duplicados

La regla `DUPLICADO` exige duplicado de alta confianza: mismo folio, subfolio, cuarto,
codigo, naturaleza contable, monto y minuto. La coincidencia por dia queda como
contexto, porque puede representar cargos legitimos repetidos.
