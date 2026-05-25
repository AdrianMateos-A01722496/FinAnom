# Model card — FINANOM (deteccion de inconsistencias)

Modelo HIBRIDO de apoyo al auditor nocturno: Isolation Forest no supervisado + reglas de negocio tipadas. Generado por `model_training/train_model.py`.

## 1. Entrada

- Matriz: `training_data/X_modelo.parquet` (1,145,526 filas × 63 features `feat_*`).
- Trazabilidad: columnas `trace_*` de `training_data/transacciones_modelado.parquet`.
- Campos crudos para reglas: `h_for_pgo`, `t_codigo`, `t_monto`, `t_referencia` de la base limpia.

## 2. Modelo

- Isolation Forest: n_estimators=200, max_samples=8192, contamination=0.01, random_state=42.
- Entrenado en el 80% temporal mas antiguo; score = -decision_function.
- Cola de revision: 23,429 (2.05%) = IF top-1% OR reglas criticas (todas) OR reglas amplias (top-K por score).

## 3. Reglas de negocio (mapeo al dolor del hotel)

| Regla | Tipo | Inconsistencia de negocio |
| ----- | ---- | ------------------------- |
| `metodo_pago_inconsistente` | critica | Metodo de pago mal usado (Visa↔Amex) |
| `monto_atipico` | critica | Monto/factura incorrecto |
| `posible_duplicado` | critica | Cargo duplicado / doble posteo |
| `reemplazo_monto` | amplia (top-K) | Reemplazo solicitado por monto incorrecto |
| `modificacion_no_autorizada` | amplia (top-K) | Modificacion no autorizada |
| `cargo_fuera_estancia` | amplia (top-K) | Cargo fuera de la estancia |

## 4. Esquema del reporte de revision (`output/reporte_revision.parquet`)

| Columna | Tipo | Descripcion |
| ------- | ---- | ----------- |
| `trace_*` | id/datetime | Trazabilidad: folio, transaccion, referencia, cuarto, codigo, timestamp. |
| `anomaly_score` | float | Score del IF (mayor = mas anomalo). |
| `anomaly_pct` | float | Percentil del score (0-1). |
| `is_anomaly_if` | bool | Marcada por el IF (top-presupuesto). |
| `regla_*` | bool | Una columna por regla de negocio. |
| `n_reglas` | int | Cuantas reglas dispararon. |
| `is_anomaly` | bool | Entra a la cola de revision. |
| `tipo_inconsistencia` | str | Reglas que dispararon (o 'atipico_general'). |
| `motivos` | str | Razon legible de la alerta. |
| `top_features` | str | Features que mas la aislaron (explicabilidad). |
| `accion_sugerida` | str | Accion propuesta para el auditor. |
| `requiere_aprobacion` | bool | Siempre True: la correccion exige aprobacion (human-in-the-loop). |

## 5. Scoring de transacciones nuevas (intra-dia)

`output/scorer_bundle.joblib` empaqueta el modelo + columnas + umbral. Para scorear transacciones nuevas hay que pasarlas antes por el pipeline de datos (consolidacion → limpieza → modelado de datos) para obtener las `feat_*`. La persistencia de los transformadores de FE para streaming puro es un paso siguiente.

## 6. Bucle de feedback del auditor

Almacen `output/feedback_labels.csv` (trace_row_id, decision, revisor, timestamp_revision, nota). `apply_feedback()` mide precision y sugiere umbral; el re-ranker supervisado queda como TODO (diseno listo, no implementado en esta fase).

## 7. Limitaciones

- No supervisado: sin etiquetas reales; calidad validada por inyeccion/inspeccion.
- Pago: Amex separable; Visa vs Mastercard no (ambos `TARCRE`).
- Sin datos de factura (vacios en MX): 'factura incorrecta' se aproxima via monto/reemplazo.
- Duplicados/modificaciones son comunes; se priorizan casos extremos/co-ocurrentes.