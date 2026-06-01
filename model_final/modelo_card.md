# Model card — FINANOM (modelo hibrido consolidado)

Apoyo al auditor nocturno. Consolida el trabajo de los tres integrantes:

- **Isolation Forest** no supervisado con **muestreo estratificado** y **umbral adaptativo** (copia local de Rogelio).
- **Reglas de negocio tipadas**: motor local de reglas de Tony (8 detectores) + regla de **metodo de pago** Visa<->Amex (Adrian).
- **Explicabilidad SHAP** (TreeExplainer, copia local de Rogelio) sobre las filas del IF.

## 1. Entrada

- Matriz: `training_data/X_modelo.parquet` (1,145,526 filas × 63 `feat_*`).
- Reglas: base limpia `model_final/data/transacciones_limpio.parquet` (columnas crudas).
- Trazabilidad: `trace_*` de `training_data/transacciones_modelado.parquet`.

## 2. Modelo

- IsolationForest: n_estimators=200, max_samples='auto', random_state=42.
- Entrenado en el 80% temporal mas antiguo (muestra estratificada de 200,000).
- Umbral IF adaptativo (mayor salto en [1.0%–5.0%]) = -0.5838 sobre score_samples.
- Cola de revision: 47,087 (4.11%) = rule_score>=60 ∪ IF ∪ metodo_pago (severidad ALTO/CRITICO), rankeada por severidad y score.

## 3. Tipos de inconsistencia (mapeo al negocio)

| Cluster | Inconsistencia de negocio |
| ------- | ------------------------- |
| `DUPLICADO` | Cargo duplicado / doble posteo |
| `SIGNO_CONTABLE` | Monto con signo contable inesperado |
| `FUERA_DE_ESTANCIA` | Cargo fuera de la ventana de estancia |
| `MONTO_ATIPICO` | Monto fuera de rango para su concepto |
| `CANCELACION_SOSPECHOSA` | Cancelacion/reposteo sospechoso |
| `CONTEXTO_RESERVACION` | Inconsistencia con el contexto de la reserva |
| `PAGO_PROVEEDOR_SOSPECHOSO` | Egreso/pago a proveedor sospechoso |
| `METODO_PAGO` | Metodo de pago mal usado (Visa<->Amex) |

## 4. Esquema del reporte (`output/reporte_revision.parquet`)

| Columna | Descripcion |
| ------- | ----------- |
| `trace_*` | Trazabilidad (folio, transaccion, referencia, cuarto, codigo, timestamp). |
| `anomaly_score` | Score del IF (mayor = mas anomalo). |
| `anomaly_pct` | Percentil del score (0-1). |
| `score_samples` | Score crudo del IF (menor = mas anomalo). |
| `is_anomaly_if` | Marcada por el IF. |
| `rule_score` | Puntaje acumulado de las reglas (Tony + metodo_pago). |
| `severidad` | CRITICO / ALTO / MEDIO / BAJO. |
| `is_anomaly` | Entra a la cola de revision (severidad ALTO/CRITICO). |
| `tipo_inconsistencia` | Clusters de regla (o ATIPICO_IF). |
| `motivos` | Razon legible para el auditor (mensaje de Tony o evidencia SHAP). |
| `evidencia_shap` | Top features SHAP que aislaron la transaccion (filas del IF). |
| `requiere_aprobacion` | Siempre True: la correccion exige aprobacion (human-in-the-loop). |

## 5. Bucle de feedback del auditor (el modelo que aprende)

Primitivas en `model_final/feedback.py` (mecanismo de **adaptacion de umbral/pesos**): `feedback_metrics()` mide precision, `suggest_threshold()` ajusta el umbral del IF y `suggest_rule_weights()` aprende un factor de confianza por regla. El lazo COMPLETO (aplicar lo aprendido y regenerar la cola, sin reentrenar) vive en `model_final/` y se presenta en el dashboard.

## 6. Limitaciones

- No supervisado: calidad validada por inyeccion/overlap, no por etiquetas reales.
- Metodo de pago: Amex (`AMEXCO`) separable; Visa vs Mastercard no (ambos `TARCRE`).
- Sin datos de factura (vacios en MX): se aproxima via monto/duplicado/reposteo.
- El IF subpondera anomalias de pocas dimensiones; las reglas cubren ese hueco.
