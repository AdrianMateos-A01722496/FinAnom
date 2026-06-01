# Reporte de evaluacion — Modelo FINANOM (hibrido consolidado)

Generado por `model_final/train_model.py`. Consolida IF (muestreo estratificado + umbral adaptativo, Rogelio), reglas de negocio (Tony) y la regla de metodo de pago (Adrian), con explicabilidad SHAP (Rogelio).

## Resumen

- Transacciones evaluadas: 1,145,526
- Marcadas para revision: 47,087 (4.11%)
- Aprox. alertas/dia: 25.5
- Umbral IF (score_samples): -0.5838 | corte temporal: 2025-02-14 04:49:00

## Composicion por severidad

| Severidad | Transacciones | % |
| --------- | ------------- | - |
| CRITICO | 18,917 | 1.65% |
| ALTO | 28,170 | 2.46% |
| MEDIO | 216,395 | 18.89% |
| BAJO | 882,044 | 77.00% |

## Composicion de la cola por tipo (cluster)

| Tipo | Marcadas | Inconsistencia de negocio |
| ---- | -------- | ------------------------- |
| `CANCELACION_SOSPECHOSA` | 36,249 | Cancelacion/reposteo sospechoso |
| `DUPLICADO` | 34,358 | Cargo duplicado / doble posteo |
| `CONTEXTO_RESERVACION` | 9,398 | Inconsistencia con el contexto de la reserva |
| `FUERA_DE_ESTANCIA` | 6,682 | Cargo fuera de la ventana de estancia |
| `MONTO_ATIPICO` | 5,546 | Monto fuera de rango para su concepto |
| `ATIPICO_IF` | 1,404 | - |
| `METODO_PAGO` | 28 | Metodo de pago mal usado (Visa<->Amex) |
| `PAGO_PROVEEDOR_SOSPECHOSO` | 15 | Egreso/pago a proveedor sospechoso |
| `SIGNO_CONTABLE` | 8 | Monto con signo contable inesperado |

## Iteracion 1 — Inyeccion sintetica (recall del IF)

Anomalias extremas en 1-2 dimensiones inyectadas en filas normales. El IF global las diluye (swamping entre 63 features); por eso NO basta solo. Las reglas tipadas las capturan por construccion -> arquitectura hibrida.

| Tipo inyectado | Recall IF |
| -------------- | --------- |
| monto_inflado | 0.00% |
| duplicado_extremo | 0.00% |
| cargo_fuera_estancia | 0.00% |
| modificacion | 0.00% |
| promedio | 0.00% |

## Iteracion 2 — Estabilidad temporal

| Periodo | p50 | p05 | p01 | tasa marcado IF |
| ------- | --- | --- | --- | --------------- |
| entrenamiento | -0.454 | -0.545 | -0.589 | 1.30% |
| reciente | -0.456 | -0.543 | -0.587 | 1.19% |

## Iteracion 3 — Overlap IF vs reglas (por que es hibrido)

- IF-flagged: 14,692
- Reglas operativas: 232,558
- Interseccion: 11,004
- Solo reglas (el IF NO las priorizaba): 221,554
- Solo IF (atipicos sin regla): 3,688

Las dos capas son mayormente disjuntas: cada una aporta alertas que la otra no ve. El IF da los atipicos multidimensionales desconocidos; las reglas, los tipos de negocio concretos y explicables.
