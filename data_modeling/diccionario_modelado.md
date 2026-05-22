# Diccionario — Base MODELADA FINANOM

Generado por `data_modeling/data_modeling.py` a partir de `data_cleaning/output/transacciones_limpio.parquet`.

## 1. Resumen

- **Archivo principal**: `data_modeling/output/transacciones_modelado.parquet`
- **Matriz numerica pura**: `data_modeling/output/X_modelo.parquet`
- **Filas**: 1,145,526 (una por transaccion; no se eliminaron filas)
- **Columnas totales**: 72
- **Columnas de trazabilidad**: 9
- **Features finales (`feat_*`)**: 63
- **Nulos en features**: 0
- **Uso recomendado**: entrenar modelos solo con las columnas `feat_*` o con `X_modelo.parquet`.

## 2. Decisiones principales

- No se codifican IDs crudos (`folio`, `referencia`, `transaccion`, `cuarto`) como features; solo se usan para trazabilidad o agregados derivados.
- Las categoricas se codifican por frecuencia relativa. Esto evita una matriz one-hot grande y permite que categorias raras sigan siendo visibles para Isolation Forest.
- Los montos se transforman con `log1p(abs(x))` y robust z por (`t_codigo`, `t_carabo`) para comparar conceptos financieros en su propio contexto.
- Los nulos de reservacion se imputan a valores neutros despues de generar flags de contexto; `feat_tiene_reservacion` explica cuando falta ese bloque.
- `t_observaciones` no se exporta crudo por posible PII; se transforma en longitud y keywords operativas.

## 3. Columnas de trazabilidad (NO features)

| Columna | Tipo | Descripcion | Transformacion |
| ------- | ---- | ----------- | -------------- |
| `trace_row_id` | int64 | Identificador estable: posicion de la fila en la base limpia. | Derivado como indice entero; no entra al modelo. |
| `trace_t_folio` | string | Folio de la cuenta donde vive la transaccion. | Copia de `t_folio`; no entra al modelo. |
| `trace_t_folio_ext` | string | Extension/subcuenta del folio. | Copia de `t_folio_ext`; no entra al modelo. |
| `trace_t_referencia` | string | Referencia operativa del movimiento. | Copia de `t_referencia`; no entra al modelo. |
| `trace_t_transaccion` | string | Numero correlativo de la transaccion. | Copia de `t_transaccion`; no entra al modelo. |
| `trace_t_cve_res` | string | Llave de reservacion enlazada, cuando existe. | Copia de `t_cve_res`; no entra al modelo. |
| `trace_t_cuarto` | string | Cuarto/habitacion asociado al cargo. | Copia de `t_cuarto`; no entra al modelo. |
| `trace_t_codigo` | string | Codigo original del concepto financiero. | Copia de `t_codigo`; el modelo usa su frecuencia y escalados por codigo. |
| `trace_t_timestamp` | datetime64[us] | Timestamp original del cargo. | Copia de `t_timestamp`; el modelo usa features temporales derivadas. |

## 4. Features finales

| Columna | Tipo | Grupo | Descripcion | Transformacion aplicada |
| ------- | ---- | ----- | ----------- | ----------------------- |
| `feat_es_abono` | int8 | Flag contable | Movimiento marcado como abono (`t_carabo == 1`). | Binario 0/1. |
| `feat_cargo_cancelado` | int8 | Flag operativo | Transaccion marcada como cancelada. | Binario 0/1 desde `t_tra_cancelada == 1`. |
| `feat_cancelacion_sin_marca` | int8 | Flag de calidad | Transaccion sin marca explicita de cancelacion. | Binario 0/1 desde nulo en `t_tra_cancelada`. |
| `feat_es_split` | int8 | Flag operativo | Cargo generado como split. | Binario 0/1 desde `es_split`; se conserva aunque sea raro por relevancia de doble conteo. |
| `feat_es_renta` | int8 | Flag financiero | Cargo de renta/hospedaje. | Binario 0/1 desde `es_renta`. |
| `feat_tiene_reservacion` | int8 | Flag de contexto | La transaccion tiene contexto de reservacion enlazado. | Binario 0/1 desde `tiene_reservacion`. |
| `feat_usuario_modificado` | int8 | Flag operativo | La transaccion tiene usuario modificador. | Binario 0/1 desde `t_usuario_mod` no nulo. |
| `feat_usuario_mod_distinto` | int8 | Flag operativo | El usuario modificador existe y es distinto al usuario creador. | Binario 0/1 comparando `t_usuario_mod` contra `t_usuario`. |
| `feat_monto_negativo_sin_abono` | int8 | Regla financiera | Monto negativo en movimiento que no esta marcado como abono. | Binario 0/1; captura inconsistencias de signo contable. |
| `feat_monto_positivo_en_abono` | int8 | Regla financiera | Monto positivo en movimiento marcado como abono. | Binario 0/1; captura inconsistencias de signo contable. |
| `feat_monto_abs_log` | float32 | Monto | Magnitud del monto sin signo. | `log1p(abs(t_monto))`; reduce cola larga sin perder orden. |
| `feat_impuesto_abs_log` | float32 | Monto | Magnitud del impuesto sin signo. | `log1p(abs(t_impuesto))`; reduce cola larga. |
| `feat_propina_abs_log` | float32 | Monto | Magnitud de la propina sin signo. | `log1p(abs(t_propina))`; reduce cola larga. |
| `feat_monto_z_codigo_carabo` | float32 | Monto escalado | Monto relativo a su concepto y naturaleza contable. | Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10]. |
| `feat_impuesto_z_codigo_carabo` | float32 | Monto escalado | Impuesto relativo a su concepto y naturaleza contable. | Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10]. |
| `feat_propina_z_codigo_carabo` | float32 | Monto escalado | Propina relativa a su concepto y naturaleza contable. | Robust z por (`t_codigo`, `t_carabo`) usando mediana/IQR, clip [-10, 10]. |
| `feat_impuesto_ratio_abs` | float32 | Ratio financiero | Impuesto absoluto contra monto absoluto. | `abs(t_impuesto) / abs(t_monto)`, imputado a 0 si monto=0, clip [0, 2]. |
| `feat_propina_ratio_abs` | float32 | Ratio financiero | Propina absoluta contra monto absoluto. | `abs(t_propina) / abs(t_monto)`, imputado a 0 si monto=0, clip [0, 2]. |
| `feat_monto_vs_tarifa_ratio` | float32 | Ratio reserva | Monto absoluto comparado contra tarifa diaria de la reserva. | `abs(t_monto) / abs(h_tfa)`, imputado a 0 si no aplica, clip [0, 10]. |
| `feat_monto_vs_tarifa_total_ratio` | float32 | Ratio reserva | Monto absoluto comparado contra tarifa total de la reserva. | `abs(t_monto) / abs(h_tfa_total)`, imputado a 0 si no aplica, clip [0, 10]. |
| `feat_t_noches_scaled` | float32 | Estancia | Noches asociadas al cargo. | Robust scale global de `t_noches`, clip [-10, 10]. |
| `feat_h_num_per_scaled` | float32 | Reservacion | Personas esperadas en la reservacion. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_h_num_noc_scaled` | float32 | Reservacion | Noches esperadas en la reservacion. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_h_tfa_scaled` | float32 | Reservacion | Tarifa diaria de la reservacion. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_h_tfa_total_scaled` | float32 | Reservacion | Tarifa total de la reservacion. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_h_tarifa_forzada_scaled` | float32 | Reservacion | Tarifa forzada manualmente. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_h_dep_sol_scaled` | float32 | Reservacion | Deposito solicitado. | Robust scale global con nulos imputados a mediana antes de escalar. |
| `feat_noches_delta_scaled` | float32 | Consistencia | Diferencia entre noches del cargo y noches de la reservacion. | Robust scale global de `t_noches - h_num_noc`; nulos imputados a 0. |
| `feat_hora_sin` | float32 | Temporal | Hora del cargo en ciclo diario. | `sin(2*pi*hora_decimal/24)`. |
| `feat_hora_cos` | float32 | Temporal | Hora del cargo en ciclo diario. | `cos(2*pi*hora_decimal/24)`. |
| `feat_dia_semana_sin` | float32 | Temporal | Dia de semana en ciclo semanal. | `sin(2*pi*dia_semana/7)`. |
| `feat_dia_semana_cos` | float32 | Temporal | Dia de semana en ciclo semanal. | `cos(2*pi*dia_semana/7)`. |
| `feat_mes_sin` | float32 | Temporal | Mes del cargo en ciclo anual. | `sin(2*pi*mes/12)`. |
| `feat_mes_cos` | float32 | Temporal | Mes del cargo en ciclo anual. | `cos(2*pi*mes/12)`. |
| `feat_es_fin_semana` | int8 | Temporal | Cargo registrado en sabado o domingo. | Binario 0/1 desde `t_timestamp.dayofweek >= 5`. |
| `feat_es_madrugada` | int8 | Temporal | Cargo registrado entre 00:00 y 05:59. | Binario 0/1 desde hora del timestamp. |
| `feat_dias_desde_llegada_scaled` | float32 | Temporal reserva | Dias entre llegada y cargo. | Robust scale global, nulos imputados a 0, clip [-10, 10]. |
| `feat_dias_hasta_salida_scaled` | float32 | Temporal reserva | Dias entre cargo y salida. | Robust scale global, nulos imputados a 0, clip [-10, 10]. |
| `feat_cargo_antes_llegada` | int8 | Temporal reserva | Cargo fechado antes de la llegada. | Binario 0/1 solo si existe reservacion. |
| `feat_cargo_fuera_estancia` | int8 | Temporal reserva | Cargo antes de llegada o despues de salida. | Binario 0/1; senal directa para revision. |
| `feat_dup_mismo_dia_flag` | int8 | Duplicados | Existe mas de un cargo con mismo folio, subfolio, codigo, monto y dia. | Binario 0/1; captura el dolor principal de cargos duplicados. |
| `feat_dup_mismo_dia_log` | float32 | Duplicados | Intensidad de duplicados exactos por dia. | `log1p(conteo - 1)` por folio/subfolio/codigo/monto/dia. |
| `feat_dup_mismo_minuto_flag` | int8 | Duplicados | Existe mas de un cargo identico incluso en el mismo minuto. | Binario 0/1 por folio/subfolio/codigo/monto/timestamp. |
| `feat_folio_codigo_dia_count_log` | float32 | Densidad folio | Repeticion del mismo concepto en el folio durante el dia. | `log1p(conteo)` por folio/subfolio/codigo/dia. |
| `feat_folio_dia_movimientos_log` | float32 | Densidad folio | Cantidad total de movimientos del folio en el dia. | `log1p(conteo)` por folio/subfolio/dia. |
| `feat_folio_total_movimientos_log` | float32 | Densidad folio | Cantidad total de movimientos historicos del folio. | `log1p(conteo)` por folio/subfolio. |
| `feat_t_codigo_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_t_folio_ext_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_t_usuario_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_tpo_hab_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_tpo_hsp_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_seg_mer_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_cod_age_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_tpo_plan_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_h_for_pgo_freq` | float32 | Encoding | Feature codificada por frecuencia relativa de la categoria original. | Frecuencia relativa incluyendo nulos como categoria `__MISSING__`. |
| `feat_obs_missing` | int8 | Texto derivado | Observaciones vacias. | Binario 0/1; no se exporta texto crudo por posible PII. |
| `feat_obs_len_log` | float32 | Texto derivado | Longitud del texto de observaciones. | `log1p(numero de caracteres)`. |
| `feat_obs_kw_ajuste` | int8 | Texto derivado | Observacion menciona ajuste. | Binario 0/1 por keyword en `t_observaciones`. |
| `feat_obs_kw_cancelacion` | int8 | Texto derivado | Observacion menciona cancelacion. | Binario 0/1 por keyword en `t_observaciones`. |
| `feat_obs_kw_error` | int8 | Texto derivado | Observacion menciona error. | Binario 0/1 por keyword en `t_observaciones`. |
| `feat_obs_kw_cortesia` | int8 | Texto derivado | Observacion menciona cortesia/compensacion. | Binario 0/1 por keyword en `t_observaciones`. |
| `feat_obs_kw_deposito` | int8 | Texto derivado | Observacion menciona deposito. | Binario 0/1 por keyword en `t_observaciones`. |
| `feat_obs_kw_reembolso` | int8 | Texto derivado | Observacion menciona reembolso/devolucion. | Binario 0/1 por keyword en `t_observaciones`. |

## 5. Columnas descartadas en modelado

| Columna/familia | Motivo |
| --------------- | ------ |
| `t_observaciones` crudo | Puede contener nombres u otra PII; se reemplaza por longitud y keywords. |
| IDs crudos como feature | Identifican cuentas/movimientos, pero no generalizan; se conservan solo en `trace_*`. |
| `feat_falta_contexto_reserva` | Inverso perfecto de `feat_tiene_reservacion`; se conserva la version positiva. |
| `feat_t_usuario_mod_missing` | Inverso perfecto de `feat_usuario_modificado`; se conserva la version positiva. |
| `feat_impuesto_signo_distinto_monto` | Constante en la base actual; no aporta separacion al Isolation Forest. |
| `feat_propina_mayor_monto` | Constante en la base actual; no aporta separacion al Isolation Forest. |
| `feat_t_usuario_mod_freq` | Correlacion >= 0.985 con `feat_usuario_modificado`; la frecuencia queda dominada por el nulo. |
| `feat_desvio_iva_16_abs` | Correlacion >= 0.985 con `feat_impuesto_ratio_abs`; se conserva el ratio fiscal mas general. |
| `feat_cargo_despues_salida` | Correlacion >= 0.985 con `feat_cargo_fuera_estancia`; se conserva la regla agregada. |

## 6. Evaluacion recursiva

Ver `data_modeling/output/reporte_calidad_modelado.md` para correlaciones, features raras, Isolation Forest proxy e importancias.