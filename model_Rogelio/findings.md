# Hallazgos del Modelo de Anomalias — FINANOM

> Generado por `model_Rogelio/train.py`

---

## 1. Metodologia

**Algoritmo:** Isolation Forest (scikit-learn 1.8.0)

Isolation Forest construye arboles de aislamiento aleatorios; las muestras que se aíslan en pocas particiones tienen scores mas bajos (mas anomalos). Ventajas para este problema:

- Escala a millones de filas (complejidad O(n log n))
- No asume distribucion de los datos — adecuado para la mezcla de flags binarios, frecuencias y magnitudes escaladas
- Compatible con SHAP TreeExplainer para explicaciones por transaccion
- Alternativas descartadas: LOF (O(n²)), OCSVM (no escala >100k filas), autoencoders (requiere infraestructura adicional)

**Hiperparametros finales:**

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `n_estimators` | 200 | Doble del proxy (100); scores mas estables |
| `max_samples` | auto (256) | Default sklearn; suficiente para capturar estructura local |
| `contamination` | auto | Umbral determinado post-hoc desde distribucion |

---

## 2. Muestra de entrenamiento

| Item | Valor |
|------|-------|
| Corpus total | 1,145,526 transacciones |
| Muestra de entrenamiento | 200,000 filas estratificadas |
| Scoring aplicado sobre | 1,145,526 filas (corpus completo) |

**Estratificacion:** proporcional por combinacion de `feat_es_abono`, `feat_cargo_cancelado`, `feat_dup_mismo_dia_flag`, `feat_es_renta`. Garantiza representacion de tipos de transaccion raros en la muestra de entrenamiento.

---

## 3. Distribucion de scores

> Ver: `output/score_distribution.png`

Scores de `score_samples` — valores mas negativos indican mayor anomalia.

| Percentil | Score |
|-----------|-------|
| p0.5 | -0.6030 |
| p1 | -0.5897 |
| p2 | -0.5725 |
| p3 | -0.5597 |
| p5 | -0.5421 |
| p10 | -0.5158 |
| p50 | -0.4494 |

La mayoria de transacciones normales se concentra en el rango superior; la cola izquierda es larga y dispersa, tipica de datos financieros con eventos raros.

---

## 4. Umbral seleccionado

| Item | Valor |
|------|-------|
| Umbral | `-0.6019` |
| Criterio | Mayor salto en rango [0.5%–5.0%] (gap=0.0000 vs mediana_diff=0.0000) |
| Transacciones flaggeadas | 6,126 (0.53% del total) |

---

## 5. Anomalias detectadas: patrones clave

### Indicadores operativos — anomalias vs. total

| Indicador | Total | Solo anomalias |
|-----------|-------|----------------|
| Duplicados mismo dia | 27.5% | 42.2% |
| Cargos cancelados | 10.2% | 49.3% |
| Cargo fuera de estancia | 1.1% | 47.7% |
| Transacciones de madrugada | 66.3% | 6.4% |
| Monto negativo sin abono | 4.7% | 10.0% |
| Modificado por usuario distinto | 13.7% | 47.3% |

### Top 10 codigos por tasa de anomalia (minimo 100 transacciones)

> Ver: `output/anomaly_by_codigo.png`

| Codigo | Total | Anomalias | Tasa |
|--------|-------|-----------|------|
| `CANCXC` | 2,045 | 641 | 31.3% |
| `EMPCXC` | 161 | 34 | 21.1% |
| `AJCUPN` | 402 | 70 | 17.4% |
| `CANXFA` | 10,661 | 1,632 | 15.3% |
| `RENAJU` | 3,693 | 393 | 10.6% |
| `CXC` | 3,578 | 315 | 8.8% |
| `CUPON` | 5,118 | 304 | 5.9% |
| `XFAC` | 10,610 | 426 | 4.0% |
| `EFE` | 24,743 | 822 | 3.3% |
| `DEVEFE` | 149 | 3 | 2.0% |

### Patrones temporales

> Ver: `output/anomaly_by_hour.png`

- **Hora pico de anomalias:** 19:00 h — 4.0% de transacciones en esa hora son anomalas

---

## 6. Importancia de features — SHAP

> Ver: `output/shap_bar.png` · `output/shap_summary.png`

**Interpretacion del valor SHAP:**
- **Negativo** → la feature empuja el score hacia abajo (mas anomalo)
- **Positivo** → la feature empuja el score hacia arriba (mas normal)
- **|SHAP| alto** → la feature tiene gran influencia en ese score, en cualquier direccion

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `feat_monto_vs_tarifa_total_ratio` | 0.4215 |
| 2 | `feat_dias_desde_llegada_scaled` | 0.3288 |
| 3 | `feat_noches_delta_scaled` | 0.2994 |
| 4 | `feat_monto_vs_tarifa_ratio` | 0.2654 |
| 5 | `feat_es_renta` | 0.2409 |
| 6 | `feat_usuario_mod_distinto` | 0.2017 |
| 7 | `feat_usuario_modificado` | 0.2014 |
| 8 | `feat_cargo_fuera_estancia` | 0.1879 |
| 9 | `feat_cargo_cancelado` | 0.1621 |
| 10 | `feat_es_madrugada` | 0.1458 |
| 11 | `feat_t_folio_ext_freq` | 0.1383 |
| 12 | `feat_t_codigo_freq` | 0.1345 |
| 13 | `feat_obs_len_log` | 0.1286 |
| 14 | `feat_dup_mismo_dia_flag` | 0.1283 |
| 15 | `feat_cancelacion_sin_marca` | 0.1241 |

---

## 7. Artefactos generados

| Archivo | Descripcion |
|---------|-------------|
| `artifacts/isolation_forest.joblib` | Modelo serializado (cargable con `joblib.load`) |
| `artifacts/scored_transactions.parquet` | 1,145,526 filas con `anomaly_score`, `anomaly_rank`, `anomaly_flag`, `anomaly_score_percentile` |
| `artifacts/shap_anomalies.parquet` | 6,126 anomalias con columnas `shap_*` para cada feature |
| `output/score_distribution.png` | Histograma completo y zoom cola anomala con umbral marcado |
| `output/shap_bar.png` | Top 20 features por mean |SHAP| en anomalias |
| `output/shap_summary.png` | Beeswarm SHAP: distribucion de contribuciones por feature y valor |
| `output/anomaly_by_hour.png` | Conteo y tasa de anomalia por hora del dia |
| `output/anomaly_by_codigo.png` | Tasa de anomalia por codigo contable (top 20) |

---

## 8. Como explorar los resultados

```python
import pandas as pd

# Cargar todas las transacciones con score
scored = pd.read_parquet("model_Rogelio/artifacts/scored_transactions.parquet")

# Ver las 50 mas anomalas
top = scored.sort_values("anomaly_rank").head(50)

# Cargar anomalias con SHAP
shap_anom = pd.read_parquet("model_Rogelio/artifacts/shap_anomalies.parquet")

# Para una transaccion especifica, ver que features la hicieron anomala
tx = shap_anom[shap_anom["trace_t_transaccion"] == "TU_ID"]
shap_cols = [c for c in tx.columns if c.startswith("shap_")]
tx[shap_cols].T.sort_values(by=tx.index[0])
```

**Flujo recomendado para auditoria:**
1. Ordenar `scored_transactions.parquet` por `anomaly_rank` (rank 1 = mas anomalo)
2. Priorizar transacciones con `anomaly_flag=True` y `feat_dup_mismo_dia_flag=1` — candidatos directos a correccion
3. Para cada transaccion flaggeada, revisar columnas `shap_*` en `shap_anomalies.parquet` para entender la causa
4. Atender especialmente la hora pico (19:00 h) y los codigos con mayor tasa de anomalia

---

## 9. Recomendaciones de ajuste

- **Si el umbral parece alto (demasiadas anomalias):** cambiar `TRAIN_SAMPLE_SIZE` o aplicar un filtro adicional por `anomaly_rank <= N`.
- **Para nuevas transacciones:** cargar `isolation_forest.joblib` con `joblib.load` y llamar `model.score_samples(X_nueva)`.
- **Etiquetado futuro:** si el equipo de auditoria valida anomalias, se puede entrenar un clasificador supervisado sobre `shap_anomalies.parquet` usando las columnas `feat_*` y las etiquetas manuales.
