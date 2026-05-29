# Documentación Técnica — Modelo FINANOM

> Cómo funciona el sistema de detección de anomalías: del dato crudo hasta la explicación en lenguaje contable.

---

## 1. Visión general

El sistema detecta transacciones financieras atípicas en el PMS del hotel. Para cada una de las 1,145,526 transacciones históricas, produce:

- Un **puntaje de anomalía** (entre −0.68 y −0.39 en este corpus)
- Una **bandera binaria** (`anomaly_flag`)
- Un **ranking global** (rank 1 = transacción más anómala de todo el corpus)
- Una **explicación en español** escrita en lenguaje contable
- Una **severidad** (Alta / Media / Informativa)

El proceso se divide en dos scripts independientes:

| Script | Propósito |
|--------|-----------|
| `train.py` | Entrena el modelo, puntúa el corpus completo, calcula SHAP |
| `explain.py` | Traduce los valores SHAP a texto legible para el auditor |

---

## 2. Datos de entrada

### `training_data/X_modelo.parquet`
- **1,145,526 filas × 63 columnas** — todas prefijadas con `feat_`
- Producido por la fase `data_modeling` del pipeline (read-only)
- Sin nulos, solo valores numéricos
- Mezcla de tipos: flags binarios (0/1), frecuencias relativas, valores log-transformados, ratios, y representaciones cíclicas de la hora

### `training_data/transacciones_modelado.parquet`
- Mismas 1,145,526 filas, alineadas por posición con `X_modelo`
- Columnas `trace_*` de trazabilidad: folio, cuarto, código de concepto, timestamp, usuario, referencia, etc.
- `trace_row_id` es el índice posicional — se usa para hacer lookup con `iloc[]`

---

## 3. Catálogo de features relevantes

Las 63 features capturan el comportamiento financiero en varias dimensiones. Las más influyentes en el modelo:

### Flags de duplicación
| Feature | Qué indica |
|---------|-----------|
| `feat_dup_mismo_dia_flag` | Hay otro cargo con mismo concepto y monto en este folio el mismo día |
| `feat_dup_mismo_minuto_flag` | El PMS registró el cargo exactamente en el mismo minuto que otro cargo del mismo folio |
| `feat_dup_mismo_dia_log` | `log1p(conteo − 1)` — cuántos duplicados hay ese día |

### Monto y relaciones financieras
| Feature | Qué indica |
|---------|-----------|
| `feat_monto_abs_log` | `log1p(|monto|)` — magnitud del cargo sin signo |
| `feat_monto_z_codigo_carabo` | Z-score del monto respecto al rango histórico del concepto y cargo/abono |
| `feat_monto_vs_tarifa_ratio` | Monto ÷ tarifa diaria de la reservación |
| `feat_monto_negativo_sin_abono` | Monto negativo pero sin marca de abono en el PMS |
| `feat_monto_positivo_en_abono` | Monto positivo en movimiento marcado como abono |
| `feat_impuesto_ratio_abs` | IVA ÷ monto absoluto |
| `feat_propina_ratio_abs` | Propina ÷ monto absoluto |

### Temporales
| Feature | Qué indica |
|---------|-----------|
| `feat_es_madrugada` | Timestamp entre 00:00 y 05:59 |
| `feat_es_fin_semana` | Registrado en sábado o domingo |
| `feat_hora_sin` / `feat_hora_cos` | Representación cíclica de la hora (sin/cos) para que 23h y 01h sean cercanas |
| `feat_cargo_antes_llegada` | Cargo fechado antes del check-in de la reservación |
| `feat_cargo_fuera_estancia` | Cargo fuera del rango de estancia (antes llegada o después salida) |
| `feat_noches_delta_scaled` | Diferencia entre noches del cargo y noches de la reservación, escalada |

### Densidad y frecuencia
| Feature | Qué indica |
|---------|-----------|
| `feat_folio_codigo_dia_count_log` | `log1p(n)` — cuántos cargos de este concepto hay en el folio ese día |
| `feat_folio_dia_movimientos_log` | `log1p(n)` — total de movimientos del folio ese día |
| `feat_folio_total_movimientos_log` | `log1p(n)` — total histórico de movimientos del folio |
| `feat_t_codigo_freq` | Frecuencia relativa del código de concepto en el corpus |
| `feat_t_usuario_freq` | Frecuencia relativa del cajero en el corpus |

### Estado de la cuenta
| Feature | Qué indica |
|---------|-----------|
| `feat_es_abono` | El movimiento es un crédito/abono |
| `feat_cargo_cancelado` | Cargo explícitamente cancelado en el PMS |
| `feat_cancelacion_sin_marca` | Campo `t_tra_cancelada` es NULL (estado normal para no canceladas) |
| `feat_es_renta` | Cargo de tipo renta/habitación |
| `feat_tiene_reservacion` | El folio tiene reservación enlazada |
| `feat_es_split` | Cargo generado como división de cuenta |
| `feat_usuario_modificado` | La transacción fue modificada después del registro original |
| `feat_usuario_mod_distinto` | La modificó un cajero diferente al que la registró |

### Observaciones textuales (keywords)
| Feature | Palabra clave detectada en el campo de observaciones |
|---------|------------------------------------------------------|
| `feat_obs_kw_error` | "error" |
| `feat_obs_kw_ajuste` | "ajuste" |
| `feat_obs_kw_cancelacion` | "cancelacion" |
| `feat_obs_kw_cortesia` | "cortesia" / "compensacion" |
| `feat_obs_kw_reembolso` | "reembolso" / "devolucion" |
| `feat_obs_missing` | El campo de observaciones está vacío |
| `feat_obs_len_log` | `log1p(len)` — longitud del texto de observaciones |

---

## 4. Algoritmo: Isolation Forest

### Intuición

Isolation Forest **aisla** observaciones construyendo árboles de decisión aleatorios. En cada árbol, el algoritmo:

1. Elige una feature al azar
2. Elige un valor de corte al azar entre el mínimo y máximo de esa feature
3. Divide los datos en dos grupos
4. Repite recursivamente hasta aislar cada punto

Una transacción **normal** está rodeada de otros puntos similares: requiere muchas divisiones para quedar sola. Una transacción **anómala** está en una región esparcida del espacio de features: se aísla en muy pocas divisiones.

### Fórmula del score

El puntaje de anomalía (`score_samples`) se calcula como:

```
score = −2^(−(longitud_promedio_de_path) / c(n))
```

donde `c(n)` es la longitud esperada de path para `n` muestras. Los valores más negativos indican mayor anomalía.

### Por qué se eligió este algoritmo

| Criterio | Isolation Forest | Alternativas descartadas |
|----------|-----------------|--------------------------|
| Velocidad | O(n log n) — segundos para 1.1M filas | LOF: O(n²), OCSVM: no escala >100k |
| Distribución | No asume ninguna | GMM asume gaussiana |
| SHAP | Compatible con TreeExplainer (exacto) | Redes neuronales: solo SHAP aproximado |
| Interpretabilidad | Score → SHAP por feature | LOF: sin equivalente |

### Hiperparámetros

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `n_estimators` | 200 | El doble del default (100); scores más estables |
| `max_samples` | `"auto"` → 256 | Default sklearn; captura estructura local |
| `contamination` | `"auto"` | El umbral se determina post-hoc desde la distribución |
| `n_jobs` | −1 | Usa todos los núcleos disponibles |
| `random_state` | 42 | Reproducibilidad |

---

## 5. Pipeline de entrenamiento (`train.py`)

### Paso 1: Muestra estratificada

Entrenar en las 1.1M filas completas es posible pero innecesario — Isolation Forest converge con muestras representativas. Se toman **200,000 filas** con muestreo proporcional por estrato.

**Estratificación:** combinación de 4 flags binarios:
- `feat_es_abono`
- `feat_cargo_cancelado`
- `feat_dup_mismo_dia_flag`
- `feat_es_renta`

Esto crea hasta 16 combinaciones (15 existen en los datos). Cada estrato aporta filas en proporción a su peso en el corpus, garantizando que tipos de transacción raros (ej. cargos cancelados con duplicado) estén representados en el entrenamiento.

### Paso 2: Entrenamiento

```python
model = IsolationForest(n_estimators=200, max_samples="auto",
                        contamination="auto", n_jobs=-1, random_state=42)
model.fit(X_train.values)  # 200,000 × 63 features
```

### Paso 3: Puntuación del corpus completo

El modelo entrenado puntúa las 1,145,526 filas:

```python
scores = model.score_samples(X.values)  # 1.1M scores
```

Rango observado en este corpus: `[−0.6778, −0.3939]`

### Paso 4: Selección de umbral

El umbral separa transacciones normales de anómalas. Se busca **el mayor salto natural** en la distribución de scores, restringido al rango 0.5%–5% del corpus:

```
Algoritmo:
1. Ordenar scores de menor a mayor
2. Calcular las diferencias consecutivas (np.diff)
3. Buscar el mayor salto dentro de la ventana [0.5%, 5%]
4. Aceptar el salto solo si es >2.5× la mediana de todos los saltos positivos
5. Si no hay salto significativo → usar el percentil 2 como fallback
```

**Resultado en este corpus:**
- Umbral encontrado: **−0.6019** (salto natural en la cola)
- Anomalías flaggeadas: **6,126** (0.53% del corpus)

Cada transacción recibe:
- `anomaly_score`: el score crudo de `score_samples`
- `anomaly_flag`: `True` si `score ≤ threshold`
- `anomaly_rank`: ranking global (rank 1 = más anómala de las 1.1M)
- `anomaly_score_percentile`: percentil en la distribución completa

### Paso 5: SHAP (valores de Shapley)

SHAP cuantifica **cuánto contribuye cada feature** al score de anomalía de cada transacción.

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_anomalias.values, check_additivity=False)
```

Se usa `TreeExplainer` porque Isolation Forest es un conjunto de árboles — los valores SHAP son **exactos** (no aproximados). Solo se calculan para las 6,126 transacciones flaggeadas, no para el corpus completo.

**Interpretación:**
- SHAP negativo → la feature empuja el score hacia abajo (más anómalo)
- SHAP positivo → la feature empuja el score hacia arriba (más normal)
- `|SHAP|` alto → la feature tiene gran influencia en ese score específico

El valor base (`base_value = 11.6429`) es la longitud de path promedio en espacio sin normalizar — no es directamente el score, sino el punto de referencia desde el cual los SHAP suman para llegar al score individual.

---

## 6. Pipeline de explicabilidad (`explain.py`)

`explain.py` toma `shap_anomalies.parquet` y produce texto en español contable para el auditor. No es un LLM — es un sistema de reglas determinístico basado en valores SHAP.

### Arquitectura general

```
Para cada anomalía:
  1. Evaluar 19 generadores de razón
  2. Filtrar por SHAP_MIN = 0.02 (solo features con impacto significativo)
  3. Deduplicar por categoría (máximo una razón por categoría)
  4. Ordenar por |SHAP| descendente, tomar las 5 más importantes
  5. Asignar severidad por ranking global
  6. Componer texto concatenando las 3 principales razones
```

### Carga de datos

`explain.py` hace un join entre:
- `shap_anomalies.parquet` (6,126 anomalías con columnas `trace_*` y `shap_*`)
- `transacciones_modelado.parquet` → columnas `feat_*` necesarias para la explicación, usando `trace_row_id` como índice posicional

También carga estadísticas por código de concepto (mediana e IQR del monto) para decir "3x el habitual".

### Los 19 generadores de razón

Cada generador es una función que recibe la fila y retorna una tupla `(categoría, texto, |SHAP|)` o `None`.

El criterio de activación siempre incluye dos condiciones:
1. El **valor de la feature** soporta la interpretación (ej. flag = 1, ratio > umbral)
2. El **|SHAP| de esa feature** supera `SHAP_MIN = 0.02` — la feature tuvo impacto real en el score

| Generador | Categoría generada | Features evaluadas | Condición de activación |
|-----------|-------------------|--------------------|------------------------|
| `_r_dup_minuto` | Posible duplicado | `feat_dup_mismo_minuto_flag` | flag=1 y SHAP≥0.02 |
| `_r_dup_dia` | Posible duplicado | `feat_dup_mismo_dia_flag`, `feat_dup_mismo_dia_log` | flag=1 y SHAP≥0.02; texto incluye el conteo reconstruido |
| `_r_monto_atipico` | Monto atípico | `feat_monto_z_codigo_carabo`, `feat_monto_abs_log` | \|z\|≥2.0 y SHAP≥0.02; compara contra mediana del concepto |
| `_r_vs_tarifa` | Cargo vs tarifa | `feat_monto_vs_tarifa_ratio`, `feat_monto_abs_log` | ratio≥2.0 y SHAP≥0.02 |
| `_r_noches_delta` | Inconsistencia de reservación | `feat_noches_delta_scaled` | \|valor\|≥1.5 y SHAP≥0.02 |
| `_r_fuera_estancia` | Fuera de estancia | `feat_cargo_fuera_estancia`, `feat_cargo_antes_llegada` | flag=1 y SHAP≥0.02; diferencia antes/después de llegada |
| `_r_horario` | Horario inusual | `feat_es_madrugada`, `feat_hora_sin`, `feat_hora_cos` | madrugada con SHAP≥0.02, o hora cíclica con SHAP≥0.08 y hora<6 o ≥22 |
| `_r_signo` | Inconsistencia de signo | `feat_monto_negativo_sin_abono`, `feat_monto_positivo_en_abono` | flag=1 y SHAP≥0.02 |
| `_r_modificacion` | Modificación sospechosa | `feat_usuario_mod_distinto` | flag=1 y SHAP≥0.02 |
| `_r_cancelacion` | Cancelación irregular | `feat_cargo_cancelado` | flag=1 y SHAP≥0.02 (no activa con `feat_cancelacion_sin_marca` — ese campo NULL es el estado normal) |
| `_r_densidad_concepto` | Alta densidad de cargos | `feat_folio_codigo_dia_count_log` | conteo≥3 y SHAP≥0.02 |
| `_r_densidad_folio` | Alta densidad de cargos | `feat_folio_dia_movimientos_log` | movimientos≥20 y SHAP≥0.02 |
| `_r_impuesto` | Proporción fiscal atípica | `feat_impuesto_ratio_abs` | monto>$100, \|ratio−0.16\|>0.05 y SHAP≥0.02 |
| `_r_propina` | Propina atípica | `feat_propina_ratio_abs` | ratio≥15% y SHAP≥0.02 |
| `_r_obs_keywords` | Observación relevante | `feat_obs_kw_*` (5 keywords) | cualquier keyword presente y SHAP≥0.02 |
| `_r_codigo_infrecuente` | Código infrecuente | `feat_t_codigo_freq` | frecuencia<0.1% y SHAP≥0.02 |
| `_r_usuario_infrecuente` | Usuario con baja actividad | `feat_t_usuario_freq` | frecuencia<0.2% y SHAP≥0.02 |
| `_r_es_split` | División de cuenta | `feat_es_split` | flag=1 y SHAP≥0.02 |
| `_r_sin_reservacion` | Sin reservación enlazada | `feat_tiene_reservacion`, `feat_es_renta` | cargo de renta sin reservación enlazada y SHAP≥0.02 |

### Deduplicación y selección

```python
# Deduplicar por categoría: si dos generadores emiten la misma categoría,
# conservar solo la razón con mayor |SHAP|
best: dict[str, Reason] = {}
for cat, text, shap_abs in raw:
    if cat not in best or shap_abs > best[cat][2]:
        best[cat] = (cat, text, shap_abs)

# Ordenar por impacto y tomar las 5 más importantes
return sorted(best.values(), key=lambda r: r[2], reverse=True)[:5]
```

Esto evita que una transacción con dos indicadores de "Posible duplicado" repita la misma categoría dos veces.

### Asignación de severidad

La severidad es **puramente por ranking global** — cuánto de anómala es la transacción respecto a todo el corpus de 1.1M:

| Severidad | Criterio | Cantidad | Acción sugerida |
|-----------|----------|----------|-----------------|
| Alta | `anomaly_rank ≤ 1,200` | 1,200 (19.6%) | Acción urgente antes del cierre |
| Media | `1,200 < rank ≤ 4,000` | 2,800 (45.7%) | Revisar durante el turno |
| Informativa | `rank > 4,000` | 2,126 (34.7%) | Documentar; baja prioridad |

El ranking ya incorpora toda la información de las 63 features — no se necesitan reglas adicionales para categorías específicas.

### Composición del texto

Las 3 razones de mayor peso SHAP se concatenan con conectores naturales:

```python
# 1 razón:
"Las noches del cargo no coinciden con las noches de la reservación."

# 2 razones:
"Las noches del cargo no coinciden con las noches de la reservación.
 Además, el cargo ($25,560) equivale a 7.0x la tarifa diaria pactada."

# 3+ razones:
"Las noches del cargo no coinciden con las noches de la reservación.
 Además, el cargo ($25,560) equivale a 7.0x la tarifa diaria pactada.
 También se detectó: la transacción fue modificada por un usuario distinto al cajero original."
```

Si ningún generador activa (0 razones), se produce el texto genérico:
> "Transacción con patrón estadístico inusual detectado por el sistema. Se recomienda revisión manual."

---

## 7. Flujo completo de datos

```
transacciones_modelado.parquet (1.1M × ~80 cols)
        │
        ▼ data_modeling (read-only)
X_modelo.parquet (1.1M × 63 feat_*)
        │
        ▼ train.py — muestra estratificada (200k)
IsolationForest.fit()
        │
        ▼ train.py — score_samples() sobre 1.1M
scores[] + umbral (−0.6019)
        │
        ├──▶ scored_transactions.parquet (1.1M filas)
        │         anomaly_score, anomaly_rank, anomaly_flag
        │
        ▼ train.py — SHAP TreeExplainer (solo 6,126 flaggeadas)
shap_anomalies.parquet (6,126 × trace_* + shap_feat_*)
        │
        ▼ explain.py — 19 generadores de razón
anomalies_explained.parquet (6,126 × todo + severity/categories/explanation)
anomalies_explained_preview.csv (columnas esenciales para revisión rápida)
```

---

## 8. Artefactos generados

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `artifacts/isolation_forest.joblib` | — | Modelo serializado; cargable con `joblib.load` |
| `artifacts/scored_transactions.parquet` | 1,145,526 | Corpus completo con `anomaly_score`, `anomaly_rank`, `anomaly_flag` |
| `artifacts/shap_anomalies.parquet` | 6,126 | Anomalías con columnas `shap_feat_*` y metadatos de traza |
| `artifacts/anomalies_explained.parquet` | 6,126 | Anomalías con `severity`, `categories`, `explanation`, `reasons_json` + feat_* y shap_* |
| `artifacts/anomalies_explained_preview.csv` | 6,126 | Columnas clave ordenadas por `anomaly_rank` para revisión rápida |
| `output/score_distribution.png` | — | Histograma completo y zoom cola con umbral marcado |
| `output/shap_bar.png` | — | Top 20 features por mean \|SHAP\| en anomalías |
| `output/shap_summary.png` | — | Beeswarm SHAP: distribución de contribuciones (muestra 2,000 pts) |
| `output/anomaly_by_hour.png` | — | Tasa de anomalía por hora del día |
| `output/anomaly_by_codigo.png` | — | Top 20 códigos por tasa de anomalía |

---

## 9. Cómo usar los artefactos

### Ver las anomalías más urgentes

```python
import pandas as pd

df = pd.read_parquet("model_Rogelio/artifacts/anomalies_explained.parquet")
top = df.sort_values("anomaly_rank").head(20)
print(top[["trace_t_folio", "trace_t_codigo", "trace_t_timestamp",
           "severity", "explanation"]].to_string())
```

### Puntuar transacciones nuevas

```python
import joblib
import pandas as pd

model = joblib.load("model_Rogelio/artifacts/isolation_forest.joblib")

# X_nueva debe tener exactamente las mismas 63 columnas feat_* en el mismo orden
X_nueva = pd.read_parquet("nueva_transaccion.parquet")
scores = model.score_samples(X_nueva.values)
# scores < −0.6019 → anómala
```

### Ver por qué una transacción específica es anómala

```python
df = pd.read_parquet("model_Rogelio/artifacts/anomalies_explained.parquet")
tx = df[df["trace_t_transaccion"] == "ID_DE_TRANSACCION"].iloc[0]

print("Severidad:", tx["severity"])
print("Categorías:", tx["categories"])
print("Explicación:", tx["explanation"])

# Detalle de todas las razones con su peso SHAP
import json
for r in json.loads(tx["reasons_json"]):
    print(f"  [{r['peso_shap']:.4f}] {r['categoria']}: {r['texto']}")
```

---

## 10. Limitaciones y consideraciones

| Limitación | Detalle |
|-----------|---------|
| Sin supervisión | El modelo no sabe si una anomalía es realmente un error — decide por rareza estadística |
| Umbral fijo | El umbral (−0.6019) se calculó sobre el corpus histórico; puede necesitar recalibración si el volumen de transacciones cambia significativamente |
| SHAP mínimo (0.02) | Features con impacto menor a 0.02 no generan razón, aunque en conjunto contribuyan al score |
| `feat_dup_mismo_minuto_flag` en el 34.5% | El PMS postea lotes de transacciones con el mismo timestamp; este flag no indica duplicado manual en todos los casos — verificar con contexto del folio |
| Datos históricos únicamente | El modelo fue entrenado sobre datos hasta la fecha de producción; eventos completamente nuevos pueden no detectarse correctamente |
| Modelo mejorable con feedback | Si el equipo de auditoría valida anomalías (verdaderos positivos / falsos positivos), se puede entrenar un clasificador supervisado sobre los SHAP values |
