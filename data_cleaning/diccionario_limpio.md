# Diccionario — Base LIMPIA FINANOM

Generado por `data_cleaning/clean.py` a partir de la base consolidada. Limpieza en dos pases (sin encoding/imputación/escalado, que son fase de modelado).

## 1. Resumen

- **Archivo**: `data_cleaning/output/transacciones_limpio.parquet`
- **Filas**: 1,145,526 (todas; no se eliminó ninguna)
- **Columnas**: 34 (antes 57; se eliminaron 23)

## 2. Columnas eliminadas y por qué

### Pase 1 — estructural (10): redundantes / constantes / colineales

| Columna | Razón |
| ------- | ----- |
| `t_centro_consumo` | Redundante con t_codigo (centro '01' ⟺ PUNTAZ); 24.4% nulos; 99.97% '00'. |
| `h_tpo_mon` | Constante (NAL=MXN 99.99%); sin varianza. |
| `h_lim_cre` | 99.74% en 0 + centinelas (999999999); sin señal. |
| `h_tfa_impuestos` | Mal nombrada; == h_tfa_total en 97.3% de filas (duplicada). |
| `h_tfa_renta` | Colineal con h_tfa_total (r=0.992). |
| `t_impuesto2` | Colineal con t_impuesto (r=0.92) y 89.7% en 0. |
| `t_num_per` | Duplica h_num_per (r=0.997); h_num_per cubre 49,474 filas más. |
| `t_num_adu` | Duplica h_num_adu (r=0.991); 69.6% nulos vs 21.5%. |
| `t_tra_hra` | Componente de t_timestamp (coincide 100%). |
| `t_tra_mto` | Componente de t_timestamp (coincide 100%). |

### Pase 2 — bajo valor para detección de anomalías (13)

| Columna | Razón |
| ------- | ----- |
| `t_fecha` | Redundante con t_timestamp (normalize == t_fecha al 100%). |
| `t_inc_tfa` | Casi-constante (97.5% 'S') + 25.8% nulos; solapa con es_renta. |
| `h_status` | Sin señal: solo 50=salida (99.98%) y 10=en casa; sin cancelación/no-show. |
| `h_tfa_extras` | Casi-constante (97.4% en 0); el extra real ya está en t_monto. |
| `h_num_adu` | Redundante: h_num_per = h_num_adu + h_num_men en 99.9% de filas. |
| `h_num_men` | Redundante: h_num_per = h_num_adu + h_num_men en 99.9% de filas. |
| `h_res_usr` | Usuario que creó la reserva; débil para anomalía a nivel cargo. |
| `h_rec_usr` | Usuario de check-in; débil para anomalía a nivel cargo. |
| `h_fec_reg` | Fecha de registro de la reserva; baja relevancia al cargo. |
| `t_can_dia` | 85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada. |
| `t_can_mes` | 85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada. |
| `t_can_hra` | 85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada. |
| `t_can_mto` | 85.4% nulo, sin año; la cancelación ya la marca t_tra_cancelada. |

## 3. Columnas conservadas (por rol)

### id — Identificador (agrupar/derivar/trazar; NO feature crudo)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_folio` | string | 0.0 | 145,493 |
| `t_folio_ext` | string | 0.0 | 42 |
| `t_referencia` | string | 0.0 | 90,788 |
| `t_transaccion` | string | 0.0 | 3,396 |
| `t_cve_res` | string | 21.4 | 122,299 |
| `t_cuarto` | string | 0.0 | 64,534 |

### cat — Categórica (feature tras encoding por frecuencia)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_codigo` | string | 0.0 | 108 |
| `t_carabo` | string | 0.0 | 2 |
| `t_usuario` | string | 0.0 | 149 |
| `t_usuario_mod` | string | 80.4 | 109 |
| `h_tpo_hab` | string | 21.5 | 12 |
| `h_tpo_hsp` | string | 21.5 | 6 |
| `h_seg_mer` | string | 21.5 | 25 |
| `h_cod_age` | string | 21.5 | 165 |
| `h_tpo_plan` | string | 21.5 | 38 |
| `h_for_pgo` | string | 21.5 | 11 |

### num — Numérica (feature; escalar, idealmente por t_codigo)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_monto` | float64 | 0.0 | 108,377 |
| `t_impuesto` | float64 | 0.0 | 57,598 |
| `t_propina` | float64 | 0.0 | 228 |
| `t_noches` | int64 | 0.0 | 53 |
| `h_num_per` | float64 | 21.5 | 16 |
| `h_num_noc` | float64 | 21.5 | 34 |
| `h_tfa` | float64 | 21.5 | 35,128 |
| `h_tfa_total` | float64 | 21.5 | 39,872 |
| `h_tarifa_forzada` | float64 | 21.5 | 31,135 |
| `h_dep_sol` | float64 | 21.5 | 18,888 |

### flag — Bandera / weak label (regla; señal débil para IF)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_tra_cancelada` | string | 24.5 | 2 |
| `es_split` | bool | 0.0 | 2 |
| `es_renta` | bool | 0.0 | 2 |
| `tiene_reservacion` | bool | 0.0 | 2 |

### datetime — Temporal (base de features; NO cruda)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_timestamp` | datetime64[us] | 0.0 | 171,575 |
| `h_fec_lld` | datetime64[us] | 21.5 | 1,837 |
| `h_fec_sda` | datetime64[us] | 21.5 | 1,839 |

### text — Texto libre (keywords / reason; NO feature numérico)

| Columna | Tipo | % nulos | Únicos |
| ------- | ---- | ------- | ------ |
| `t_observaciones` | string | 1.4 | 404,106 |

## 4. Siguientes pasos (fase de modelado, NO limpieza)

1. **Imputar nulos** (sklearn IsolationForest no acepta NaN): centinela + flag de faltante; `tiene_reservacion` ya marca el bloque `h_`.
2. **Encoding por frecuencia** de las categóricas (no enmascara códigos raros, a diferencia de un cubo 'otros').
3. **Escalar `t_monto` por `t_codigo`** (z-score robusto por concepto): el paso de mayor impacto para que el IF encuentre inconsistencias y no solo 'números grandes'.
4. **Feature `n_duplicados`**: conteo de cargos idénticos por (folio, código, monto, día) — el 'cargo duplicado' del user story.
