# Diccionario — Base consolidada FINANOM

Base única para la detección no supervisada de anomalías/inconsistencias financieras. Generada por `data_consolidation/consolidate.py` a partir de `tablas_parquet/`.

## 1. Resumen

- **Archivo**: `data_consolidation/output/transacciones_consolidado.parquet`
- **Filas**: 1,145,526 (una por transacción de `hottra`)
- **Columnas**: 57
- **Grano**: una transacción = una predicción del modelo
- **Rango temporal**: 2021-02-19 → 2026-03-11
- **Cobertura de reservación** (LEFT JOIN con `hothsp`): 78.5%
- **Tablas fuente**: `hottra` (transacciones) + `hothsp` (contexto de reservación)

## 2. Cómo se construyó

1. Se carga `hottra` y se conservan solo columnas relevantes para anomalías financieras (montos, impuestos, propina, control/cancelación, autoría, ocupación, código de concepto, timestamp).
2. Se limpia: `strip` de strings, blancos → `<NA>`, fechas `YYYYMMDD` → datetime (`00000000` → NaT), ocupación string → entero nullable, y se arma `t_timestamp` (fecha+hora+minuto).
3. Se une por LEFT JOIN `t_cve_res == h_res_cve` con un subconjunto de `hothsp` (estado, tarifas, ocupación esperada, agencia/segmento, fechas de estancia, usuarios). Se conservan TODAS las transacciones; las que no tienen reservación quedan con contexto nulo y `tiene_reservacion = False`.

## 3. Columnas

Prefijo `t_` = transacción, `h_`/`Num_` = reservación, sin prefijo = derivada.

| Columna | Origen | Tipo | % nulos | Únicos | Ejemplos | Descripción / utilidad |
| --- | --- | --- | ---: | ---: | --- | --- |
| `t_folio` | hottra | string | 0.0 | 145,493 | 7518, 7338, 7548 | Folio de la cuenta del huésped (cuenta donde se agrupan los cargos). Trazabilidad y agrupación de cargos por cuenta. |
| `t_folio_ext` | hottra | string | 0.0 | 42 | 0, 1, 15 | Extensión del folio (sub-cuenta). Junto con t_folio identifica la cuenta exacta. |
| `t_referencia` | hottra | string | 0.0 | 90,788 | A1320A, 1320H2, 1320A2 | Referencia cruzada del movimiento. Ayuda a emparejar cargos con sus reversos/abonos. |
| `t_transaccion` | hottra | string | 0.0 | 3,396 | 213, 214, 215 | Número correlativo de la transacción. Identificador secuencial del movimiento. |
| `t_codigo` | hottra | string | 0.0 | 108 | PROPTI, RENHAB, MENEXT | Código del concepto del cargo (RENHAB=renta, PROPTI=propina, DEPOS=depósito, AJ*=ajustes, DEV*=devoluciones, etc.). Eje para z-score de monto por concepto. SIN catálogo oficial entregado. |
| `t_cve_res` | hottra | string | 21.4 | 122,299 | I   41791     1, G     199    17, I   41177     1 | Llave de la reservación a la que pertenece el cargo (formato tpo+num+mbo). Llave de join con la reservación (hothsp). Vacío = walk-in / depósito / cargo sin reservación. |
| `t_cuarto` | hottra | string | 0.0 | 64,534 | 13207, 13208, 13209 | Número de habitación donde se generó el cargo. |
| `t_centro_consumo` | hottra | string | 24.4 | 2 | 00, 01 | Centro de consumo (00=recepción domina; 01 marginal). Casi siempre recepción: señal débil en este hotel. |
| `t_fecha` | hottra | datetime64[us] | 0.0 | 1,839 | 2021-06-28 00:00:00, 2021-06-30 00:00:00, 2021-06-29 00:00:00 | Fecha del cargo (parseada). Eje temporal de la transacción. |
| `t_tra_hra` | hottra | string | 0.0 | 24 | 05, 15, 01 | Hora del cargo (HH, 00–23). Cargos en madrugada fuera del night audit son sospechosos. |
| `t_tra_mto` | hottra | string | 0.0 | 60 | 31, 32, 18 | Minuto del cargo (MM, 00–59). Componente del timestamp; útil para detectar duplicados en segundos/minutos. |
| `t_monto` | hottra | float64 | 0.0 | 108,377 | 72.0, 1478.0, 48.0 | Monto del movimiento (MXN). Eje del modelo. ~5% negativos (reversos/notas de crédito legítimos): el signo se modela como feature, NO se filtra. |
| `t_impuesto` | hottra | float64 | 0.0 | 57,598 | 0.0, 216.91, 306.99 | Impuesto principal del cargo. Un ratio impuesto/monto fuera de ~16% es señal de anomalía. |
| `t_impuesto2` | hottra | float64 | 0.0 | 17,880 | 0.0, 74.12, -63.03 | Impuesto secundario. Generalmente 0; valores grandes ameritan revisión. |
| `t_propina` | hottra | float64 | 0.0 | 228 | 72.0, 0.0, 48.0 | Propina del cargo. Propina anormal respecto al monto es una regla candidata. |
| `t_carabo` | hottra | string | 0.0 | 2 | 0, 1 | Naturaleza del movimiento: 0=CARGO, 1=ABONO. Define el signo contable; clave para conciliar cargos vs abonos. |
| `t_tra_cancelada` | hottra | string | 24.5 | 2 | 1, 0 | Estado de cancelación: 0=activa, 1=cancelada, <NA>=no aplica/sin marca. Weak label: las canceladas son candidatas a anomalía. |
| `t_can_dia` | hottra | string | 85.4 | 31 | 01, 02, 03 | Día de la cancelación (DD, sin año). Componente del momento de cancelación. |
| `t_can_mes` | hottra | string | 85.4 | 12 | 07, 06, 09 | Mes de la cancelación (MM, sin año). Componente del momento de cancelación. |
| `t_can_hra` | hottra | string | 85.4 | 24 | 11, 05, 06 | Hora de la cancelación (HH). Cancelaciones en horario atípico son señal. |
| `t_can_mto` | hottra | string | 85.4 | 60 | 53, 11, 15 | Minuto de la cancelación (MM). |
| `t_usuario` | hottra | string | 0.0 | 149 | MCC, IVL, ALM | Usuario que generó el cargo (clave de 3 letras). Detección de patrones por operador (fraude/error sistemático). |
| `t_usuario_mod` | hottra | string | 80.4 | 109 | ALM, RFR, IVL | Usuario que modificó el cargo. Modificaciones por usuarios distintos al que creó pueden ser señal. |
| `t_num_adu` | hottra | Int64 | 69.6 | 8 | 3.0, 2.0, 1.0 | Adultos asociados al cargo. Inconsistencia vs ocupación de la reservación es señal. |
| `t_num_per` | hottra | Int64 | 25.8 | 9 | 3.0, 2.0, 4.0 | Personas totales asociadas al cargo. |
| `t_noches` | hottra | int64 | 0.0 | 53 | 3.0, 2.0, 4.0 | Noches asociadas al cargo (0–134). Estancias muy largas son outliers. |
| `t_inc_tfa` | hottra | string | 25.8 | 2 | S, N | Concepto incluido en tarifa: S / N / <NA>. Distingue cargos que forman parte de la tarifa. |
| `t_observaciones` | hottra | string | 1.4 | 404,106 | RENTA:  HABITACIONI   , CARDENAS, AMERICA G   , RENTA:  HABITACIONI    | Texto libre del cargo (p.ej. 'RENTA: HABITACION...'). Útil para reasons del auditor y detección de keywords (ajuste, cortesía, error). Puede contener nombres: tratar como sensible. |
| `t_timestamp` | derivada | datetime64[us] | 0.0 | 171,575 | 2021-06-28 05:31:00, 2021-06-30 15:32:00, 2021-06-29 01:18:00 | DERIVADA: fecha+hora+minuto del cargo en un solo datetime. Base para features temporales y detección de duplicados por ventana. |
| `es_split` | derivada | bool | 0.0 | 2 | False, True | DERIVADA: True si el cargo es partido (split). Colapsa las 3 columnas de split originales (77 cargos). Los splits pueden generar doble conteo. |
| `es_renta` | derivada | bool | 0.0 | 2 | True, False | DERIVADA: True si es cargo de renta/hospedaje (flag original t_renta='S', night audit). Distingue renta de extras/ajustes. |
| `h_status` | hothsp | string | 21.5 | 2 | 50, 10 | Estado de la reservación (00=registro, 01=cancelada, 02=no show, 10=en casa, 50=salida, ...). Contexto del ciclo de vida; cargos sobre reservas canceladas/no-show son sospechosos. |
| `h_tpo_hab` | hothsp | string | 21.5 | 12 | C2D, C1K, C6D | Tipo de habitación de la reservación. Baseline para comparar tarifas por categoría. |
| `h_tpo_hsp` | hothsp | string | 21.5 | 6 | NOR, VP1, VP3 | Tipo de huésped (NOR, VIP, VP2...). Las tarifas VIP/cortesía explican algunos outliers. |
| `h_seg_mer` | hothsp | string | 21.5 | 25 | ONL255, GPO410, MAY330 | Segmento de mercado (ONL, MAY, GPO...). Baseline tarifario por segmento. |
| `h_cod_age` | hothsp | string | 21.5 | 165 | 1BKINGMX, 1RESNAL, 1IMAGDL | Agencia / cuenta que originó la reservación. Anomalías por agencia (tarifas 0, descuadres). |
| `h_tpo_plan` | hothsp | string | 21.5 | 38 | IHOS, ROOMR2, MAYN | Plan tarifario contratado (B2C, MAYN, GSHO...). |
| `h_for_pgo` | hothsp | string | 21.5 | 11 | TARCRE, EFE, TRANSF | Forma de pago de la reservación (EFE, TARCRE, XFAC...). |
| `h_tpo_mon` | hothsp | string | 21.5 | 2 | NAL, DLS | Moneda de la reservación (NAL=MXN domina, DLS marginal). Confirma que casi todo es MXN. |
| `h_num_per` | hothsp | float64 | 21.5 | 16 | 3.0, 2.0, 4.0 | Personas esperadas en la reservación. Comparar vs ocupación del cargo. |
| `h_num_adu` | hothsp | float64 | 21.5 | 12 | 3.0, 2.0, 1.0 | Adultos esperados en la reservación. |
| `h_num_men` | hothsp | float64 | 21.5 | 11 | 0.0, 2.0, 1.0 | Menores esperados en la reservación. |
| `h_num_noc` | hothsp | float64 | 21.5 | 34 | 3.0, 2.0, 4.0 | Noches de la reservación (máx 527). Estancias extremas son outliers. |
| `h_tfa` | hothsp | float64 | 21.5 | 35,128 | 3007.61, 1526.0, 2098.51 | Tarifa diaria de la reservación. Comparar contra cargos de renta. |
| `h_tfa_total` | hothsp | float64 | 21.5 | 39,872 | 9022.82, 3052.0, 4197.02 | Tarifa total esperada de la reservación. Descuadre vs suma de cargos (t_monto) del folio = regla de anomalía. |
| `h_tfa_renta` | hothsp | float64 | 21.5 | 52,128 | 7514.36, 2522.18, 3458.2 | Componente de renta de la tarifa. |
| `h_tfa_impuestos` | hothsp | float64 | 21.5 | 40,547 | 9022.82, 3052.0, 4197.02 | Componente de impuestos de la tarifa (semántica a confirmar; mean alto). |
| `h_tfa_extras` | hothsp | float64 | 21.5 | 808 | 0.0, 605.0, 9648.0 | Componente de extras de la tarifa. |
| `h_tarifa_forzada` | hothsp | float64 | 21.5 | 31,135 | 3007.61, 1526.0, 2098.51 | Tarifa forzada manualmente (override). Forzar tarifa fuera de rango es señal de autorización indebida. |
| `h_dep_sol` | hothsp | float64 | 21.5 | 18,888 | 0.0, 4293.0, 3501.0 | Depósito solicitado en la reservación. Conciliación contra cargos de depósito (DEP*). |
| `h_lim_cre` | hothsp | float64 | 21.5 | 5 | 0.0, 999999.0, 99999999.0 | Límite de crédito (contiene valores centinela 999999999). Tratar centinelas como faltante. |
| `h_fec_lld` | hothsp | datetime64[us] | 21.5 | 1,837 | 2021-06-28 00:00:00, 2021-06-27 00:00:00, 2021-06-26 00:00:00 | Fecha de llegada (check-in) de la reservación. |
| `h_fec_sda` | hothsp | datetime64[us] | 21.5 | 1,839 | 2021-07-01 00:00:00, 2021-06-29 00:00:00, 2021-06-30 00:00:00 | Fecha de salida (check-out) de la reservación. Cargos posteriores a la salida son sospechosos. |
| `h_fec_reg` | hothsp | datetime64[us] | 22.3 | 1,915 | 2021-06-28 00:00:00, 2021-05-07 00:00:00, 2021-06-27 00:00:00 | Fecha de registro de la reservación. |
| `h_res_usr` | hothsp | string | 22.6 | 118 | CRS, POS, DV1 | Usuario que creó la reservación. |
| `h_rec_usr` | hothsp | string | 21.5 | 102 | IEP, FJL, LBB | Usuario que registró (check-in) la reservación. |
| `tiene_reservacion` | derivada | bool | 0.0 | 2 | True, False | DERIVADA: True si el cargo cruzó con una reservación en hothsp. Separa cargos con/sin contexto de reservación. |

## 4. Qué se descartó y por qué

- **Columnas muertas de `hottra`** (un solo valor): `ibuff`, `t_ya_facturada`, `t_tipo_trans`, `t_folio_origen`, `t_tra_origen`, `t_ref_origen`, `t_autorizacion`, `t_claveorigen`, `t_tipo_cambio`. Las cuatro de trazabilidad de origen y `t_ya_facturada` fueron confirmadas como NO aplicables a México / vacías en estos datos.
- **Campos de facturación de `hothsp`** (100% vacíos): `h_ya_fact`, `h_num_fac`, `h_fec_fac`, `h_status_pago`. Confirmado: facturación adelantada y WebCheckIn no aplican / no se exportaron.
- **PII** (`h_nom`, `h_nombre`, `h_apellido_*`, `h_tar_cre`, correos, etc.): no aportan a la detección financiera y son sensibles.
- **Columnas casi vacías o constantes al grano de transacción**: `t_numctapago` (99% vacío), `t_num_men` (99.9% vacío, redundante con `t_num_per`), `Num_cancelacion` (las reservas canceladas casi no generan cargos → 0.04% lleno) y `h_tot_hab` (constante = 1 en los cargos enlazados). Las 3 columnas de split se colapsaron en el booleano `es_split`.
- **`hotcag`**: NO es el catálogo de `t_codigo` (solo 43 de 1.14M filas cruzan; 107 de 108 códigos quedan huérfanos). No enriquece.
- **`hotvta`** (ventas agregadas), **`hotcvt`** (tipo de cambio) y catálogos menores: distinto grano o no necesarios; casi todo es MXN.
- **Tablas operativas** (bloqueos, status de cuarto, requerimientos, eventos): no financieras.

## 5. Limitaciones conocidas

- Los 108 `t_codigo` no tienen catálogo oficial; su semántica se infiere por prefijo (REN=renta, DEP=depósito, AJ=ajuste, DEV=devolución, PROP=propina...).
- La cancelación trae día/mes/hora/minuto pero **no año**: no se puede armar un datetime completo de cancelación.
- `h_lim_cre` contiene valores centinela (999999999); tratarlos como faltante.
- ~21.5% de cargos no tienen reservación enlazada (walk-in / depósito / centro de consumo); su contexto `h_*` es nulo por diseño.
- No hay conversión de moneda: se asume MXN (los datos son de un hotel en México).
