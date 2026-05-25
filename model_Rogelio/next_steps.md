# FINANOM — Próximos Pasos: De Prototipo a Herramienta de Auditoría

> Documento de visión de producto para el equipo de desarrollo y stakeholders.

---

## 1. Visión del producto

El objetivo es convertir el modelo de detección de anomalías en una herramienta integrada al PMS que el auditor nocturno —y en general cualquier miembro del equipo contable— pueda usar sin conocimientos técnicos. El sistema debe:

- Mostrar anomalías pasadas y presentes con explicaciones en lenguaje contable
- Emitir alertas en tiempo real cuando aparezca una transacción atípica
- Permitir que el auditor tome una acción sobre cada alerta (autorizar, desestimar, escalar)
- Generar un reporte diario como cierre del turno de auditoría
- Ser accesible desde el mismo PMS, sin instalar nada adicional

---

## 2. Arquitectura propuesta

```
PMS (fuente de datos)
        │
        ▼
┌─────────────────────┐
│  Motor de detección │  ← Isolation Forest + reglas
│  (batch + streaming)│
└──────────┬──────────┘
           │  anomaly_score, shap_*, categoría, explicación
           ▼
┌─────────────────────┐
│   Base de datos     │  ← transacciones + decisiones del auditor
│   de anomalías      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│               Interfaz web (acceso desde PMS)        │
│  Dashboard │ Lista │ Detalle │ Alertas │ Reportes    │
└─────────────────────────────────────────────────────┘
```

**Stack sugerido:**
| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Backend API | FastAPI (ya instalado) | Expone endpoints de anomalías y acciones |
| Orquestación batch | Prefect (ya instalado) | Ejecuta scoring nocturno y reportes |
| Base de datos | PostgreSQL o SQLite (demo) | Guarda anomalías, acciones, historial |
| Frontend | React + Tailwind o Streamlit (demo rápida) | Interfaz ligera embebible en PMS |
| Alertas | WebSocket o email/SMS | Notificaciones en tiempo real |

---

## 3. Módulos de la interfaz

### 3.1 Dashboard de auditoría

La pantalla principal que el auditor ve al entrar:

```
┌────────────────────────────────────────────────────────────────┐
│  FINANOM — Hotel Paraíso · Auditoría 24 may 2026               │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Anomalías   │  Revisadas   │  Escaladas   │  Pendientes       │
│  detectadas  │  hoy         │  a gerencia  │  de revisión      │
│     47       │     31       │      3       │      13           │
└──────────────┴──────────────┴──────────────┴───────────────────┘

⚠ ALERTAS RECIENTES (últimas 2 horas)
  · Cargo duplicado — Folio 4821 — RENHAB $2,400 — hace 8 min  [Ver]
  · Monto 9x el promedio — Folio 3302 — PROPTI $18,000 — hace 23 min  [Ver]

📋 COLA DE REVISIÓN (13 pendientes, ordenadas por prioridad)
  [Ver lista completa]
```

### 3.2 Lista de anomalías

Tabla filtrable por fecha, turno, tipo de anomalía, usuario cajero, estado de revisión.

| # | Folio | Concepto | Monto | Por qué es atípico | Estado |
|---|-------|----------|-------|--------------------|--------|
| 1 | 4821 | RENHAB | $2,400 | Cargo duplicado en el mismo minuto | Pendiente |
| 2 | 3302 | PROPTI | $18,000 | 9 veces el monto usual del concepto | Pendiente |
| 3 | 1107 | DEPOSI | -$500 | Monto negativo sin marca de abono | Revisado |

### 3.3 Vista de detalle de la anomalía

El núcleo de la herramienta. Cada anomalía abre una pantalla con:

**Encabezado:**
```
Transacción #4821-A  ·  24 may 2026, 03:12 AM  ·  Puntuación de riesgo: ████░ Alta
Folio: 4821  ·  Habitación: 214  ·  Cajero: MLOPEZ  ·  Concepto: RENHAB
```

**Por qué el sistema marcó esta transacción:**
```
⚠ Cargo registrado a las 3:12 AM, fuera del horario operativo habitual
    → El 95% de los cargos RENHAB se registran entre 6:00 AM y 11:00 PM

⚠ Existe un cargo idéntico ($2,400 RENHAB) en el mismo folio 3 minutos antes
    → Posible doble posteo por el cajero MLOPEZ

⚠ El cargo fue creado y modificado por usuarios distintos
    → Creado por RGARCIA, modificado por MLOPEZ a las 3:14 AM
```

**Transacción en contexto:**
```
Reservación enlazada: RES-2890
  Huésped: ████████ (nombre enmascarado)    Llegada: 23 may  ·  Salida: 26 may
  Tarifa diaria pactada: $2,100 MXN         Noches: 3

Este cargo ($2,400) representa 1.14× la tarifa diaria de la reservación.
Otros cargos RENHAB en el folio:  $2,100 (22 may), $2,400 (23 may), $2,400 (24 may, 03:09 AM), $2,400 (24 may, 03:12 AM) ← este
```

**Acciones:**
```
[ ✓ Autorizar — el cargo es correcto ]
[ ✗ Desestimar — no requiere corrección ]  
[ ↑ Escalar a gerencia — requiere revisión superior ]
[ 📝 Agregar nota antes de decidir... ]
```

### 3.4 Alertas en tiempo real

Durante el turno, el sistema monitorea cada nueva transacción postada al PMS. Si su score supera el umbral:

- Aparece un **banner** en la interfaz mientras el auditor trabaja
- Se envía un **correo o SMS** al auditor en turno
- Se registra en el **log de alertas** con timestamp

Configurables por tipo:
- Duplicados → alerta inmediata (alta urgencia)
- Monto atípico → alerta dentro de 5 minutos (media urgencia)
- Horario inusual → incluir en reporte al cierre del turno (baja urgencia)

### 3.5 Historial y búsqueda

El auditor o gerente puede buscar anomalías pasadas por:
- Rango de fechas
- Folio de cuenta
- Cajero/usuario
- Tipo de anomalía
- Decisión tomada (autorizado / desestimado / escalado)

---

## 4. Explicabilidad para el equipo contable

> **Principio fundamental:** nunca mostrar el score numérico del modelo ni términos como "Isolation Forest" o "SHAP". Traducir cada señal técnica a lenguaje de auditoría hotelera.

### 4.1 Taxonomía de anomalías (categorías visibles al usuario)

En lugar de mostrar un score, clasificar cada anomalía en una o más categorías con ícono y texto:

| Ícono | Categoría | Cuándo aplica | Ejemplo de texto para el auditor |
|-------|-----------|---------------|----------------------------------|
| 🔁 | **Posible duplicado** | `feat_dup_mismo_dia_flag=1` o `feat_dup_mismo_minuto_flag=1` | "Existe un cargo idéntico registrado N minutos antes en este folio" |
| 💰 | **Monto fuera de rango** | `feat_monto_z_codigo_carabo` > umbral | "Este cargo de $X es N veces el monto habitual del concepto CÓDIGO" |
| 📅 | **Fuera de período de estancia** | `feat_cargo_fuera_estancia=1` | "El cargo es anterior a la llegada / posterior a la salida del huésped" |
| 🌙 | **Horario inusual** | `feat_es_madrugada=1` | "Registrado a las HH:MM AM, fuera del horario operativo habitual" |
| 📊 | **Proporción fiscal incorrecta** | `feat_impuesto_ratio_abs` fuera de rango | "El IVA calculado difiere del 16% esperado para este tipo de cargo" |
| ↔️ | **Inconsistencia de signo** | `feat_monto_negativo_sin_abono=1` | "Monto negativo sin marca de abono — revisar si es devolución o error" |
| ✏️ | **Modificación sospechosa** | `feat_usuario_mod_distinto=1` | "Modificado por un usuario distinto al cajero original" |
| 📦 | **Alta densidad de cargos** | `feat_folio_codigo_dia_count_log` alto | "Más de N cargos del mismo concepto en este folio el mismo día" |
| 🚫 | **Cancelación irregular** | `feat_cargo_cancelado=1` + score alto | "Cargo cancelado con características atípicas — verificar si se aplicó crédito" |

### 4.2 Generación de texto explicativo

El sistema combina los valores de las features de la transacción (disponibles en `shap_anomalies.parquet`) con plantillas predefinidas para generar la explicación. Las N razones con mayor `|SHAP|` se presentan en orden de importancia.

**Plantillas de ejemplo:**

```python
# Pseudocódigo de generación de explicación
razones = []

if feat_dup_mismo_minuto_flag == 1:
    razones.append(
        f"⚠ Cargo idéntico ({codigo}, ${monto:,.0f}) registrado hace "
        f"{minutos_diferencia} min en el mismo folio por {usuario}"
    )

if abs(feat_monto_z_codigo_carabo) > 3:
    multiplicador = monto_abs / mediana_codigo
    razones.append(
        f"⚠ Monto {multiplicador:.1f}× el promedio del concepto {codigo} "
        f"(promedio: ${mediana_codigo:,.0f}, este cargo: ${monto_abs:,.0f})"
    )

if feat_cargo_antes_llegada == 1:
    razones.append(
        f"⚠ Cargo fechado antes de la llegada del huésped "
        f"(llegada: {fecha_llegada}, cargo: {fecha_cargo})"
    )
```

### 4.3 Nivel de severidad visible

En lugar de decimales de score, mostrar tres niveles con color:

- 🔴 **Alta prioridad** — revisar antes del cierre del turno
- 🟡 **Media prioridad** — revisar en las próximas horas
- 🟢 **Informativa** — documentada, no requiere acción urgente

---

## 5. Flujo de trabajo del auditor

```
Inicio de turno
      │
      ▼
Ver dashboard: ¿cuántas anomalías detectadas desde el último turno?
      │
      ├─► Revisar alertas urgentes primero (🔴)
      │         │
      │         ├─► Autorizar (cargo correcto, continúa en folio)
      │         ├─► Desestimar + nota ("huésped solicitó cargo adicional, OK")
      │         └─► Escalar + nota → notificación automática a gerencia
      │
      ├─► Revisar cola media prioridad (🟡)
      │
      └─► Cierre de turno: generar reporte diario → enviar a gerencia
```

Cada acción se almacena con: usuario que decidió, timestamp, nota libre, estado final.

---

## 6. Reportes diarios

### 6.1 Contenido del reporte

El reporte se genera automáticamente al final del turno nocturno (configurable, ej. 07:00 AM) y se envía por correo a gerencia y al archivo del hotel:

**Secciones:**
1. **Resumen ejecutivo** — N anomalías detectadas, N revisadas, N escaladas, N pendientes
2. **Anomalías de alta prioridad** — tabla con folio, concepto, monto, tipo, acción tomada
3. **Estadísticas del turno** — cajeros con más anomalías, conceptos más frecuentes, horas pico
4. **Comparativo semanal** — ¿aumentaron o disminuyeron las anomalías vs. la semana pasada?
5. **Pendientes para el siguiente turno** — lista de anomalías sin revisar

### 6.2 Formatos de entrega

- **PDF** — para archivo y firma del auditor
- **Excel** — para análisis contable manual
- **Email automático** — al gerente de turno y al correo del hotel
- **Panel web** — histórico de reportes consultable desde la interfaz

### 6.3 Orquestación

Usar el flujo de Prefect ya existente para agregar una tarea de reporte diario:

```python
# En pipeline/finanom_flow.py (sugerencia)
@flow
def reporte_diario():
    cargar_nuevas_transacciones()
    score_incrementales()       # solo las del día
    generar_alertas()
    compilar_reporte_pdf()
    enviar_por_correo()
```

---

## 7. Mejoras al modelo

### 7.1 Bucle de retroalimentación supervisada (prioridad alta)

Es la mejora con mayor impacto. Cada vez que un auditor toma una decisión (autorizar / escalar), esa etiqueta se convierte en dato de entrenamiento. Con suficientes etiquetas (~500+), se puede entrenar un clasificador supervisado:

```
Hoy:  Isolation Forest (no supervisado) → score genérico
Fase 2:  XGBoost / LightGBM sobre feat_* con etiquetas del auditor
         → mayor precisión, menos falsos positivos
         → el modelo aprende qué anomalías importan *para este hotel*
```

Ventaja: cada hotel tiene su perfil operativo particular; el modelo supervisado captura eso.

### 7.2 Detección contextual por grupos (peer-group analysis)

El modelo actual compara cada transacción contra el universo completo. Una mejora es comparar dentro de grupos homogéneos:

- Mismo concepto financiero (`t_codigo`) → ¿el monto es atípico para *este* concepto?
- Mismo tipo de habitación → ¿la tarifa aplicada es coherente con el tipo?
- Mismo cajero → ¿este usuario tiene un patrón distinto al resto?
- Mismo segmento de mercado → ¿los cargos son coherentes con el canal de venta?

Implementación: entrenar un IF separado por grupo, o usar `feat_monto_z_codigo_carabo` (ya existe) y ampliar a más dimensiones.

### 7.3 Detección de anomalías temporales (series de tiempo)

El modelo actual evalúa cada transacción de forma aislada. Se puede complementar con:

- **Prophet / SARIMA** sobre el monto total diario por concepto → detectar días con volumen inusual
- **Análisis de secuencia por folio** → un LSTM o reglas de secuencia que detecte patrones raros en el orden de cargos de una cuenta (ej. RENHAB → CANCELACIÓN → RENHAB en 5 minutos)

Esto captura anomalías que no se ven transacción por transacción pero sí en el flujo temporal de una cuenta.

### 7.4 Ensemble de modelo + reglas de negocio

Combinar el score del IF con un score de reglas explícitas. Las reglas son 100% interpretables y no requieren entrenamiento:

| Regla | Score adicional |
|-------|-----------------|
| Duplicado exacto en < 5 min | +30 puntos de riesgo |
| IVA ≠ 16% en concepto gravado | +20 puntos |
| Cargo post-checkout en folio cerrado | +25 puntos |
| Modificación nocturna por usuario no-auditor | +15 puntos |

Score final = α × score_modelo + (1−α) × score_reglas. El auditor siempre puede ver la descomposición.

### 7.5 Monitoreo de drift del modelo

Con el tiempo, los patrones del hotel cambian (temporada alta, remodelaciones, cambios de PMS). Implementar:

- **Detección de drift de features**: monitorear que la distribución de las 63 features no cambie bruscamente semana a semana
- **Alerta de reentrenamiento**: cuando el drift supera un umbral, notificar al equipo técnico para reentrenar con datos recientes
- **Reentrenamiento programado**: mensual o trimestral con los últimos N meses de datos

### 7.6 Modelos alternativos a evaluar

| Modelo | Ventaja vs. IF actual | Cuándo considerarlo |
|--------|----------------------|---------------------|
| **XGBoost/LightGBM** (supervisado) | Mayor precisión con etiquetas | Cuando se tengan ≥500 etiquetas del auditor |
| **Autoencoder tabular** | Captura interacciones no lineales complejas | Si el IF produce demasiados falsos positivos |
| **LOF por segmento** | Mejor para anomalías de densidad local | Si los grupos son bien definidos y pequeños |
| **Copula-based (ECOD/COPOD)** | Modela dependencias entre features | Como segundo modelo en ensemble |
| **Regresión de cuantiles** | Predice rangos esperados para monto por contexto | Para el módulo de "monto fuera de rango" |

---

## 8. Funcionalidades adicionales recomendadas

### 8.1 Perfil de riesgo por cajero

El sistema puede acumular, por usuario del PMS, un historial de anomalías asociadas:
- ¿Qué porcentaje de sus transacciones son flaggeadas?
- ¿Qué tipos de anomalías aparecen más en su operación?
- Tendencia: ¿mejora o empeora semana a semana?

Útil para identificar si un problema es sistemático (entrenamiento) o puntual (error aislado).

### 8.2 Vista multi-hotel (cadena)

Si el sistema se despliega en varios hoteles de una cadena:
- Dashboard consolidado con anomalías de todas las propiedades
- Comparativa entre hoteles: ¿cuál tiene mayor tasa de anomalías?
- Modelo compartido vs. modelo por propiedad (a definir según homogeneidad de los hoteles)

### 8.3 Integración directa con el PMS

En lugar de exportar parquets, conectar directamente al PMS vía API o base de datos:
- Scoring incremental: cada N minutos procesar solo las transacciones nuevas
- Retroescritura: marcar la transacción en el PMS con un flag de revisión pendiente
- Link directo: desde la alerta, abrir la transacción en el PMS con un clic

### 8.4 Modo de inspección libre

Para auditorías profundas, ofrecer una pantalla donde el auditor pueda:
- Buscar cualquier folio y ver todas sus transacciones ordenadas por anomalía
- Comparar un cargo contra el histórico del mismo concepto
- Exportar el detalle de un folio a Excel con los scores incluidos

### 8.5 Explicaciones en lenguaje natural (LLM)

Una mejora de experiencia de usuario: en lugar de plantillas fijas, usar un LLM pequeño (ej. Claude claude-haiku-4-5) para generar una explicación en prosa a partir de las features y el contexto de la transacción:

```
Entrada al LLM:
  - Transacción: RENHAB $2,400, 03:12 AM, folio 4821
  - Anomalías detectadas: duplicado en 3 min, horario inusual, modificación por usuario distinto
  - Contexto: tarifa pactada $2,100/noche, estancia 23-26 may

Salida esperada:
  "Esta transacción de renta ($2,400) fue registrada a las 3:12 AM y coincide con
   un cargo idéntico postado tres minutos antes por el cajero MLOPEZ. El monto
   es ligeramente superior a la tarifa pactada en la reservación ($2,100/noche).
   Se recomienda verificar si se trata de un doble posteo o de un ajuste tarifario
   no documentado."
```

Ventaja: el auditor lee una oración, no una tabla de métricas.

---

## 9. Hoja de ruta sugerida

### Fase 1 — MVP funcional (1–2 meses)

- [ ] API FastAPI que sirve anomalías del día desde `scored_transactions.parquet`
- [ ] Interfaz web básica (Streamlit o React): lista, detalle, botones de acción
- [ ] Explicaciones por plantillas (taxonomía de la Sección 4)
- [ ] Reporte diario en PDF/email con Prefect
- [ ] Base de datos para guardar decisiones del auditor

### Fase 2 — Producción básica (2–3 meses)

- [ ] Integración con PMS (API o polling a BD)
- [ ] Scoring incremental cada 15 minutos (alertas near-real-time)
- [ ] Alertas por correo/SMS configurables por tipo y urgencia
- [ ] Perfil de riesgo por cajero
- [ ] Reentrenamiento mensual programado

### Fase 3 — Madurez del modelo (3–6 meses)

- [ ] Bucle supervisado: clasificador entrenado con etiquetas del auditor
- [ ] Ensemble modelo + reglas de negocio
- [ ] Detección de drift y alertas de reentrenamiento
- [ ] Multi-hotel si aplica
- [ ] Explicaciones en lenguaje natural (LLM)

---

## 10. Consideraciones de seguridad y privacidad

- **Datos de huéspedes:** los nombres y datos personales no deben mostrarse en la interfaz de anomalías (ya están ausentes del modelo; implementar enmascaramiento en la UI)
- **Control de acceso:** rol de auditor (ve y decide sobre su turno), rol de gerente (ve histórico y reportes), rol admin (configura umbrales y modelos)
- **Trazabilidad:** toda acción del auditor queda registrada con usuario, timestamp y nota — cumple requerimientos de auditoría contable
- **Retención de datos:** definir política de cuánto tiempo se conservan las anomalías y decisiones (recomendación: mínimo 2 años)
