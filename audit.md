# Auditoría metodológica — Clasificación de riesgo de emergencias de tránsito (Grupo 10, P2)

**Fecha de auditoría:** 16 de agosto de 2026
**Alcance:** repositorio completo `Clasificaci-nRiesgo/` (código, notebook, artefactos, tests, documentación)
**Modo:** solo inspección y verificación. No se modificó código, README, tests, notebook ni configuración.

**Método:** además de leer el código, se recompiló el pipeline completo desde los 54 CSV originales
(15 510 884 filas nacionales, 90 s) con un script independiente y se ejecutaron **9 pruebas de
inyección de fuga**: perturbar deliberadamente el futuro y comprobar que el pasado no se mueve.
Las cifras publicadas se reconstruyeron aritméticamente desde la matriz de confusión, y `aucROC_ovr`
y `brier` se contrastaron contra implementaciones independientes.

---

## A. VEREDICTO GENERAL

**SÓLIDO CON CORRECCIONES — de documentación e interpretación, no de metodología.**

No se encontró fuga de información. Las nueve pruebas de fuga pasan. La partición temporal, los
umbrales, los bins, la elegibilidad y la predicción de la semana objetivo son correctos y
verificables. El Random Forest es genuinamente propio.

El problema real no es el código: es que **la métrica titular (F1 macro = 0.7822) está inflada por
una clase trivialmente separable**, y el README no lo declara con la fuerza que corresponde. Además
el README reporta un entorno y unos tiempos que no son los de la corrida almacenada.

Ningún hallazgo obliga a reentrenar ni invalida los resultados.

---

## B. RESUMEN EJECUTIVO (10 hallazgos)

| # | Sev | Hallazgo |
|---|---|---|
| 1 | 🟠 | La clase **Alto ≡ cabecera cantonal**. Regla trivial `unidad == Guayaquil → Alto` da **F1 = 1.0000** en test (0.9972 en val). Ninguna otra parroquia es Alto jamás, en ningún split. |
| 2 | 🟠 | `src/metricas.py` (155 líneas) tiene **cero cobertura de tests**. Todas las cifras publicadas salen de ahí. (Verificadas a mano: son correctas.) |
| 3 | 🟠 | `ParticionadorTemporal` — el mecanismo antifuga central — **no tiene ninguna prueba**. |
| 4 | 🟠 | README §2 y §20 reportan **entorno y tiempos que no son los de la corrida almacenada** (Windows 10 / Py 3.13.14 / 4 CPU / 143.6 s vs. Windows 11 / Py 3.14.3 / 8 CPU / 128.6 s en `metricas.json`). |
| 5 | 🟡 | **Corrida no reproducible hoy**: `requirements.txt` sin pines, el `.venv` del kernel no existe, y no hay ningún intérprete en la máquina con sklearn/joblib/pyarrow/pytest. |
| 6 | 🟡 | Los **16 días "Bajo" de la cabecera en TRAIN son exactamente los de la anomalía de enero 2024**. Etiquetas espurias derivadas de un defecto de la fuente. |
| 7 | 🟡 | Con P60 = 0, "Bajo" significa **exactamente cero incidentes**. Para 5 de las 6 parroquias el problema real es binario (¿hay ≥ 1 incidente?). No está explicado. |
| 8 | 🟡 | `densidad_historica_parroquia` actúa como **proxy de identidad de unidad** (Guayaquil ocupa bins 26–31, casi disjuntos). El docstring del código lo admite; el README no. |
| 9 | 🟡 | `metricas.json` contiene literales `NaN` → **no es JSON estándar**. |
| 10 | 🟢 | **Sin fuga temporal.** 9/9 pruebas de inyección pasan. Partición, umbrales, bins, elegibilidad y semana objetivo verificados independientemente. |

---

## C. HALLAZGOS ALTOS

### C.1 🟠 La clase Alto es trivialmente separable por identidad de unidad

**Evidencia** (recomputada desde los CSV):

```
Distribución bajo/medio/alto por parroquia
                          TRAIN            VAL           TEST
GUAYAQUIL cabecera   [ 16, 472, 716]  [  0,  1, 181]  [  0,  0, 175]
PROGRESO             [472, 732,   0]  [ 52,130,   0]  [ 81, 94,   0]
MORRO                [1098,106,   0]  [164, 18,   0]  [162, 13,   0]
POSORJA              [924, 280,   0]  [138, 44,   0]  [126, 49,   0]
PUNÁ                 [1198,  6,   0]  [182,  0,   0]  [170,  5,   0]
TENGUEL              [893, 311,   0]  [134, 48,   0]  [136, 39,   0]
```

Regla trivial sin modelo, `unidad == cabecera → Alto`:

```
val : precision=0.9945  recall=1.0000  F1=0.9972
test: precision=1.0000  recall=1.0000  F1=1.0000   <- idéntico al del modelo
```

**Causa estructural:** `P90 = 121` es un umbral **absoluto** sobre el conteo. El máximo diario
histórico de las otras cinco parroquias es 9, 4, 6, 1 y 5 incidentes. Superar 121 es geométricamente
imposible para ellas.

**Impacto:** F1 macro promedia tres clases; una se obtiene gratis. El titular 0.7822 incluye ~0.333
de crédito no ganado. También explica por qué las seis configuraciones de la rejilla dieron
`recall_alto = 1.0000` y `precision_alto = 0.9945` idénticos: esa clase no discrimina entre
hiperparámetros.

**Lo que sí es defendible.** Métrica recalculada sobre la parte no trivial (media de F1 Bajo y F1 Medio):

| Modelo | F1 macro (3 clases) | Media F1 Bajo/Medio (parte difícil) |
|---|---:|---:|
| Bosque propio | 0.7822 | **0.6734** |
| sklearn RF | 0.7807 | 0.6711 |
| Moda histórica | 0.7132 | 0.6532 |
| Persistencia | 0.7197 | 0.5796 |
| Clase mayoritaria | 0.2609 | 0.3913 |

**El modelo sigue ganando a los tres baselines en la parte difícil** (+0.020 sobre moda histórica,
+0.094 sobre persistencia). Eso es defendible y honesto. El resultado no se cae; cambia el titular.

**Recomendación:** reportar F1 macro junto a la métrica Bajo/Medio, y decir explícitamente que Alto
es espacialmente degenerado. No eliminar nada.

### C.2 🟠 `src/metricas.py` sin cobertura de tests

**Evidencia:** `grep` sobre `tests/` — cero referencias a `EvaluadorMetricas`, `Baselines`, `brier`,
`aucROC_ovr`, `f1Macro`, `matrizConfusion`, `curvaCalibracion`.

**Verificación realizada en la auditoría** (la que faltaba):

- Aritmética desde la matriz de confusión publicada: accuracy 0.811428…, F1 bajo 0.85441,
  F1 medio 0.49230, F1 macro 0.78224. **Coincide al cuarto decimal con lo reportado.** ✓
- `aucROC_ovr` contra una implementación trapezoidal independiente: idéntica a 1e-9 ✓. Con empates
  masivos difiere de la trapezoidal ingenua — y el rank-based con promedio de rangos es
  **el correcto**. ✓
- `brier`: forma-suma multiclase, rango [0, 2]. Uniforme = 0.6667, perfecto = 0.0, reportado
  0.2845 ✓.
- `matrizConfusion`: filas = real ✓.

**Impacto:** ninguno sobre los números actuales. Riesgo: una regresión futura en evaluación no la
detecta nadie.

### C.3 🟠 `ParticionadorTemporal` sin pruebas

El componente que impide la fuga no tiene test. Verificado manualmente:

```
train  172 sem  s=[12,183]   2021-09-20 -> 2024-12-30
val     26 sem  s=[184,209]  2025-01-06 -> 2025-06-30
test    25 sem  s=[210,234]  2025-07-07 -> 2025-12-22
solapamientos: 0 / 0 / 0     monotonía: True
burn-in excluido: semanas [0..11]
```

Coincide **exactamente** con README §5. Pero no hay regresión que lo proteja.

### C.4 🟠 README reporta un entorno y tiempos que no son los de la corrida almacenada

| Dato | README §2/§20 | `metricas.json` (corrida real) |
|---|---|---|
| Sistema | Windows 10 | Windows 11 |
| Python | 3.13.14 | 3.14.3 |
| CPUs | 4 | 8 |
| Preprocesamiento | 143.6 s | 128.6 s |
| Entrenamiento final | 25.5 s | 18.9 s |
| Memoria pico | 562.6 MB | 593.2 MB |

Son cifras de una corrida **anterior** que sobrevivieron a la actualización del README. El resto del
README (F1, matriz de confusión, particiones, 739 registros, convergencia) sí coincide con la corrida
almacenada.

---

## D. HALLAZGOS MEDIOS

### D.1 🟡 Corrida no reproducible hoy

El kernel del notebook es `.venv`, que no existe en el proyecto. Ningún Python de la máquina
(3.14 en `C:\Python314`, 3.11) tiene sklearn, joblib, pyarrow, streamlit ni pytest. Además
`requirements.txt` no tiene pines y las versiones instaladas (numpy 2.4.6 / pandas 3.0.3) difieren
de las de la corrida (2.5.2 / 3.0.5). Por eso no se pudieron leer `modelo.joblib` ni
`matriz_riesgo.parquet` directamente.

### D.2 🟡 Enero 2024 contamina las etiquetas de TRAIN

Los 16 días etiquetados "Bajo" de la cabecera en entrenamiento son **todos** de enero 2024:

```
s=131 2024-01-01  c=[1,0,0,1,0,1,1]
s=132 2024-01-08  c=[1,0,1,1,0,0,1]
s=133 2024-01-15  c=[2,0,0,0,0,2,0]
s=134 2024-01-22  c=[0,0,2,2,1,1,1]
s=135 2024-01-29  c=[0,0,0,134,165,136,110]   <- los datos vuelven el 1-feb
eventos enero 2024: 572   |  enero 2023: 3,242  |  enero 2025: 6,130
```

Verificado que el archivo trae 20 registros de Tránsito para Guayaquil, tal como documenta el README.
La anomalía **es de la fuente**, no del pipeline: `descartados_fecha = 0`,
`descartados_sin_unidad = 0`, ningún filtro pierde registros. No hay imputación silenciosa. La
decisión de no corregir es defendible; el efecto colateral (16 etiquetas espurias en train) no está
declarado.

### D.3 🟡 Semántica de las clases no explicada

Con `P60 = 0`: Bajo = *cero* incidentes, Medio = 1–121, Alto = >121. Consecuencia real, por parroquia
(train):

```
                     max diario   %días ≥1   %días >121
GUAYAQUIL cabecera        450       98.7%      59.5%
PROGRESO                    9       60.8%       0.0%
MORRO                       4        8.8%       0.0%
POSORJA                     6       23.3%       0.0%
PUNÁ                        1        0.5%       0.0%
TENGUEL                     5       25.8%       0.0%
```

Para cinco parroquias el modelo resuelve un problema binario: "¿habrá al menos un incidente?". Es
metodológicamente coherente (riesgo absoluto, no relativo) y el umbral **es** el percentil 60 real
— verificado `np.percentile(train, [60,90]) = [0, 121]` exactamente. **No se recomienda cambiar
P60/P90.** Se recomienda explicar qué significan.

### D.4 🟡 `densidad_historica_parroquia` como proxy de unidad

Bins ocupados por parroquia:

```
GUAYAQUIL  [0, 26,27,28,29,30,31]      PUNÁ     [0, 1,2,3,4,5]
PROGRESO   [0, 12,18,21..26]           MORRO    [0, 5..10]
POSORJA    [0, 10..18]                 TENGUEL  [0, 10..21]
```

`Cod_Parroquia` **no** entra como número — eso es cierto y está verificado. Pero la densidad
histórica identifica a la cabecera casi unívocamente. El docstring de `caracteristicas.py:534` lo
dice ("la señal espacial que queda es la propia identidad de la unidad, vía su densidad histórica");
el README no.

### D.5 🟡 `metricas.json` no es JSON estándar

Contiene `NaN` en los Brier de los baselines. `json.load` de Python lo acepta; muchos parsers no.

### D.6 🟡 `f1_macro` por parroquia es engañoso en el artefacto

Guayaquil aparece con `f1_macro = 0.3333` y `exactitud = 1.0` porque el macro promedia tres clases y
solo hay una presente. El notebook imprime un aviso y el README hace bien en mostrar solo Accuracy;
el artefacto conserva la columna sin contexto.

### D.7 🟡 `Baselines` de `metricas.py` es código muerto

El notebook reimplementa persistencia y moda inline (correctamente, con máscara de elegibilidad y
layout `(g,t,s)`). La clase `Baselines` produce un layout distinto y **no aplica elegibilidad** — si
alguien la usara, obtendría cifras equivocadas. Comprobado ejecutándola.

---

## E. HALLAZGOS BAJOS

- **🔵 E.1** `test_representacion.py` duplicado y obsoleto en la raíz del repo, con ediciones locales
  sin commitear; le faltan los 2 tests nuevos de semanas incompletas. No lo recoge pytest.
- **🔵 E.2** Los bins cuantílicos colapsan por cero-inflación: `rezago_*` usa 10 de 32 bins (bordes
  `[0,1,2,3,83,110,123,137,156]`), `media_movil_4` usa 14, `media_movil_12` usa 20. El bin superior
  de `rezago_1` agrupa valores 157–450. Hay resolución alrededor de 121 (bordes 110 y 123), así que
  no daña la clase Alto.
- **🔵 E.3** `_ultima_semana_completa` usa `max(fecha)` de los datos **ya filtrados**. Si la última
  semana real no tuviera ningún incidente de tránsito en Guayaquil, se descartaría por error. No
  ocurrió aquí.
- **🔵 E.4** La primera semana (2021-06-28) es parcial: los datos empiezan el 01-07. Cuenta 3 días
  como cero. Queda en burn-in, pero alimenta `media_movil_12` de la semana 12, la primera supervisada.
  Efecto despreciable, no declarado.
- **🔵 E.5** Convención del Brier ([0,2], forma-suma) no declarada en el README.
- **🔵 E.6** `bosque_base` (100 árboles, prof. 12) se entrena en la celda 13 y se descarta. Coste
  inútil.
- **🔵 E.7** sklearn recibe los hiperparámetros elegidos por la validación **del modelo propio**, sin
  búsqueda propia; y la diferencia +0.0015 es de 1–2 instancias, con una sola semilla.

---

## F. LO QUE ESTÁ CORRECTO — NO TOCAR

1. **Orden del pipeline antifuga** (`caracteristicas.py` docstring) — partición → elegibilidad →
   umbrales → bins → X.
2. **`_rezago`**: verificado `r[s] == c[s-k]` para k=1..4, con ceros iniciales.
3. **`_media_movil`**: verificado, ventana `[s-w, s-1]`, no incluye s.
4. **`_densidad_acumulada`**: verificado idéntica a la media expandida de semanas < s.
5. **Ajuste de bordes solo con TRAIN + filas elegibles** (`construirX`, `sel`).
6. **Umbrales solo con TRAIN**, congelados en `construirY`.
7. **Máscara de elegibilidad solo con TRAIN.**
8. **Descarte de la semana parcial 29–31/12** — el mecanismo correcto y bien testeado.
9. **Feriados**: los 8 fijos de la Ley Orgánica + Carnaval (lunes y martes) + Viernes Santo,
   calculados con Meeus/Jones/Butcher. Verificados para 2022, 2025 y 2026. Marca la fecha exacta,
   no la semana.
10. **Calendario de la semana objetivo**: verificado día a día — `mes` salta de 11 (dic) a 0 (ene)
    el 1-ene, `es_feriado = 1` exactamente en Año Nuevo, fin de semana correcto.
11. **Random Forest propio**: CART, Gini ponderado, histograma por bins con `bincount`, bootstrap,
    subespacio `sqrt(d)=3` por nodo, pesos `(n/(K·n_k))^α`, probabilidades de hoja, agregación,
    importancia. Sin sklearn en ninguna ruta.
12. **Orden de ejecución del notebook**: `execution_count` 1…21 monótono, sin huecos, sin celdas sin
    ejecutar. Test se toca en `exec=13`, después de la selección en `exec=11`.
13. **`app.py`** no importa sklearn, no reentrena, solo lee `salidas/`.
14. **Predicción final**: 42 filas, probabilidades suman 1, `prob_clase_predicha = max(p)`, unidades
    no elegibles en NaN.
15. **Descarte de la semana parcial no pierde registros**: 268 049 − 267 310 = **739**, exactamente
    lo documentado.

---

## G. LEAKAGE AUDIT

Se ejecutaron pruebas de **inyección**, no de inspección: perturbar el futuro y comprobar que el
pasado no se mueve.

| # | Prueba | Resultado |
|---|---|---|
| 1 | Destruir todas las semanas ≥ test (`c[:,:,210:] = 9999`) y recomputar X | **0 celdas distintas** en las 8 820 filas con s < 210 |
| 2 | Destruir val+test y recomputar los bordes de discretización | **Bordes idénticos** en las 11 columnas |
| 3 | Destruir val+test y recomputar umbrales | **0.0 / 121.0**, idénticos |
| 4 | Vaciar TRAIN → máscara | 0 de 42 elegibles (correcto) |
| 4b | Vaciar VAL+TEST → máscara | 42, **idéntica** a la original |
| 5 | Perturbar la semana futura `c[S] = 5555` | X de la semana objetivo **idéntico** |
| 6 | Semana 2025-12-29 en el tensor | **Ausente**. S = 235, última = 2025-12-22 |
| 7 | `densidad(s)` vs media expandida de semanas < s | **Coincide** |
| 8 | `rezago_k[s] == c[s-k]`, `media_movil_w` | **Correcto** para k=1..4, w=4,12 |
| 9 | ¿`Cod_Parroquia` como feature? | **No.** 11 columnas, ninguna es el identificador |

**Vector por vector:**

- **Entrenamiento** — solo `sem_train` (s=12..183). Sin shuffle.
- **Etiquetas** — `np.percentile(c[:,:,sem_train])`. Congeladas para val/test.
- **Features** — todas estrictamente `< s`. Nota importante y **legítima**: las features de una
  semana de test usan conteos observados de semanas de test anteriores. Eso es predicción a un paso
  con historial observado, disponible en operación real. No es fuga.
- **Bins** — `sel = mask_train & mask_elegible`.
- **Elegibilidad** — solo train; una parroquia que apareciera en 2025 quedaría fuera (probado con
  tensor sintético).
- **Normalización** — no existe; solo discretización por cuantiles.
- **Baselines** — persistencia usa `y[s-1]` (pasado observado); moda usa solo `sem_train`;
  mayoritaria usa la moda de `ytr`.
- **Hiperparámetros** — rejilla evaluada solo contra `yva`; test se toca después.
- **Evaluación** — una sola pasada sobre test.
- **Predicción final** — `predecir_matriz_semana` solo pasa filas elegibles; el resto queda NaN.

**Conclusión: sin fuga.**

---

## H. LABEL AUDIT

```
plano = c[:, :, sem_train][mask_elegible].ravel()     # 7,224 valores
np.percentile(plano, [60, 90]) = [0.0, 121.0]         # verificado
63.69% ceros, media 23.4164, máx 450
```

Regla: `y = (c > 0) + (c > 121)`. Proporciones en train: **63.690 / 26.398 / 9.911 %**.

Escalera: se detuvo en el **primer peldaño (`global`)**. `escalera_recorrida` en `metricas.json`
tiene una sola entrada. El README acierta: no se recurrió a fallbacks.
`corresponde_a_percentiles_solicitados = true` — correcto, porque los tres primeros peldaños usan
`np.percentile` por construcción; solo `discreto` los abandonaría.

**P60 = 0 no es un error.** Con 63.69 % de ceros, el percentil 60 de una variable discreta *es* 0.
La escalera existe precisamente para esto y no se activó porque el peldaño global sí produjo tres
clases no vacías. No se recomienda cambiarlo.

---

## I. FEATURE AUDIT

11 columnas, `uint8`, confirmadas:

| Feature | Definición | Temporalidad | Bins usados | Importancia |
|---|---|---|---|---:|
| `dia_semana` | 0=lun…6=dom | conocida | 7 | 0.0366 |
| `es_fin_de_semana` | día ≥ 5 | conocida | 2 | 0.0062 |
| `mes` | mes de la **fecha exacta** (t,s) | conocida | 12 | 0.0547 |
| `es_feriado` | feriado en la fecha exacta | conocida | 2 | 0.0036 |
| `rezago_1..4` | `c[s-k]` | estricto pasado | 10 c/u | 0.182 / 0.131 / 0.061 / 0.049 |
| `media_movil_4` | media `c[s-4..s-1]` | estricto pasado | 14 | 0.1461 |
| `media_movil_12` | media `c[s-12..s-1]` | estricto pasado | 20 | 0.1690 |
| `densidad_historica_parroquia` | media expandida por unidad, semanas `< s` | estricto pasado | 32 | 0.1601 |

- **Faltantes**: no hay NaN. Las semanas iniciales sin historial reciben 0, y quedan cubiertas por
  `BURN_IN = 12`.
- **`mes` y feriados por fecha exacta**, no replicados desde el lunes — verificado en la semana
  objetivo, que cruza el cambio de año.
- **Sin vecindad espacial** — decisión correcta: la fuente no trae coordenadas y Puná es una isla.
- **`Cod_Parroquia` no es feature.** Confirmado. (Ver D.4 sobre el proxy.)

---

## J. MODEL AUDIT

**Es una implementación propia real.** `modelo.joblib` guarda `BosqueAleatorioPropio`; ninguna ruta
de predicción toca sklearn.

- **CART**: recursión en `construir`, paradas por profundidad, `2·minMuestrasHoja` y nodo puro.
- **Gini ponderado**: `1 − Σ(W_k/W)²`, vectorizado para (m,3).
- **Búsqueda de cortes**: histograma `(bin, clase)` con `bincount` + suma acumulada. Evalúa los 31
  cortes de una vez. `O(|Q|)` por nodo en lugar de `O(|Q| log|Q|)`. Decisión de ingeniería sólida y
  bien documentada.
- **Bootstrap**: `rng.integers(0, n, size=min(200000, n))` → 7 224 con reemplazo.
- **Subespacio**: `rng.choice(d, size=3, replace=False)` **por nodo**.
- **Pesos**: `w = [0.5234, 1.2627, 3.3631]` = `n/(3·n_k)` con α=1.0. Verificado contra la
  distribución 63.69/26.40/9.91.
- **Hojas**: distribución normalizada; el árbol se compila a arrays planos y predice vectorizado por
  niveles.
- **Importancia**: reducción de impureza ponderada acumulada, normalizada.

**Hiperparámetros confirmados** en `metricas.json` y en la salida del notebook: 100 árboles,
prof. 16, hoja 5, 32 bins, semilla 42, α=1.0, m=3, submuestra 7 224.

**Selección limpia.** Celda 16 (rejilla 3×2, 30 árboles, solo val) → celda 17 (elige 16/1.0,
**reentrena** con 100 árboles) → celda 18 (convergencia, solo val) → celda 20 (**primer contacto con
test**). `execution_count` 10 → 11 → 12 → 13. El README dice la verdad.

Detalle: la rejilla con 30 árboles dio 0.8294 para la ganadora; el modelo definitivo con 100 dio
0.8268 en validación. No es contradicción — son modelos distintos — y ambos números están
correctamente etiquetados en README y artefacto.

---

## K. EVALUATION AUDIT

**Métricas correctas.** Se reconstruyó toda la aritmética desde la matriz de confusión publicada y
coincide al cuarto decimal (§C.2). AUC y Brier verificados contra implementaciones independientes.

**Baselines válidos y sin futuro:**

- Mayoritaria → moda de `ytr` (clase Bajo). Accuracy 0.6429 = 675/1050 ✓
- Persistencia → `y[s-1]`, etiqueta real de la semana anterior, disponible en operación.
- Moda histórica → moda por unidad-día calculada **solo con `sem_train`**.
- Los tres se restringen a las mismas 1 050 filas elegibles vía `a_filas`.

**Curva de calibración**: confianza `max_k p_k` vs acierto, 10 bins. Es un diagrama de fiabilidad de
top-label, la construcción estándar para multiclase. Válida y bien descrita.

**Comparación con sklearn**: mismos `Xtr/ytr/Xte/yte`, mismos hiperparámetros,
`class_weight="balanced"` ≈ α=1.0, `max_features="sqrt"` ≈ m=3. Justa en lo esencial. La diferencia
+0.0015 no es significativa y el README §12 ya lo dice con cuidado. XGBoost no estaba instalado —
confirmado en la salida (`XGBoost no disponible`) — y no aparece en las métricas. Correcto.

---

## L. FINAL PREDICTION AUDIT

**Válida.** Trazado completo:

1. Los 739 registros del 29–31/12 **entran** en `consolidar` y `filtrarAlcance` (aportan a
   `origen_semanas` y al catálogo de nombres).
2. `agregarConteos` calcula `s_max = 234`, detecta 1 semana incompleta y **trunca el tensor**:
   `c = c[:, :, :235]`. Los 739 registros desaparecen del tensor. Verificado:
   268 049 − 267 310 = 739.
3. `construirX` añade una semana futura de ceros en el índice 235.
4. Las features de esa fila leen `c[234], c[233], …` — todas observadas y completas.
5. Prueba de inyección 5: perturbar `c[235]` no altera ni un bit de X.
6. `predecir_matriz_semana` envía **42 de 42** filas (las 6 parroquias son elegibles).
7. Resultado: Alto 7, Medio 7, Bajo 28. Suma 42 ✓. Probabilidades suman 1, la mostrada es la de la
   clase predicha ✓.

**No existe ningún camino por el que los datos del 29–31 de diciembre entren al modelo.** Los 7 Alto
son la cabecera (los 7 días); los 7 Medio, Progreso.

Matiz honesto: dado el hallazgo C.1, esas 7 predicciones "Alto" equivalen a "es la cabecera
cantonal", no a una alerta discriminativa.

---

## M. TEST COVERAGE — 37 tests

**Qué garantizan:**

| Archivo | n | Garantiza |
|---|---:|---|
| `test_gini.py` | 7 | Pesos de clase, Gini a mano, mejor corte, nodo puro, árbol separable, probabilidades suman 1 |
| `test_pipeline.py` | 12 | **Semana futura no lee `c[S]`** (fuga), fecha +7 días, escalera de umbrales (4 peldaños, casos degenerados), elegibilidad excluye unidades tardías |
| `test_representacion.py` | 18 | T=7, lunes=0, semana lun–dom, filtros cantón/categoría, sin coordenadas, semanas vacías, mes cambia dentro de la semana, feriado en fecha exacta, Carnaval, bins no colapsan, **semanas incompletas (2 regresiones)**, solo elegibles pasan por el modelo |

**Qué NO garantizan — supuestos críticos sin protección:**

1. **Que los bordes de discretización se ajusten solo con TRAIN.** Se testea que no colapsen, no de
   dónde salen. Si alguien pasara `sel` completo, ningún test fallaría. ← el más grave.
2. **Que los umbrales de etiqueta se ajusten solo con TRAIN.** Los tests llaman
   `ajustarUmbrales(c, semanas)` pero nunca comprueban que perturbar val/test no mueva el resultado.
3. **`ParticionadorTemporal` por completo**: sin solapamiento, monotonía, burn-in, cortes.
4. **Todo `metricas.py`**: F1, accuracy, AUC, Brier, matriz de confusión, calibración, baselines.
5. **Que el modelo no vea filas no elegibles en entrenamiento** (se testea en predicción, no en
   `filas()`).
6. **Reproducibilidad por semilla** del bosque completo.

Las cuatro pruebas de fuga ejecutadas en esta auditoría (§G, #1–#4) son exactamente los tests que
faltan. **Un proyecto con 37/37 puede tener las cuatro puertas abiertas y no enterarse** — aquí están
cerradas, pero por el código, no por la suite.

---

## N. MATRIZ DE CONSISTENCIA

`CLAUDE.md` está en la **raíz** del repo, no en `Grupo10_P2/`. No contiene métricas, así que solo
aplica a las filas estructurales.

| Afirmación | README | CLAUDE.md | Código | Notebook | Artefacto | Estado |
|---|---|---|---|---|---|---|
| 54 archivos mensuales | 54 | — | — | 54 leídos | `archivos_validos` n/d | 🟢 |
| 15 510 884 filas nacionales | ✓ | — | — | ✓ | ✓ (reporte) | 🟢 |
| 268 049 tras filtros | ✓ | — | — | ✓ | ✓ | 🟢 |
| 739 descartados / 267 310 en tensor | ✓ | — | ✓ | ✓ | — | 🟢 verificado |
| 235 semanas, 6 parroquias, T=7 | ✓ | ✓ | ✓ | ✓ | ✓ | 🟢 |
| Train 172 / Val 26 / Test 25 sem | ✓ | ✓ | ✓ | ✓ | — | 🟢 verificado |
| n = 7 224 / 1 092 / 1 050 | ✓ | — | — | ✓ | ✓ | 🟢 |
| 11 features | ✓ | ✓ | ✓ | ✓ | `d=11` | 🟢 |
| P60=0 / P90=121, modo `global` | ✓ | ✓ | ✓ | ✓ | ✓ | 🟢 verificado |
| Hiperparámetros 100/16/5/32/42/1.0 | ✓ | — | ✓ | ✓ | ✓ | 🟢 |
| F1 macro test 0.7822 | ✓ | — | — | ✓ | ✓ | 🟢 |
| Accuracy 0.8114 | ✓ | — | — | ✓ | ✓ | 🟢 |
| AUC OvR 0.8595/0.7903/1.0 | ✓ | — | — | ✓ | ausente | 🔵 no está en `metricas.json` |
| Brier 0.2845 | ✓ | — | — | ✓ | ✓ | 🟢 |
| Baselines .7197/.7132/.2609 | ✓ | — | — | ✓ | ✓ | 🟢 |
| sklearn 0.7807 | ✓ | — | — | ✓ | ✓ | 🟢 |
| Semana objetivo 2025-12-29 | ✓ | — | ✓ | ✓ | ✓ | 🟢 |
| Alto 7 / Medio 7 / Bajo 28 | ✓ | — | — | ✓ | parquet ✓ | 🟢 |
| 37 tests | ✓ | — | 7+12+18 | `37 passed` | — | 🟢 |
| **Entorno de ejecución** | Win10/3.13.14/4 CPU | — | — | Win11/3.14.3/8 | Win11/3.14.3/8 | 🟠 **README erróneo** |
| **Tiempos y memoria** | 143.6/25.5/562.6 | — | — | 128.6/18.9/593.2 | 128.6/18.9/593.2 | 🟠 **README erróneo** |
| F1 Alto = 1.0 interpretado | parcial (§14) | — | — | aviso | sin contexto | 🟠 insuficiente |
| `Cod_Parroquia` no es feature | ✓ | — | ✓ + matiz | ✓ | — | 🟡 README sin el matiz |
| `metricas.json` es JSON válido | implícito | — | — | — | contiene `NaN` | 🟡 |

---

## O. CAMBIOS RECOMENDADOS — **no implementados**

### O.1 Correcciones obligatorias (antes de entregar)

1. **Corregir README §2 y §20**: copiar entorno y tiempos desde `metricas.json`.
2. **Reforzar la interpretación de F1 Alto = 1.0**: añadir en §11 y §27 que la clase Alto coincide
   con la cabecera cantonal en los tres splits, que la regla trivial `unidad == cabecera → Alto`
   alcanza F1 = 1.0000 en test, y reportar la métrica Bajo/Medio (propio 0.6734 vs moda 0.6532 vs
   persistencia 0.5796) como evidencia de la capacidad no trivial.
3. **Explicar la semántica de las clases** con P60 = 0: Bajo = cero incidentes; para cinco parroquias
   el problema es binario.

### O.2 Correcciones recomendadas

4. Cuatro tests de regresión de fuga (§G #1–#4): bordes solo-train, umbrales solo-train, elegibilidad
   solo-train, futuro no altera pasado.
5. Tests unitarios de `metricas.py` contra la matriz de confusión publicada.
6. Test de `ParticionadorTemporal` (solapamiento, monotonía, burn-in).
7. Declarar en §15.5 que los 16 días "Bajo" de la cabecera en entrenamiento provienen de la anomalía
   de enero 2024.
8. Añadir a §7 el matiz de `densidad_historica_parroquia` como señal de identidad de unidad (ya está
   en el docstring del código).
9. Pinear `requirements.txt` a las versiones de la corrida (numpy 2.5.2, pandas 3.0.5, …) y anotar
   Python 3.14.3.
10. Borrar el `test_representacion.py` obsoleto de la raíz del repo.
11. Serializar los Brier ausentes como `null`, no `NaN`.
12. Guardar el AUC OvR en `metricas.json` (hoy solo está en el README).

### O.3 Mejoras opcionales — ⚪ no implementar automáticamente

13. Repetir la comparación con sklearn con 5–10 semillas y reportar dispersión, en vez de un único
    +0.0015.
14. Intervalos de confianza bootstrap sobre F1 macro de test.
15. Evaluar `nivel_espacial="canton"` o `alcance=("provincia","GUAYAS")` para romper la degeneración
    espacial de la clase Alto (el `Preprocesador` ya lo soporta).
16. Explorar umbrales **relativos por unidad** como variante documentada, conservando la global como
    principal.
17. Eliminar `bosque_base` o reducirlo a 30 árboles.
18. Retirar `Baselines` de `metricas.py` o alinearla con la implementación del notebook.

---

## Cierre

Metodología limpia y verificable, sin fuga; el trabajo es defendible tal como está. Lo que hay que
arreglar es el relato — dos bloques de cifras obsoletas en el README y una métrica titular que
necesita el asterisco que hoy solo aparece a medias en §14.
