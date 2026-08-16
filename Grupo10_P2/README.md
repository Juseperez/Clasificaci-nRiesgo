# Clasificación del nivel de riesgo de emergencias de tránsito en el cantón Guayaquil

**CCPG1044 — Inteligencia Artificial**  
**Escuela Superior Politécnica del Litoral (ESPOL)**  
**Grupo #10 · Paralelo P2**

### Integrantes

- Bohórquez Villao Andrés Martín
- Pérez Zamora Juan Sebastián
- Ullaguari Cagua John Jairo

---

## 1. Descripción

Este proyecto desarrolla un sistema de Machine Learning para **clasificar el nivel de riesgo de emergencias de tránsito en el cantón Guayaquil**, utilizando datos históricos abiertos publicados por el Servicio Integrado de Seguridad ECU 911.

El sistema clasifica el riesgo en tres niveles:

- **Bajo**
- **Medio**
- **Alto**

La predicción final se realiza para cada combinación:

```text
parroquia × día de la semana
```

con un horizonte de **una semana**.

El clasificador principal es un **Random Forest implementado desde cero en NumPy**. La solución propia incluye:

- inducción de árboles CART;
- criterio de impureza Gini ponderada;
- selección aleatoria de atributos;
- bootstrap;
- ponderación de clases;
- predicción probabilística multiclase;
- agregación de probabilidades entre árboles;
- cálculo de importancia de variables.

Las implementaciones de Machine Learning de terceros no participan en la solución principal. `scikit-learn` se utiliza únicamente como referencia experimental de comparación.

---

## 2. Estado final del proyecto

El proyecto fue **implementado, probado y ejecutado completamente** utilizando los **54 archivos mensuales consecutivos correspondientes al periodo julio de 2021 – diciembre de 2025**.

La versión final verifica:

- **37/37 pruebas automáticas superadas**;
- preprocesamiento completo de los datos del ECU 911;
- descarte explícito de semanas finales incompletas para evitar días faltantes tratados como cero;
- construcción de la representación parroquia × día;
- entrenamiento del Random Forest propio;
- ajuste mediante conjunto de validación;
- evaluación final sobre un conjunto de prueba temporal independiente;
- comparación contra tres baselines;
- comparación contra Random Forest de `scikit-learn`;
- generación de probabilidades por clase;
- análisis de importancia de variables;
- evaluación por parroquia;
- generación de la matriz de riesgo de la semana siguiente;
- interfaz gráfica desarrollada con Streamlit;
- generación y almacenamiento de los artefactos finales.

La corrida definitiva se realizó el **16 de agosto de 2026**.

Entorno registrado:

```text
Sistema operativo : Windows 10
Python            : 3.13.14
CPU disponibles   : 4
NumPy             : 2.5.2
pandas            : 3.0.5
```

---

## 3. Representación final del problema

En las etapas iniciales del proyecto se trabajó con una representación basada en:

```text
celda geográfica de 1 km × 1 km
+
franja horaria de 3 horas
```

Durante la implementación se inspeccionaron directamente los archivos mensuales publicados por el ECU 911 y se comprobó que la fuente abierta utilizada contiene las siguientes siete columnas:

```text
Fecha
provincia
Canton
Cod_Parroquia
Parroquia
Servicio
Subtipo
```

La fuente proporciona:

- fecha a nivel de día;
- provincia;
- cantón;
- parroquia;
- categoría del servicio;
- subtipo del incidente.

Sin embargo, **no proporciona coordenadas geográficas ni hora del incidente**.

Por esta razón, la representación computacional final fue ajustada a la máxima granularidad verificable disponible en los datos.

| Elemento | Representación inicial | Representación final |
|---|---|---|
| Unidad espacial | Celda 1 km × 1 km | **Parroquia (`Cod_Parroquia`)** |
| Unidad temporal | Franja de 3 horas, \|T\| = 56 | **Día de la semana, \|T\| = 7** |
| Horizonte | Una semana | **Una semana** |
| Alcance | Cantón Guayaquil, tránsito | **Cantón Guayaquil, tránsito** |

La instancia final de aprendizaje se representa como:

```text
(parroquia, día de la semana, semana)
```

Este ajuste no modifica el propósito general del proyecto: **anticipar el nivel de riesgo de emergencias de tránsito para la semana siguiente**.

---

## 4. Fuente y alcance de los datos

Fuente:

**Servicio Integrado de Seguridad ECU 911**

Periodo descargado:

```text
julio 2021 – diciembre 2025
```

Categoría:

```text
Tránsito y Movilidad
```

Alcance geográfico:

```text
Cantón Guayaquil
```

Se procesaron **54 archivos mensuales consecutivos**. Después de filtrar categoría y cantón se obtuvieron:

```text
268,049 registros relevantes
```

La última semana calendario iniciada el **29 de diciembre de 2025** no estaba completa: la fuente disponible terminaba el **31 de diciembre de 2025**, por lo que solo contenía tres de sus siete días.

Para impedir que los cuatro días no observados fueran interpretados como conteos cero, esa semana se excluyó íntegramente del tensor observado.

La exclusión retiró **739 registros** pertenecientes a esa semana parcial. El tensor final utilizado por el pipeline contiene:

```text
267,310 eventos
235 semanas completas
6 parroquias
7 días por semana
```

La cabecera cantonal concentra aproximadamente el **98.9 %** de los eventos del tensor, por lo que las métricas agregadas se complementan con resultados por parroquia.

---

## 5. Partición temporal

Para evitar fuga de información entre pasado y futuro se utiliza una **partición estrictamente temporal por semanas**, sin división aleatoria.

El dataset comienza en julio de 2021. Las primeras **12 semanas** se reservan como periodo de inicialización (`BURN_IN = 12`) para disponer de historial suficiente para la ventana `media_movil_12`; por ello no se utilizan como objetivos supervisados.

Las semanas se asignan a cada conjunto según la fecha del **lunes que las inicia**:

| Conjunto | Semanas utilizadas | Nº semanas | Instancias |
|---|---|---:|---:|
| Entrenamiento | 20/09/2021 – 30/12/2024 | 172 | **7,224** |
| Validación | 06/01/2025 – 30/06/2025 | 26 | **1,092** |
| Prueba | 07/07/2025 – 22/12/2025 | 25 | **1,050** |

Cada semana aporta:

```text
6 parroquias × 7 días = 42 instancias
```

El conjunto de prueba se utiliza únicamente para la evaluación final.

La semana parcial iniciada el 29/12/2025 queda fuera de entrenamiento, validación y prueba y se trata como horizonte futuro a clasificar.

---

## 6. Preprocesamiento

El componente principal de preprocesamiento se encuentra en:

```text
src/preprocesamiento.py
```

Sus responsabilidades incluyen:

1. lectura de los archivos mensuales;
2. detección y normalización de columnas;
3. filtrado de la categoría `Tránsito y Movilidad`;
4. filtrado del cantón `GUAYAQUIL`;
5. validación de fechas;
6. validación de unidades espaciales;
7. asignación de parroquia;
8. asignación del día de la semana;
9. agregación temporal de incidentes;
10. detección y descarte de semanas finales incompletas.

### Optimización de memoria

Los archivos originales contienen registros de todo Ecuador.

Para evitar conservar simultáneamente millones de registros nacionales en memoria, cada archivo mensual se filtra **antes de incorporarse al conjunto consolidado**.

```text
CSV mensual
    ↓
lectura
    ↓
filtro Tránsito y Movilidad
    ↓
filtro Guayaquil
    ↓
conservación de registros relevantes
    ↓
siguiente archivo
```

En la corrida definitiva se leyeron **15,510,884 filas nacionales**, de las cuales **268,049** pertenecían al alcance final.

El prefiltrado mensual permitió ejecutar el pipeline sin concatenar previamente todos los registros nacionales.

### Semanas incompletas

Una semana observada se considera válida únicamente si sus siete días están cubiertos por el periodo disponible.

La última semana iniciada el 29/12/2025 solo tenía datos hasta el 31/12/2025 y fue descartada para evitar que los días 1–4 de enero de 2026 fueran representados artificialmente como cero.

El comportamiento está protegido por pruebas de regresión específicas en `tests/test_representacion.py`.

---

## 7. Características predictoras

El modelo utiliza **11 características**:

```text
dia_semana
es_fin_de_semana
mes
es_feriado

rezago_1
rezago_2
rezago_3
rezago_4

media_movil_4
media_movil_12

densidad_historica_parroquia
```

Todas las variables utilizan exclusivamente información disponible **antes de la semana objetivo**.

### Variables temporales

```text
dia_semana
es_fin_de_semana
mes
es_feriado
```

### Variables históricas

```text
rezago_1
rezago_2
rezago_3
rezago_4
media_movil_4
media_movil_12
densidad_historica_parroquia
```

### Feriados

`es_feriado` considera:

- feriados nacionales de fecha fija;
- lunes de Carnaval;
- martes de Carnaval;
- Viernes Santo.

No incorpora traslados extraordinarios de días de descanso definidos mediante decretos anuales.

### Vecindad espacial

No se incluyen características de vecindad entre parroquias.

La fuente no contiene coordenadas, por lo que no existe una relación geométrica verificable que permita construir una vecindad espacial sin introducir supuestos artificiales.

---

## 8. Construcción de la etiqueta

El conteo de emergencias de cada combinación parroquia–día se transforma en tres niveles de riesgo:

```text
Bajo
Medio
Alto
```

Los umbrales se calculan **únicamente utilizando el conjunto de entrenamiento**.

La regla adoptada en la ejecución final fue:

```text
Bajo  : c <= P60
Medio : P60 < c <= P90
Alto  : c > P90
```

La estrategia utilizada por el constructor de etiquetas fue:

```text
global
```

Los umbrales obtenidos fueron:

```text
P60 = 0
P90 = 121
```

Distribución resultante en entrenamiento:

| Clase | Proporción |
|---|---:|
| Bajo | 63.690 % |
| Medio | 26.398 % |
| Alto | 9.911 % |

No fue necesario recurrir a reglas alternativas de construcción de etiquetas.

---

## 9. Random Forest propio

La implementación principal se encuentra en:

```text
src/bosque.py
```

El modelo fue desarrollado desde cero utilizando NumPy.

Incluye:

- `NodoArbol`;
- `ArbolDecisionPropio`;
- `BosqueAleatorioPropio`;
- impureza Gini ponderada;
- búsqueda de cortes;
- selección aleatoria de atributos;
- bootstrap;
- pesos de clase;
- criterios de parada;
- probabilidades en hojas;
- agregación de probabilidades;
- importancia de características.

### Hiperparámetros finales

```text
Número de árboles        : 100
Profundidad máxima       : 16
Mínimo muestras por hoja : 5
Bins                     : 32
Semilla                  : 42
Potencia de pesos        : 1.0
```

La selección aleatoria de atributos se realiza automáticamente siguiendo aproximadamente:

```text
sqrt(d)
```

Con `d = 11`, se consideran aproximadamente **3 atributos candidatos por nodo**.

El parámetro:

```text
maxMuestrasPorArbol = 200000
```

funciona como una cota de seguridad para conjuntos grandes.

En la corrida final:

```text
n_train = 7,224
```

por lo que cada árbol utiliza una muestra bootstrap del mismo tamaño que el conjunto de entrenamiento, con reemplazo.

### Selección de hiperparámetros

La búsqueda se realizó únicamente sobre el conjunto de validación.

Se evaluaron:

```text
profundidad ∈ {8, 12, 16}
potenciaPesos ∈ {1.0, 0.5}
```

utilizando **30 árboles por configuración candidata** para reducir el costo de exploración.

La mejor configuración encontrada fue:

```text
profundidadMax = 16
potenciaPesos  = 1.0
```

El modelo definitivo se reentrenó con **100 árboles**.

El F1 macro del modelo final sobre validación fue:

```text
0.8268
```

---

## 10. Evaluación

La métrica principal es:

```text
F1 macro
```

También se evalúan:

- exactitud;
- precisión, recall y F1 por clase;
- recall de la clase Alto;
- matriz de confusión;
- AUC-ROC One-vs-Rest;
- Brier Score;
- curva de calibración;
- comportamiento por parroquia.

La clase **Alto** recibe especial atención debido al costo potencial de no identificar correctamente un periodo de mayor riesgo.

### Matriz de confusión del bosque propio

Filas = clase real.  
Columnas = clase predicha.

| Real \ Predicha | Bajo | Medio | Alto |
|---|---:|---:|---:|
| Bajo | **581** | 94 | 0 |
| Medio | 104 | **96** | 0 |
| Alto | 0 | 0 | **175** |

### AUC-ROC OvR

```text
Bajo  = 0.8595
Medio = 0.7903
Alto  = 1.0000
```

---

## 11. Resultados finales

### Comparación sobre el conjunto de prueba

| Modelo | Accuracy | F1 macro | F1 Bajo | F1 Medio | F1 Alto | Recall Alto | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Bosque propio (NumPy)** | **0.8114** | **0.7822** | **0.8544** | 0.4923 | **1.0000** | **1.0000** | **0.2845** |
| sklearn RandomForest | 0.8000 | 0.7807 | 0.8421 | **0.5000** | 1.0000 | 1.0000 | 0.3060 |
| Persistencia | 0.7533 | 0.7197 | 0.8083 | 0.3509 | 1.0000 | 1.0000 | — |
| Moda histórica | 0.7743 | 0.7132 | 0.8640 | 0.4424 | 0.8333 | 0.7143 | — |
| Clase mayoritaria | 0.6429 | 0.2609 | 0.7826 | 0.0000 | 0.0000 | 0.0000 | — |

Resultados principales del modelo propio:

```text
Accuracy     = 0.8114
F1 macro     = 0.7822
F1 Bajo      = 0.8544
F1 Medio     = 0.4923
F1 Alto      = 1.0000
Recall Alto  = 1.0000
Brier Score  = 0.2845
```

El modelo supera la meta inicial de referencia de:

```text
F1 macro = 0.60
```

y también supera los tres baselines evaluados sobre exactamente el mismo conjunto de prueba.

La clase **Medio** continúa siendo la más difícil de discriminar.

El desempeño perfecto observado para la clase **Alto** debe interpretarse junto con la fuerte concentración de esa clase en la cabecera cantonal.

---

## 12. Comparación con scikit-learn

El Random Forest propio obtuvo:

```text
F1 macro propio   = 0.7822
F1 macro sklearn  = 0.7807
```

La diferencia fue:

```text
+0.0015
```

a favor de la implementación propia en esta corrida, equivalente a **0.15 puntos porcentuales de F1 macro**.

El propósito de esta comparación no es demostrar superioridad universal, sino verificar que la implementación desarrollada desde cero alcanza un comportamiento comparable a una referencia ampliamente utilizada.

`scikit-learn` **no participa en las predicciones finales de la solución**; se emplea únicamente como benchmark externo.

### XGBoost

El notebook conserva XGBoost como benchmark externo opcional.

En la corrida definitiva XGBoost no estaba instalado, por lo que **no forma parte de las métricas reportadas**.

La comparación reproducida y almacenada corresponde al Random Forest de `scikit-learn`.

---

## 13. Importancia de variables

Las características con mayor importancia por reducción de impureza fueron:

| Variable | Importancia |
|---|---:|
| `rezago_1` | 0.1823 |
| `media_movil_12` | 0.1690 |
| `densidad_historica_parroquia` | 0.1601 |
| `media_movil_4` | 0.1461 |
| `rezago_2` | 0.1315 |
| `rezago_3` | 0.0606 |
| `mes` | 0.0547 |
| `rezago_4` | 0.0493 |
| `dia_semana` | 0.0366 |
| `es_fin_de_semana` | 0.0062 |
| `es_feriado` | 0.0036 |

Según la importancia por reducción de impureza del bosque propio, los **rezagos, medias móviles y la densidad histórica de la parroquia** tuvieron mayor peso en las decisiones internas del modelo que variables simples de calendario como feriado o fin de semana.

Estas importancias describen el comportamiento interno del clasificador y **no deben interpretarse como relaciones causales**.

---

## 14. Desempeño por parroquia

La distribución de los registros no es espacialmente uniforme.

La cabecera cantonal concentra la gran mayoría de las emergencias registradas, por lo que se evalúa también el desempeño de manera desagregada.

Cada parroquia aporta **175 instancias** al conjunto de prueba.

| Parroquia | n | Bajo / Medio / Alto | Accuracy |
|---|---:|---:|---:|
| Guayaquil, cabecera cantonal | 175 | 0 / 0 / 175 | **1.0000** |
| Juan Gómez Rendón (Progreso) | 175 | 81 / 94 / 0 | 0.5200 |
| Morro | 175 | 162 / 13 / 0 | 0.9257 |
| Posorja | 175 | 126 / 49 / 0 | 0.7143 |
| Puná | 175 | 170 / 5 / 0 | 0.9714 |
| Tenguel | 175 | 136 / 39 / 0 | 0.7371 |

La clase **Alto** aparece concentrada en la cabecera cantonal durante el periodo de prueba.

Por esa razón, su:

```text
F1 Alto = 1.0000
```

no debe interpretarse como evidencia de desempeño uniforme en todo el cantón.

La diferenciación entre **Bajo** y **Medio** constituye el principal reto del clasificador.

---

## 15. Limitaciones

### 15.1 Resolución espacial

La fuente utilizada no contiene coordenadas.

Por ello no es posible construir de manera verificable una grilla de:

```text
1 km × 1 km
```

con la fuente abierta utilizada.

La máxima resolución espacial disponible es la **parroquia**.

### 15.2 Resolución temporal

La fuente únicamente incluye la fecha del incidente y no contiene hora.

Por ello no es posible construir de manera verificable:

```text
franjas horarias de 3 horas
```

La representación temporal final utiliza el **día de la semana**.

### 15.3 Concentración espacial

La cabecera cantonal concentra aproximadamente el **98.9 % de los eventos de tránsito del tensor final**.

Esto limita la cantidad de ejemplos disponibles para parroquias con menor volumen y obliga a interpretar las métricas agregadas junto con los resultados por unidad.

### 15.4 Cobertura en los bordes temporales

La publicación utilizada comienza el **1 de julio de 2021** y termina el **31 de diciembre de 2025**.

La primera semana calendario del tensor comienza el **28/06/2021**, por lo que sus primeros tres días quedan fuera del periodo disponible.

Esta semana se encuentra dentro del:

```text
BURN_IN = 12
```

y no se utiliza como objetivo supervisado.

El sistema también descarta explícitamente la semana final incompleta.

Como resultado, el conjunto de prueba termina en la última semana completa iniciada el:

```text
22/12/2025
```

### 15.5 Anomalía de enero de 2024

Durante la inspección de calidad se identificó un comportamiento anómalo en el archivo correspondiente a enero de 2024.

En Guayaquil se observaron:

```text
Diciembre 2023 : 4,511 registros de Tránsito y Movilidad
Enero 2024     :    20 registros de Tránsito y Movilidad
Febrero 2024   : 4,041 registros de Tránsito y Movilidad
```

El archivo de enero de 2024 contiene:

```text
208,194 registros nacionales
40,351 registros correspondientes a Guayaquil
```

por lo que no corresponde a un archivo vacío en términos de volumen total.

A nivel nacional se observaron:

```text
Seguridad Ciudadana  : 200,318
Tránsito y Movilidad :   2,634
```

También se verificó que los subtipos de tránsito observados en febrero no aparecen reclasificados directamente dentro de `Seguridad Ciudadana` en enero para Guayaquil.

No se identificó una regla objetiva y verificable que permitiera corregir o reclasificar los registros de enero de 2024.

Por ello, **no se realizaron imputaciones ni reclasificaciones artificiales**: el archivo se conservó tal como fue publicado y la anomalía se documenta como una limitación de calidad de la fuente.

---

## 16. Predicción de la semana siguiente

La última semana completa observada comienza el:

```text
2025-12-22
```

La matriz de riesgo final corresponde a la semana inmediatamente siguiente:

```text
2025-12-29
```

La fuente contiene registros parciales del 29, 30 y 31 de diciembre, pero esa semana no se utiliza como semana observada porque faltan cuatro de sus siete días.

Por ello se trata íntegramente como el horizonte a clasificar.

Las seis parroquias elegibles producen:

```text
6 parroquias × 7 días = 42 combinaciones
```

Distribución de predicciones:

```text
Alto  : 7
Medio : 7
Bajo  : 28
```

La matriz permite consultar tanto el nivel como la probabilidad asignada a la **clase predicha**.

> La probabilidad mostrada es la confianza probabilística del clasificador en la clase asignada; **no** es la probabilidad absoluta de que ocurra una emergencia.

---

## 17. Interfaz gráfica

La interfaz se encuentra en:

```text
app.py
```

y fue desarrollada utilizando **Streamlit**.

Incluye:

- matriz parroquia × día;
- niveles Bajo / Medio / Alto;
- probabilidad de cada predicción;
- filtros por parroquia;
- filtros por día de la semana;
- filtros por nivel de riesgo;
- filtro por confianza mínima;
- detalle de predicciones;
- distribución de niveles;
- métricas del modelo;
- comparación contra baselines;
- desempeño por parroquia;
- importancia de variables.

La interfaz puede ejecutarse con:

```bash
streamlit run app.py
```

La probabilidad presentada corresponde a la probabilidad de la **clase predicha** por el clasificador.

No debe interpretarse como una estimación directa de la probabilidad absoluta de ocurrencia de una emergencia.

---

## 18. Elegibilidad de parroquias

Una parroquia sin incidentes históricos durante la ventana de entrenamiento no debe interpretarse automáticamente como de riesgo bajo.

Por esta razón:

```text
unidad sin historial
→ no ingresa al modelo
→ riesgo no estimado
```

En la interfaz estas unidades se representan como:

```text
Sin datos suficientes
```

La ausencia de registros puede representar falta de cobertura y no necesariamente ausencia real de riesgo.

En la ejecución final:

```text
Parroquias totales    : 6
Parroquias elegibles  : 6
Parroquias excluidas  : 0
```

---

## 19. Baselines

La evaluación utiliza tres referencias simples sobre exactamente el mismo conjunto de prueba de **1,050 instancias**.

### Clase mayoritaria

Predice siempre la clase más frecuente.

```text
F1 macro = 0.2609
```

### Persistencia

Utiliza como predicción la clase observada en el periodo inmediatamente anterior.

```text
F1 macro = 0.7197
```

### Moda histórica

Utiliza la clase históricamente más frecuente para cada unidad.

```text
F1 macro = 0.7132
```

El bosque propio:

```text
F1 macro = 0.7822
```

supera los tres baselines.

---

## 20. Rendimiento computacional

La corrida definitiva registró:

```text
Preprocesamiento            : 143.6 s
Construcción de X           : ~0.0 s
Entrenamiento bosque final  : 25.5 s
Memoria pico                : 562.6 MB
```

El prefiltrado mensual controla el uso de memoria al evitar conservar simultáneamente registros nacionales fuera del alcance.

### Convergencia del ensamble

La curva de validación del **modelo definitivo** fue:

```text
B =   1  → F1 macro 0.7740
B =   5  → F1 macro 0.8064
B =  10  → F1 macro 0.8210
B =  25  → F1 macro 0.8277
B =  50  → F1 macro 0.8275
B =  75  → F1 macro 0.8268
B = 100  → F1 macro 0.8268
```

El desempeño se estabiliza aproximadamente a partir de **25–50 árboles**.

Se conservan **100 árboles** como configuración final.

---

## 21. Estructura del proyecto

```text
Grupo10_P2/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── 00_inspeccionar_csv.py
├── proyecto.ipynb
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── preprocesamiento.py
│   ├── caracteristicas.py
│   ├── bosque.py
│   ├── metricas.py
│   └── riesgo.py
│
├── tests/
│   ├── test_gini.py
│   ├── test_pipeline.py
│   └── test_representacion.py
│
├── datos/
│   └── archivos CSV mensuales del ECU 911
│
└── salidas/
    ├── metricas.json
    ├── modelo.joblib
    ├── matriz_riesgo.parquet
    ├── unidades.json
    ├── exploratorio.png
    ├── convergencia.png
    ├── calibracion.png
    └── importancias.png
```

---

## 22. Instalación

Se recomienda utilizar un entorno virtual.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las dependencias necesarias están declaradas en:

```text
requirements.txt
```

Incluyen:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
streamlit
pytest
ipykernel
psutil
pyarrow
```

---

## 23. Ejecución

### Paso 1 — Inspeccionar los archivos

```bash
python 00_inspeccionar_csv.py datos/
```

### Paso 2 — Ejecutar pruebas automáticas

```bash
python -m pytest tests/ -q
```

Resultado de la versión final:

```text
..................................... [100%]
37 passed
```

### Paso 3 — Ejecutar el notebook

Abrir:

```text
proyecto.ipynb
```

y ejecutar todas las celdas utilizando el entorno Python del proyecto.

El notebook realiza:

```text
carga
→ preprocesamiento
→ representación
→ partición temporal
→ construcción de etiquetas
→ generación de características
→ entrenamiento
→ validación
→ prueba
→ benchmarks
→ generación de artefactos
```

### Paso 4 — Ejecutar la interfaz

```bash
streamlit run app.py
```

---

## 24. Pruebas automáticas

El proyecto contiene **37 pruebas automáticas**:

```text
7  pruebas del criterio Gini
12 pruebas del pipeline
18 pruebas de representación parroquia × día
```

Para ejecutarlas:

```bash
python -m pytest tests/ -q
```

Resultado final:

```text
..................................... [100%]
37 passed
```

Las pruebas de representación incluyen regresiones específicas para comprobar el tratamiento correcto de semanas incompletas.

---

## 25. Artefactos finales

La ejecución genera:

```text
salidas/metricas.json
salidas/modelo.joblib
salidas/matriz_riesgo.parquet
salidas/unidades.json

salidas/exploratorio.png
salidas/convergencia.png
salidas/calibracion.png
salidas/importancias.png
```

### `metricas.json`

Contiene, entre otros:

- métricas de validación;
- métricas de prueba;
- métricas por clase;
- resultados por parroquia;
- baselines;
- hiperparámetros;
- características utilizadas;
- criterios de elegibilidad;
- entorno de ejecución;
- tiempos;
- consumo de memoria.

### `modelo.joblib`

Contiene el bosque propio entrenado y los objetos necesarios para realizar predicciones.

### `matriz_riesgo.parquet`

Contiene la clasificación generada para la semana siguiente.

### Gráficas

Los artefactos gráficos incluyen:

- análisis exploratorio;
- convergencia del ensamble definitivo;
- curva de calibración;
- importancia de variables.

---

## 26. Conclusiones

Se implementó satisfactoriamente un **Random Forest propio en NumPy** para clasificar el nivel de riesgo de emergencias de tránsito por parroquia y día de la semana en el cantón Guayaquil.

En el conjunto de prueba temporal definitivo obtuvo:

```text
F1 macro = 0.7822
Accuracy = 0.8114
```

superando los tres baselines:

```text
Persistencia      = 0.7197
Moda histórica    = 0.7132
Clase mayoritaria = 0.2609
```

y alcanzando un desempeño prácticamente equivalente al Random Forest de referencia:

```text
Bosque propio = 0.7822
sklearn       = 0.7807
```

La clase **Alto** alcanzó:

```text
F1     = 1.0000
Recall = 1.0000
```

en prueba.

Este resultado debe interpretarse considerando que sus ejemplos se encuentran concentrados en la cabecera cantonal.

La clase **Medio**:

```text
F1 = 0.4923
```

continúa siendo el principal espacio de mejora.

Las importancias del bosque muestran que los rezagos, medias móviles y densidad histórica de la parroquia tuvieron mayor peso interno que las variables simples de calendario, sin que ello implique causalidad.

La versión final integra:

- procesamiento reproducible de 54 archivos mensuales reales;
- control de semanas incompletas;
- representación parroquia × día sin inventar coordenadas ni hora;
- prevención de fuga temporal;
- Random Forest propio;
- ajuste temporal y prueba independiente;
- tres baselines;
- benchmark externo;
- métricas por clase y por parroquia;
- probabilidades y calibración;
- predicción de la semana siguiente;
- interfaz Streamlit;
- **37 pruebas automatizadas**.

---

## 27. Resultado principal

```text
Modelo          : Random Forest propio en NumPy

Validación:
F1 macro        : 0.8268

Prueba:
n               : 1,050
F1 macro        : 0.7822
Accuracy        : 0.8114
F1 Bajo         : 0.8544
F1 Medio        : 0.4923
F1 Alto         : 1.0000
Recall Alto     : 1.0000
Brier           : 0.2845

AUC OvR:
Bajo            : 0.8595
Medio           : 0.7903
Alto            : 1.0000

Baselines:
Persistencia    : 0.7197
Moda histórica  : 0.7132
Mayoritaria     : 0.2609

Referencia:
sklearn RF      : 0.7807

Semana objetivo : 2025-12-29
Pruebas         : 37/37
```

**Horizonte de predicción:** una semana.  
**Unidad espacial:** parroquia.  
**Unidad temporal:** día de la semana.  
**Alcance:** emergencias de Tránsito y Movilidad del cantón Guayaquil.
