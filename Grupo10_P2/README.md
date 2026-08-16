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

La ejecución final verificó:

- **35/35 pruebas automáticas superadas**;
- preprocesamiento completo de los datos del ECU 911;
- construcción de la representación temporal;
- entrenamiento del Random Forest propio;
- ajuste mediante conjunto de validación;
- evaluación final sobre un conjunto de prueba temporal independiente;
- comparación contra baselines;
- comparación contra Random Forest de `scikit-learn`;
- generación de probabilidades por clase;
- análisis de importancia de variables;
- evaluación por parroquia;
- generación de la matriz de riesgo de la semana siguiente;
- interfaz gráfica desarrollada con Streamlit;
- generación y almacenamiento de los artefactos finales.

La ejecución final se realizó el **16 de agosto de 2026**.

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

Periodo utilizado:

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

El preprocesamiento final consolidó aproximadamente:

```text
268,049 registros de Tránsito y Movilidad
del cantón Guayaquil
```

---

## 5. Partición temporal

Para evitar fuga de información entre pasado y futuro se utiliza una **partición estrictamente temporal**.

| Conjunto | Periodo |
|---|---|
| Entrenamiento | julio 2021 – diciembre 2024 |
| Validación | enero 2025 – junio 2025 |
| Prueba | julio 2025 – diciembre 2025 |

Instancias utilizadas:

```text
Entrenamiento : 7,224
Validación    : 1,092
Prueba        : 1,092
```

El conjunto de prueba se utiliza únicamente para la evaluación final.

No se realiza una división aleatoria entre semanas.

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
9. agregación temporal de incidentes.

### Optimización de memoria

Los archivos originales contienen registros de todo Ecuador.

Para evitar conservar simultáneamente millones de registros nacionales en memoria, cada archivo mensual se filtra **antes de incorporarse al conjunto consolidado**.

El flujo es:

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

Esta optimización permitió ejecutar el pipeline completo en un equipo de recursos limitados sin modificar la lógica del modelo.

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

La selección aleatoria de atributos se realiza automáticamente siguiendo la regla de aproximadamente:

```text
sqrt(d)
```

atributos candidatos por nodo.

El criterio utilizado para seleccionar la configuración fue:

```text
F1 macro sobre el conjunto de validación
```

El modelo final obtuvo en validación:

```text
F1 macro = 0.8268
```

---

## 10. Evaluación

La métrica principal es:

```text
F1 macro
```

También se evalúan:

- exactitud;
- precisión por clase;
- recall por clase;
- F1 por clase;
- recall de la clase Alto;
- AUC-ROC OvR;
- Brier Score;
- comportamiento por parroquia.

La clase **Alto** recibe especial atención debido al costo potencial de no identificar correctamente un periodo de mayor riesgo.

---

## 11. Resultados finales

### Comparación sobre el conjunto de prueba

| Modelo | Accuracy | F1 macro | F1 Bajo | F1 Medio | F1 Alto | Recall Alto |
|---|---:|---:|---:|---:|---:|---:|
| **Bosque propio (NumPy)** | **0.8104** | **0.7783** | **0.8547** | 0.4912 | **0.9889** | **1.0000** |
| sklearn RandomForest | 0.7985 | 0.7762 | 0.8422 | **0.4977** | 0.9889 | 1.0000 |
| Persistencia | 0.7491 | 0.7141 | 0.8057 | 0.3478 | 0.9889 | 1.0000 |
| Moda histórica | 0.7738 | 0.7096 | 0.8639 | 0.4404 | 0.8247 | 0.7135 |
| Clase mayoritaria | 0.6520 | 0.2631 | 0.7894 | 0.0000 | 0.0000 | 0.0000 |

Resultados principales del modelo propio:

```text
Accuracy     = 0.8104
F1 macro     = 0.7783
F1 Bajo      = 0.8547
F1 Medio     = 0.4912
F1 Alto      = 0.9889
Recall Alto  = 1.0000
Brier Score  = 0.2888
```

El modelo supera la referencia inicial de:

```text
F1 macro = 0.60
```

y también supera los baselines definidos sobre el mismo conjunto de prueba.

---

## 12. Comparación con scikit-learn

El Random Forest propio obtuvo:

```text
F1 macro propio   = 0.7783
F1 macro sklearn  = 0.7762
```

La diferencia es:

```text
+0.0021
```

a favor de la implementación propia en esta ejecución.

Esto muestra que el bosque desarrollado desde cero alcanza un comportamiento comparable al Random Forest utilizado como referencia.

`scikit-learn` **no participa en las predicciones finales de la solución**; se emplea únicamente para contrastar experimentalmente el funcionamiento del algoritmo propio.

### XGBoost

El notebook conserva XGBoost como benchmark externo opcional.

No forma parte de la solución principal y **no se incluye en las métricas de la corrida final reportada**.

La comparación reproducida en esta ejecución corresponde al Random Forest de `scikit-learn`.

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

Según la importancia por reducción de impureza del bosque propio, los **rezagos, medias móviles y la densidad histórica de la parroquia** tuvieron mayor peso en las decisiones del modelo que las variables de calendario como feriado o fin de semana.

Estas importancias describen el comportamiento interno del clasificador y no deben interpretarse como relaciones causales.

---

## 14. Desempeño por parroquia

La distribución de los registros no es espacialmente uniforme.

La cabecera cantonal concentra la gran mayoría de las emergencias registradas, por lo que se evalúa también el desempeño de manera desagregada.

| Parroquia | n | Bajo / Medio / Alto | Accuracy |
|---|---:|---:|---:|
| Guayaquil, cabecera cantonal | 182 | 4 / 0 / 178 | 0.9780 |
| Juan Gómez Rendón (Progreso) | 182 | 86 / 96 / 0 | 0.5110 |
| Morro | 182 | 169 / 13 / 0 | 0.9286 |
| Posorja | 182 | 133 / 49 / 0 | 0.7253 |
| Puná | 182 | 177 / 5 / 0 | 0.9725 |
| Tenguel | 182 | 143 / 39 / 0 | 0.7473 |

La clase **Alto** aparece fuertemente concentrada en la cabecera cantonal.

Por esta razón, el alto desempeño de esa clase debe interpretarse junto con la distribución espacial de los datos.

La diferenciación entre las clases **Bajo** y **Medio** constituye el problema más difícil para el clasificador.

---

## 15. Limitaciones

### 15.1 Resolución espacial

La fuente utilizada no contiene coordenadas.

Por ello no es posible construir de manera verificable:

```text
celdas de 1 km × 1 km
```

La máxima resolución espacial disponible es la **parroquia**.

### 15.2 Resolución temporal

La fuente únicamente incluye la fecha del incidente.

No contiene hora.

Por ello no es posible construir de manera verificable:

```text
franjas horarias de 3 horas
```

La representación temporal final utiliza el **día de la semana**.

### 15.3 Concentración espacial

La cabecera cantonal concentra aproximadamente el **98.9 % de los eventos de tránsito observados** en el cantón.

Esto limita la cantidad de ejemplos disponibles para las parroquias con menor volumen de incidentes y debe considerarse al interpretar las métricas agregadas.

### 15.4 Anomalía de enero de 2024

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

por lo que no corresponde a un archivo vacío o incompleto en términos de número total de registros.

A nivel nacional se observaron:

```text
Seguridad Ciudadana      : 200,318
Tránsito y Movilidad     :   2,634
```

También se verificó que los subtipos de tránsito observados en febrero no aparecen reclasificados directamente dentro de `Seguridad Ciudadana` en enero para Guayaquil.

No se identificó una regla objetiva y verificable que permitiera corregir o reclasificar los registros de enero de 2024.

Por ello, **no se realizaron imputaciones ni reclasificaciones artificiales**: el archivo se conservó tal como fue publicado y la anomalía se documenta como una limitación de calidad de la fuente.

---

## 16. Predicción de la semana siguiente

La ejecución final genera una matriz de riesgo correspondiente a la semana que inicia:

```text
2026-01-05
```

Las seis parroquias elegibles producen:

```text
6 parroquias × 7 días
=
42 combinaciones
```

En la ejecución final se obtuvieron:

```text
Alto  : 7
Medio : 7
Bajo  : 28
```

La matriz permite consultar tanto el nivel de riesgo como la probabilidad asignada por el clasificador.

La probabilidad mostrada corresponde a:

> la probabilidad estimada de la **clase predicha** por el modelo.

No debe interpretarse como la probabilidad absoluta de que ocurra una emergencia.

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

La evaluación utiliza tres referencias simples sobre exactamente el mismo conjunto de prueba.

### Clase mayoritaria

Predice siempre la clase más frecuente.

```text
F1 macro = 0.2631
```

### Persistencia

Utiliza como predicción la clase observada en el periodo inmediatamente anterior.

```text
F1 macro = 0.7141
```

### Moda histórica

Utiliza la clase históricamente más frecuente para cada unidad.

```text
F1 macro = 0.7096
```

El modelo propio supera los tres baselines.

---

## 20. Rendimiento computacional

La ejecución final registró aproximadamente:

```text
Preprocesamiento            : 251.1 s
Entrenamiento bosque final  : 46.1 s
Memoria pico                : 561.8 MB
```

El prefiltrado mensual de los archivos permitió controlar el consumo de memoria durante el procesamiento de los datos nacionales.

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
35 passed
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

El proyecto contiene **35 pruebas automáticas**:

```text
7  pruebas del criterio Gini
12 pruebas del pipeline
16 pruebas de representación parroquia × día
```

Para ejecutarlas:

```bash
python -m pytest tests/ -q
```

Resultado final:

```text
................................... [100%]
35 passed
```

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

---

## 26. Conclusiones

Se implementó satisfactoriamente un **Random Forest propio** para clasificar el nivel de riesgo de emergencias de tránsito por parroquia y día de la semana en el cantón Guayaquil.

El modelo obtuvo en el conjunto de prueba:

```text
F1 macro = 0.7783
```

superando:

```text
Persistencia      = 0.7141
Moda histórica    = 0.7096
Clase mayoritaria = 0.2631
```

y alcanzando un desempeño comparable al Random Forest de `scikit-learn`:

```text
Bosque propio = 0.7783
sklearn       = 0.7762
```

El modelo mostró un desempeño especialmente alto para la clase **Alto**, aunque este resultado debe interpretarse considerando la fuerte concentración de eventos en la cabecera cantonal.

La clase **Medio** resultó ser la más difícil de discriminar, mostrando el principal espacio de mejora de la solución.

La ejecución también confirmó que el comportamiento histórico reciente —particularmente rezagos, medias móviles y densidad histórica— tiene un papel relevante dentro de las decisiones del bosque.

Finalmente, el proyecto integra en una única solución:

- procesamiento reproducible de datos reales;
- representación temporal sin fuga de información;
- Random Forest implementado desde cero;
- validación temporal;
- baselines;
- benchmark externo;
- métricas por clase;
- evaluación por parroquia;
- probabilidades;
- predicción de la semana siguiente;
- interfaz gráfica;
- pruebas automatizadas.

---

## 27. Resultado principal

```text
Modelo          : Random Forest propio en NumPy
F1 macro test   : 0.7783
Accuracy test   : 0.8104
Recall Alto     : 1.0000
F1 Alto         : 0.9889
F1 Medio        : 0.4912
F1 Bajo         : 0.8547

Baselines:
Persistencia    : 0.7141
Moda histórica  : 0.7096
Mayoritaria     : 0.2631

Referencia:
sklearn RF      : 0.7762
```

**Horizonte de predicción:** una semana.  
**Unidad espacial:** parroquia.  
**Unidad temporal:** día de la semana.  
**Alcance:** emergencias de Tránsito y Movilidad del cantón Guayaquil.
