# Grupo10_P2 — Clasificación del nivel de riesgo de emergencias de tránsito en el cantón Guayaquil

**CCPG1044 Inteligencia Artificial · ESPOL · Grupo #10 · Paralelo P2**

**Integrantes:**

- Bohórquez Villao Andrés Martín
- Pérez Zamora Juan Sebastián
- Ullaguari Cagua John Jairo

---

## 1. Descripción del proyecto

El proyecto desarrolla un sistema de Machine Learning para clasificar el nivel de riesgo de emergencias de tránsito en el cantón Guayaquil utilizando datos históricos abiertos del Servicio Integrado de Seguridad ECU 911.

La salida del sistema corresponde a tres niveles de riesgo:

- **Bajo**
- **Medio**
- **Alto**

La predicción se realiza para cada combinación de **parroquia × día de la semana** de la semana siguiente.

El clasificador principal es un **Random Forest implementado desde cero en NumPy**, incluyendo:

- árboles CART;
- criterio de impureza Gini ponderada;
- selección aleatoria de atributos;
- bootstrap;
- agregación de probabilidades;
- clasificación multiclase.

`scikit-learn` se utiliza únicamente como referencia experimental y no como parte de la solución principal.

---

## 2. Estado final

**Proyecto implementado, ejecutado y probado con datos reales continuos de julio de 2021 a diciembre de 2025.**

Resultados verificados:

- **35/35 pruebas automáticas superadas**.
- Pipeline ejecutado de punta a punta.
- Notebook ejecutado y guardado con salidas.
- Modelo final entrenado.
- Matriz de riesgo de la semana siguiente generada.
- Interfaz gráfica Streamlit validada.
- Resultados y artefactos almacenados en `salidas/`.

La ejecución final se realizó el **16 de agosto de 2026** sobre:

- Windows 10
- Python 3.13.14
- 4 CPU
- NumPy 2.5.2
- pandas 3.0.5

---

## 3. Representación final del problema

Durante los avances iniciales se asumió que la fuente abierta del ECU 911 contenía coordenadas y hora de cada incidente.

La inspección directa de los archivos mensuales mostró que la publicación disponible contiene siete columnas:

```text
Fecha
provincia
Canton
Cod_Parroquia
Parroquia
Servicio
Subtipo
```

La fuente proporciona ubicación administrativa hasta parroquia y fecha a nivel de día, pero **no contiene coordenadas geográficas ni hora del incidente**.

Por esta razón, la representación inicial fue ajustada a la máxima granularidad soportada por los datos.

| Elemento | Diseño inicial | Representación final |
|---|---|---|
| Unidad espacial | Celda 1 km × 1 km | **Parroquia (`Cod_Parroquia`)** |
| Unidad temporal | Franja de 3 horas, \|T\| = 56 | **Día de la semana, \|T\| = 7** |
| Horizonte | Una semana | **Una semana** |
| Alcance | Cantón Guayaquil, tránsito | **Cantón Guayaquil, tránsito** |

La instancia de aprendizaje final es:

```text
(parroquia, día de la semana, semana)
```

---

## 4. Partición temporal

La evaluación mantiene una separación estrictamente temporal:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | julio 2021 – diciembre 2024 |
| Validación | enero 2025 – junio 2025 |
| Prueba | julio 2025 – diciembre 2025 |

Número de instancias:

```text
Train: 7,224
Validación: 1,092
Test: 1,092
```

No se realiza mezcla aleatoria entre periodos, evitando fuga de información futura.

---

## 5. Preprocesamiento

El módulo `src/preprocesamiento.py`:

1. carga los archivos mensuales del ECU 911;
2. detecta las columnas disponibles;
3. filtra la categoría `Tránsito y Movilidad`;
4. filtra el cantón `GUAYAQUIL`;
5. valida fechas y unidades espaciales;
6. asigna parroquia y día de la semana;
7. agrega los incidentes para formar la serie temporal.

Para reducir el consumo de memoria, cada archivo mensual es **filtrado antes de ser incorporado al conjunto consolidado**.

Esto evita mantener simultáneamente en RAM millones de registros nacionales que no pertenecen al alcance del proyecto.

---

## 6. Características

El modelo utiliza **11 variables predictoras**:

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

Todas las características utilizan exclusivamente información disponible **antes de la semana a predecir**.

### Feriados

`es_feriado` considera:

- feriados nacionales de fecha fija;
- lunes y martes de Carnaval;
- Viernes Santo.

No incorpora traslados extraordinarios de días de descanso establecidos mediante decretos anuales.

### Vecindad espacial

No se utilizan características de vecinos.

La fuente no proporciona coordenadas, por lo que no existe una relación geométrica verificable entre parroquias que permita construir una vecindad espacial sin introducir supuestos artificiales.

---

## 7. Construcción de la etiqueta

La variable objetivo representa el conteo de emergencias de cada combinación parroquia–día.

La clasificación final utiliza:

```text
Bajo  : c <= P60
Medio : P60 < c <= P90
Alto  : c > P90
```

Los percentiles se calculan **únicamente con datos de entrenamiento**.

En la ejecución final se utilizó directamente el modo:

```text
global
```

con:

```text
P60 = 0
P90 = 121
```

Distribución de clases en entrenamiento:

| Clase | Proporción |
|---|---:|
| Bajo | 63.690 % |
| Medio | 26.398 % |
| Alto | 9.911 % |

No fue necesario recurrir a los mecanismos alternativos de etiquetado.

---

## 8. Modelo Random Forest propio

El clasificador fue desarrollado desde cero en `src/bosque.py` utilizando NumPy.

La implementación incluye:

- nodos CART;
- divisiones mediante Gini ponderado;
- selección aleatoria de atributos;
- muestras bootstrap;
- pesos de clase;
- predicción de probabilidades;
- combinación de árboles por promedio de probabilidades.

### Hiperparámetros finales

```text
Número de árboles        : 100
Profundidad máxima       : 16
Mínimo muestras por hoja : 5
Bins                     : 32
Semilla                  : 42
Potencia de pesos        : 1.0
```

Criterio de selección:

```text
F1 macro en validación
```

F1 macro obtenido durante la selección:

```text
0.8268
```

---

## 9. Resultados finales

### Conjunto de prueba

| Modelo | Accuracy | F1 macro | F1 bajo | F1 medio | F1 alto | Recall alto |
|---|---:|---:|---:|---:|---:|---:|
| **Bosque propio (NumPy)** | **0.8104** | **0.7783** | 0.8547 | 0.4912 | **0.9889** | **1.0000** |
| sklearn RandomForest | 0.7985 | 0.7762 | 0.8422 | 0.4977 | 0.9889 | 1.0000 |
| Persistencia | 0.7491 | 0.7141 | 0.8057 | 0.3478 | 0.9889 | 1.0000 |
| Moda histórica | 0.7738 | 0.7096 | 0.8639 | 0.4404 | 0.8247 | 0.7135 |
| Clase mayoritaria | 0.6520 | 0.2631 | 0.7894 | 0.0000 | 0.0000 | 0.0000 |

El modelo propio obtuvo:

```text
Accuracy   = 0.8104
F1 macro   = 0.7783
F1 Alto    = 0.9889
Recall Alto = 1.0000
Brier      = 0.2888
```

El resultado supera la referencia de **F1 macro = 0.60** establecida durante el proyecto y supera los baselines definidos sobre el mismo conjunto de datos.

También presenta un desempeño comparable al Random Forest de referencia de `scikit-learn`, obteniendo ligeramente mayor F1 macro:

```text
Propio  : 0.7783
sklearn : 0.7762
```

---

## 10. Importancia de variables

Las variables con mayor importancia en el bosque propio fueron:

| Variable | Importancia |
|---|---:|
| `rezago_1` | 0.1823 |
| `media_movil_12` | 0.1690 |
| `densidad_historica_parroquia` | 0.1601 |
| `media_movil_4` | 0.1461 |
| `rezago_2` | 0.1315 |

Los resultados indican que el historial reciente y el comportamiento temporal acumulado de cada parroquia aportan más información al clasificador que variables de calendario como feriado o fin de semana.

---

## 11. Desempeño por parroquia

La fuente presenta una fuerte concentración espacial de eventos en la cabecera cantonal de Guayaquil.

Por esta razón, además de las métricas agregadas, se evalúa el rendimiento por parroquia.

| Parroquia | n | Distribución Bajo/Medio/Alto | Accuracy |
|---|---:|---:|---:|
| Guayaquil, cabecera cantonal | 182 | 4 / 0 / 178 | 0.9780 |
| Juan Gómez Rendón | 182 | 86 / 96 / 0 | 0.5110 |
| Morro | 182 | 169 / 13 / 0 | 0.9286 |
| Posorja | 182 | 133 / 49 / 0 | 0.7253 |
| Puná | 182 | 177 / 5 / 0 | 0.9725 |
| Tenguel | 182 | 143 / 39 / 0 | 0.7473 |

La clase **Alto** se concentra principalmente en la cabecera cantonal.

Por ello, el elevado F1 de la clase Alto debe interpretarse junto con la distribución espacial de clases.

La separación entre las clases **Bajo** y **Medio** constituye el problema de clasificación más difícil del conjunto de datos.

---

## 12. Limitaciones de los datos

### Resolución espacial y temporal

La fuente abierta no proporciona:

- coordenadas;
- dirección georreferenciable;
- hora del incidente.

Por ello, no es posible construir honestamente una grilla espacial de 1 km × 1 km ni franjas horarias de tres horas.

### Concentración espacial

La cabecera cantonal concentra aproximadamente el **98.9 % de los eventos de tránsito** observados dentro del cantón.

Esto limita la cantidad de ejemplos de riesgo alto disponibles para otras parroquias.

### Anomalía de enero de 2024

Durante la inspección se detectó una anomalía notable en el recurso correspondiente a enero de 2024.

En Guayaquil se encontraron:

```text
Diciembre 2023 : 4,511 registros de Tránsito y Movilidad
Enero 2024     :    20 registros de Tránsito y Movilidad
Febrero 2024   : 4,041 registros de Tránsito y Movilidad
```

El archivo de enero contiene 208,194 registros nacionales y 40,351 registros de Guayaquil, por lo que no corresponde a un archivo vacío.

A nivel nacional se observaron únicamente 2,634 registros clasificados como `Tránsito y Movilidad`, mientras que 200,318 aparecen como `Seguridad Ciudadana`.

También se verificó que los subtipos de tránsito presentes en febrero no aparecen reclasificados dentro de `Seguridad Ciudadana` en enero.

Al no existir una regla objetiva para reconstruir los registros faltantes, **no se realizaron imputaciones ni reclasificaciones artificiales**. El archivo se conserva tal como fue publicado y la situación se documenta como una limitación de calidad de la fuente.

---

## 13. Semana de predicción

La ejecución final genera la matriz correspondiente a la semana que inicia:

```text
2026-01-05
```

Para las seis parroquias elegibles se producen:

```text
6 parroquias × 7 días = 42 combinaciones
```

La interfaz permite consultar:

- nivel de riesgo;
- probabilidad de la clase predicha;
- parroquia;
- día de la semana;
- distribución de niveles;
- métricas del modelo;
- desempeño por parroquia;
- importancia de variables.

La probabilidad mostrada corresponde a la **probabilidad asignada por el clasificador a la clase predicha** y no a la probabilidad absoluta de que ocurra una emergencia.

---

## 14. Interfaz gráfica

La interfaz se encuentra en:

```text
app.py
```

y fue desarrollada con Streamlit.

Incluye:

- matriz parroquia × día;
- clasificación visual Bajo / Medio / Alto;
- probabilidades;
- filtros por parroquia;
- filtros por día;
- filtros por nivel;
- umbral mínimo de confianza;
- detalle de predicciones;
- comparación contra baselines;
- desempeño por parroquia;
- importancia de características.

Para ejecutarla:

```bash
streamlit run app.py
```

---

## 15. Estructura del proyecto

```text
Grupo10_P2/
│
├── 00_inspeccionar_csv.py
├── proyecto.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
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
│   └── CSV mensuales del ECU 911
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

## 16. Instalación

Se recomienda utilizar un entorno virtual.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las dependencias principales se encuentran en:

```text
requirements.txt
```

---

## 17. Ejecución

### 1. Inspeccionar los CSV

```bash
python 00_inspeccionar_csv.py datos/
```

### 2. Ejecutar las pruebas

```bash
python -m pytest tests/ -q
```

Resultado esperado:

```text
35 passed
```

### 3. Ejecutar el notebook

Abrir:

```text
proyecto.ipynb
```

y ejecutar todas las celdas utilizando el entorno Python del proyecto.

### 4. Ejecutar la interfaz

```bash
streamlit run app.py
```

---

## 18. Pruebas

El proyecto incluye **35 pruebas automáticas**:

```text
7  pruebas del criterio Gini
12 pruebas del pipeline
16 pruebas de representación parroquia × día
```

Para ejecutarlas:

```bash
python -m pytest tests/ -q
```

La corrida final obtuvo:

```text
35 passed
```

---

## 19. Artefactos generados

La ejecución final genera en `salidas/`:

```text
metricas.json
modelo.joblib
matriz_riesgo.parquet
unidades.json
exploratorio.png
convergencia.png
calibracion.png
importancias.png
```

`metricas.json` contiene las métricas finales, hiperparámetros, resultados por unidad, características, tiempos y entorno de ejecución.

---

## 20. Rendimiento computacional

En la ejecución final:

```text
Preprocesamiento              : 251.1 s
Entrenamiento bosque final    : 46.1 s
Memoria pico                  : 561.8 MB
```

La carga de datos fue optimizada mediante prefiltrado mensual para evitar conservar en memoria registros nacionales fuera del alcance del proyecto.

---

## 21. Conclusión

Se implementó satisfactoriamente un clasificador Random Forest propio para estimar el nivel de riesgo de emergencias de tránsito por parroquia y día de la semana en el cantón Guayaquil.

El modelo obtuvo un **F1 macro de 0.7783 en el conjunto de prueba**, superando los baselines del proyecto y la referencia inicial de 0.60.

La comparación con `scikit-learn` muestra que la implementación propia alcanza un desempeño equivalente al algoritmo de referencia.

Los resultados evidencian, sin embargo, una fuerte concentración de eventos en la cabecera cantonal y una mayor dificultad para diferenciar las clases Bajo y Medio en las parroquias con menor cantidad de incidentes.

La solución final incluye:

- procesamiento reproducible;
- Random Forest propio;
- evaluación temporal;
- comparación contra baselines;
- análisis por parroquia;
- matriz de riesgo de la semana siguiente;
- interfaz gráfica;
- pruebas automatizadas.

---

## 22. Fuente de datos

Datos históricos abiertos del:

**Servicio Integrado de Seguridad ECU 911**

Periodo utilizado:

```text
julio 2021 – diciembre 2025
```

Categoría:

```text
Tránsito y Movilidad
```

Alcance:

```text
Cantón Guayaquil
```
