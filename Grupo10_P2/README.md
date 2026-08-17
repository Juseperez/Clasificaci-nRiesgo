# Clasificación del nivel de riesgo de emergencias de tránsito en el cantón Guayaquil

**CCPG1044 - Inteligencia Artificial**  
**Escuela Superior Politécnica del Litoral (ESPOL)**  
**Grupo #10 - Paralelo P2**

## Integrantes

- Bohórquez Villao Andrés Martín
- Pérez Zamora Juan Sebastián
- Ullaguari Cagua John Jairo

## Descripción

Sistema de Machine Learning para clasificar el nivel de riesgo de emergencias de **Tránsito y Movilidad** en el cantón Guayaquil por combinación **parroquia × día de la semana**, con horizonte de predicción de una semana.

El clasificador principal es un **Random Forest implementado desde cero en NumPy**. `scikit-learn` se utiliza únicamente como benchmark externo y no participa en la solución principal.

Repositorio del proyecto:

https://github.com/Juseperez/Clasificaci-nRiesgo

## Datos y periodo experimental

La corrida final utiliza **54 archivos mensuales consecutivos**, desde **julio de 2021 hasta diciembre de 2025**.

El portal del ECU 911 dispone actualmente de archivos posteriores a diciembre de 2025; esos archivos **no forman parte del periodo experimental definido para esta ejecución** y no se utilizaron para recalcular las métricas finales.

El paquete contiene:

```text
datos/
```

- 54 CSV utilizados para reproducir íntegramente la corrida final.

```text
datos_ejemplo/
```

- 10 CSV consecutivos de marzo a diciembre de 2025 como muestra de la estructura real de entrada y para demostraciones/inspección.

**Importante:** `datos_ejemplo/` no reproduce por sí solo las métricas finales; la reproducción completa requiere `datos/`.

## Representación final

La fuente utilizada contiene siete campos:

```text
Fecha
provincia
Canton
Cod_Parroquia
Parroquia
Servicio
Subtipo
```

No contiene coordenadas geográficas ni hora del incidente. Por ello:

```text
unidad espacial : parroquia (Cod_Parroquia)
unidad temporal : día de la semana
instancia        : (parroquia, día, semana)
horizonte        : una semana
```

No se fabrican coordenadas, franjas horarias ni relaciones de vecindad.

## Conjunto procesado

```text
15,510,884 filas nacionales
268,049 registros de Tránsito y Movilidad en Guayaquil
267,310 eventos en el tensor final
6 unidades espaciales presentes en el conjunto filtrado
7 días de la semana
235 semanas representadas
```

La semana iniciada el **29/12/2025** cruza el límite del periodo experimental. Dentro del corte definido solo están presentes 29, 30 y 31 de diciembre; para evitar interpretar artificialmente los cuatro días externos al corte como conteos cero, esa semana se excluye íntegramente del tensor observado y se utiliza como horizonte de clasificación.

La primera semana representada comienza el 28/06/2021 y es parcial porque el periodo experimental empieza el 01/07/2021. Permanece únicamente dentro del `BURN_IN` y no se utiliza como objetivo supervisado.

## Partición temporal

| Conjunto | Semanas utilizadas | Semanas | Instancias |
|---|---|---:|---:|
| Entrenamiento | 20/09/2021 - 30/12/2024 | 172 | 7,224 |
| Validación | 06/01/2025 - 30/06/2025 | 26 | 1,092 |
| Prueba | 07/07/2025 - 22/12/2025 | 25 | 1,050 |

Las primeras **12 semanas** son el periodo de inicialización (`BURN_IN = 12`).

## Características

El modelo utiliza **11 características**, todas construidas con información anterior a la semana objetivo:

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

`Cod_Parroquia` identifica la unidad espacial, pero no ingresa al bosque como variable numérica u ordinal.

## Etiqueta de riesgo

Umbrales ajustados exclusivamente con entrenamiento:

```text
P60 = 0
P90 = 121

Bajo  : c <= 0
Medio : 1 <= c <= 121
Alto  : c > 121
```

Distribución en entrenamiento:

```text
Bajo  : 63.690 %
Medio : 26.398 %
Alto  : 9.911 %
```

`P60 = 0` es consecuencia de la distribución discreta del conteo y de la concentración de ceros; no es un error de cálculo.

## Random Forest propio

Configuración definitiva:

```text
nArboles            = 100
profundidadMax       = 16
minMuestrasHoja      = 5
mAtributos           = floor(sqrt(11)) = 3
maxMuestrasPorArbol  = 200000
nBins                = 32
semilla              = 42
potenciaPesos        = 1.0
```

La implementación incluye CART, Gini ponderado por clase, bootstrap, selección aleatoria de atributos por nodo, probabilidades por hoja, agregación de probabilidades e importancia interna por reducción de impureza.

## Resultados finales

Conjunto de prueba: **1,050 instancias**.

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Random Forest propio | **0.8114** | **0.7822** |
| Random Forest sklearn | 0.8000 | 0.7807 |
| Persistencia | 0.7533 | 0.7197 |
| Moda histórica | 0.7743 | 0.7132 |
| Clase mayoritaria | 0.6429 | 0.2609 |

```text
F1 Bajo     = 0.8544
F1 Medio    = 0.4923
F1 Alto     = 1.0000
Recall Alto = 1.0000
Brier       = 0.2845

AUC Bajo    = 0.8595
AUC Medio   = 0.7903
AUC Alto    = 1.0000
```

La clase Alto debe interpretarse con cautela: sus 175 instancias de prueba pertenecen a la cabecera cantonal y el baseline de persistencia también obtiene F1 Alto = 1.0000. La principal dificultad predictiva permanece en la clase Medio.

## Estructura del proyecto

```text
Grupo10_P2/
│
├── README.md
├── requirements.txt
├── .gitignore
├── 00_inspeccionar_csv.py
├── proyecto.ipynb
├── app.py
├── reporte_final_v6.pdf
├── PosterGrupo10.pptx
│
├── src/
├── tests/
├── datos/          # 54 CSV, julio 2021 - diciembre 2025
├── datos_ejemplo/  # 10 CSV, marzo - diciembre 2025
└── salidas/
```

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Corrida definitiva:

```text
Windows 10
Python 3.13.14
NumPy 2.5.2
pandas 3.0.5
4 CPU
```

También se verificó en Windows 11 / Python 3.14.3 / 8 CPU. Las métricas principales y los umbrales se reprodujeron; los tiempos y la memoria variaron según el entorno.

## Demo rápida en clase

Pruebas automáticas:

```powershell
python -m pytest tests/ -q
```

Resultado esperado:

```text
............................................ [100%]
44 passed
```

Interfaz gráfica:

```powershell
streamlit run app.py
```

La interfaz consume los artefactos finales de `salidas/` y **no necesita reentrenar el bosque para mostrar la matriz y los resultados finales**.

## Reproducción completa del experimento

1. Verificar que `datos/` contenga los 54 archivos mensuales.
2. Instalar `requirements.txt`.
3. Abrir `proyecto.ipynb`.
4. Ejecutar todas las celdas en orden.

El notebook realiza:

```text
carga
→ preprocesamiento
→ representación
→ partición temporal
→ construcción de etiquetas
→ generación de características
→ búsqueda de hiperparámetros
→ entrenamiento definitivo
→ validación
→ prueba
→ benchmarks
→ generación de artefactos
```

La ejecución completa puede tomar varios minutos dependiendo del equipo.

## Inspección de los datos de ejemplo

```powershell
python 00_inspeccionar_csv.py datos_ejemplo/
```

Conjunto completo:

```powershell
python 00_inspeccionar_csv.py datos/
```

## Pruebas automáticas

La versión final contiene **44 pruebas**:

```text
8  - Gini y bosque
16 - pipeline y control de fuga temporal
18 - representación parroquia × día
2  - métricas
```

Todas pasan en la versión entregada.

## Artefactos finales

`salidas/` contiene:

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

La probabilidad mostrada en la interfaz corresponde a la **confianza del clasificador en la clase predicha**, no a la probabilidad absoluta de que ocurra una emergencia.

## Limitaciones principales

- La fuente abierta no contiene coordenadas ni hora del incidente.
- La cabecera cantonal concentra aproximadamente el 98.9 % de los eventos filtrados.
- Ninguna unidad del conjunto de prueba presenta las tres clases simultáneamente.
- Enero de 2024 muestra una anomalía fuerte en `Tránsito y Movilidad`; se mantiene tal como fue publicada, sin imputar ni reclasificar sin evidencia.
- El indicador de feriado no modela traslados extraordinarios definidos por decretos anuales.
- El sistema debe interpretarse como una prueba de concepto reproducible con datos abiertos, no como una herramienta operativa institucional definitiva.

## Documentación final

- `reporte_final_v6.pdf` - reporte final del proyecto.
- `PosterGrupo10.pptx` - póster del grupo.
- `proyecto.ipynb` - notebook ejecutado de la corrida final.
