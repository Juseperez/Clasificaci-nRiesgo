# Grupo10_P2 — Clasificación del nivel de riesgo de emergencias de tránsito en el cantón Guayaquil

CCPG1044 Inteligencia Artificial · ESPOL · Grupo #10 · Paralelo P2

Bohórquez Villao Andrés Martín · Pérez Zamora Juan Sebastián · Ullaguari Cagua John Jairo

---

## Estado del proyecto

**Implementación completa y probada; pendiente la corrida final con datos continuos.**

**Verificable desde este paquete**, sin datos ni ejecución previa:

- 35 pruebas automáticas pasan (7 + 12 + 16 en `tests/`).
- Los 12 módulos `.py` compilan y el notebook (33 celdas, 20 de código) tiene sintaxis válida.
- El notebook viene sin salidas guardadas y `salidas/` está vacío: no hay ningún
  resultado que pueda confundirse con una medición real.

**Verificado durante el desarrollo, pero no reproducible desde este paquete**
hasta incorporar los datos continuos y ejecutar el notebook:

- La interfaz Streamlit inicia y renderiza correctamente cuando dispone de
  `salidas/matriz_riesgo` y `salidas/metricas.json`.
- El pipeline completo se ejecuta de punta a punta con CSV reales del ECU 911.
  Esa corrida usó tres meses **no contiguos** y parámetros reducidos, así que
  demuestra que el código funciona, no produce resultados reportables.

**Pendiente:** `datos/` está vacío. No hay ninguna métrica final, ni matriz de
riesgo definitiva, ni notebook ejecutado con salidas.

Ninguna métrica de desempeño de este proyecto es válida hasta ejecutar el
notebook con los meses continuos de julio 2021 a diciembre 2025.

## Re-especificación de la representación

En los avances anteriores se asumió, a partir de la descripción inicial de la
fuente, que los recursos mensuales del ECU 911 incluían coordenadas y hora del
incidente. Durante la implementación se verificó directamente la estructura de
los archivos de distintos años y se determinó que la publicación abierta contiene
exactamente siete columnas:

```
Fecha ; provincia ; Canton ; Cod_Parroquia ; Parroquia ; Servicio ; Subtipo
```

Es decir, ubicación administrativa hasta parroquia y fecha a nivel de día, pero
**no coordenadas ni hora**. La celda de 1 km × 1 km y la franja de tres horas son
inconstruibles con esta fuente. La representación se ajustó a la máxima
granularidad verificable:

| | Tareas #3 y #4 | Versión final |
|---|---|---|
| Unidad espacial | celda de 1 km × 1 km (UTM 17S) | **parroquia** (`Cod_Parroquia`) |
| Unidad temporal | franja de 3 h, \|T\| = 56 | **día de la semana, \|T\| = 7** |
| Horizonte | una semana | una semana (sin cambio) |
| Alcance | cantón Guayaquil, tránsito | cantón Guayaquil, tránsito (sin cambio) |

Todo lo demás se conserva: instancia (parroquia, día, semana), etiqueta por
percentiles ajustados solo con entrenamiento, partición temporal, bosque
aleatorio propio en NumPy, tres componentes y tres casos de uso.

**Limitación que hay que declarar en el reporte:** dentro del cantón Guayaquil el
dataset distingue 6 parroquias, y la cabecera cantonal concentra alrededor del
98.7 % de los eventos de tránsito. La resolución espacial que ofrece la fuente
abierta es por tanto limitada, y las métricas deben reportarse también por
unidad, no solo agregadas. El `Preprocesador` lo advierte automáticamente cuando
una unidad supera el 90 %.

## Estructura

```
00_inspeccionar_csv.py   Paso 0: qué traen realmente los CSV del portal
proyecto.ipynb           Notebook principal (entregable a)
app.py                   Interfaz gráfica Streamlit (entregable c)
src/preprocesamiento.py  Componente 1 — UC1: CSV crudos -> tensor c[g,t,s]
src/caracteristicas.py   Partición temporal, elegibilidad, etiqueta, matriz X
src/bosque.py            Componente 2 — bosque aleatorio propio (NumPy)
src/metricas.py          Métricas y los tres baselines
src/riesgo.py            Componente 3 — matriz de riesgo parroquia × día
tests/test_gini.py       Verificación del criterio de división (7 pruebas)
tests/test_pipeline.py   Regresión del pipeline (12 pruebas)
tests/test_representacion.py  Semántica parroquia × día (16 pruebas)
datos/                   CSV descargados manualmente del portal
salidas/                 Artefactos que consume la interfaz
```

## Cómo correrlo

```bash
pip install numpy pandas matplotlib scikit-learn joblib streamlit
# opcionales: xgboost pyarrow

python 00_inspeccionar_csv.py datos/     # 1. confirmar las columnas reales
python -m tests.test_gini                # 2. criterio de división
python -m tests.test_pipeline            # 3. pipeline
python -m tests.test_representacion      # 4. representación parroquia × día
jupyter notebook proyecto.ipynb          # 5. ejecutar y GUARDAR CON SALIDAS
streamlit run app.py                     # 6. interfaz gráfica
```

No hacen falta pyproj, GeoPandas ni Folium: al no haber coordenadas no hay
geometría que proyectar ni que dibujar.

## El modelo

Bosque aleatorio de árboles de decisión **implementado desde cero en NumPy**
(`src/bosque.py`), siguiendo CART (Breiman et al., 1984) y Random Forests
(Breiman, 2001). NumPy se usa solo como librería de álgebra matricial: la
inducción del árbol, la impureza de Gini ponderada, el muestreo bootstrap y la
agregación de probabilidades están escritos en el proyecto.

`scikit-learn` y `XGBoost` aparecen únicamente en una celda del notebook marcada
como benchmark externo, y no participan de la solución.

Decisiones que mantienen tratable la implementación propia:

- `X` se guarda discretizada en `uint8` (bins por cuantiles ajustados solo con
  filas elegibles de entrenamiento).
- La búsqueda del mejor corte acumula un histograma de peso por (bin, clase) con
  `np.bincount` en lugar de ordenar: O(\|Q\|) por nodo.
- Cada árbol se entrena sobre una submuestra bootstrap acotada por
  `maxMuestrasPorArbol`, **no** sobre el conjunto completo. Hay que reportarlo
  así y no como "N árboles entrenados sobre n instancias".

Los tiempos y la memoria se miden en la corrida real y quedan en
`salidas/metricas.json`, junto con el entorno de ejecución.

## Características (11 variables)

```
dia_semana  es_fin_de_semana  mes  es_feriado
rezago_1  rezago_2  rezago_3  rezago_4
media_movil_4  media_movil_12
densidad_historica_parroquia
```

Todas usan información estrictamente anterior a la semana `s`.

**No hay características de vecindad.** Sin coordenadas no existe una relación de
adyacencia defendible entre parroquias: Puná es una isla y Tenguel está al
extremo sur, y ambas pertenecen al mismo cantón que la cabecera. Se prescinde de
la vecindad en lugar de fabricar una proximidad que no se puede sostener.

`mes` y `es_feriado` se calculan sobre la **fecha exacta** de cada par (día,
semana), no sobre la fecha del lunes replicada a los siete días.

**Redacción exacta para el reporte sobre `es_feriado`:** indicador de feriados
nacionales de fecha fija y de feriados móviles derivados de Pascua (Carnaval
lunes y martes, Viernes Santo); **no incorpora traslados de días de descanso**,
que dependen de decretos anuales. No describirlo como "calendario oficial
completo".

## La etiqueta

```
y = 0 (bajo)  si c <= u_bajo
    1 (medio) si u_bajo < c <= u_alto
    2 (alto)  si c > u_alto
```

Umbrales ajustados **solo con la ventana de entrenamiento**. Como el conteo es
una variable discreta, `ConstructorEtiquetas` recorre una escalera y declara en
qué peldaño se detuvo:

| Peldaño | Qué hace | ¿Sigue siendo un percentil? |
|---|---|---|
| `global` | P60/P90 sobre todos los conteos de entrenamiento | Sí |
| `positivos` | P60/P90 sobre los conteos > 0 | Sí |
| `por_dia` | P60/P90 calculados por día de la semana | Sí |
| `discreto` | cortes enteros forzados sobre la ECDF | **No** |

Criterios de aceptación de un peldaño: las tres clases con masa suficiente y, en
`por_dia`, que ningún día pierda individualmente su clase media (si un día tiene
P60 = P90, dentro de él la clase media desaparece y todo `c ≥ 1` salta a "alto").

**Que la clase alta salga más frecuente que la media NO invalida el etiquetado.**
En una variable discreta puede ocurrir con los percentiles correctos: con 85 % de
ceros, 5 % de unos y 10 % de doses, `np.percentile` da P60 = 0 y P90 = 1.1 y las
clases quedan 85/5/10; "alto" sigue siendo exactamente el extremo superior. Se
reporta como diagnóstico, no como error.

Del mismo modo, `corresponde_a_percentiles` es **por construcción** (`True` para
los tres primeros peldaños, `False` solo para `discreto`) y no se deduce de la
masa acumulada `F(u)`: el cuantil de una discreta es una función escalón, así que
con 85 % de ceros el valor 0 es a la vez el percentil 60, el 70 y el 80.

Si la corrida real llega a `discreto`, el reporte no puede seguir hablando de
"P60 y P90": hay que declarar la regla efectivamente usada y las proporciones
resultantes.

## Desempeño por parroquia

Como la cabecera cantonal concentra alrededor del 98.7 % de los eventos, un buen
F1 macro agregado podría deberse solo a ella. El notebook produce una tabla de
desempeño unidad por unidad (sección 2.6) con el número de instancias, la
distribución de clases y las métricas de cada parroquia, y la guarda en
`salidas/metricas.json` bajo `por_unidad`; la interfaz la muestra en la pestaña
de métricas.

En parroquias con pocas instancias, o donde alguna clase no aparece en el
conjunto de prueba, el F1 macro aislado es inestable y no debe leerse igual que
el de la cabecera. Por eso la tabla incluye `n` y `clases_presentes`, y el
notebook avisa cuando alguna unidad no tiene las tres clases. Esta tabla es la
respuesta directa a la pregunta "¿el buen rendimiento agregado es solo porque
Guayaquil cabecera domina los datos?".

## Elegibilidad

Las unidades sin incidentes en la ventana de entrenamiento **no ingresan al
modelo**: se excluyen de `X_train`, `X_val` y `X_test`, no solo del mapa, y en la
interfaz se muestran en gris como "sin datos suficientes" (riesgo no estimado),
porque la ausencia de registros puede deberse a falta de cobertura y no equivale
a riesgo bajo.

`NIVEL_ELEGIBILIDAD = "unidad"` (por defecto) excluye la parroquia completa; es
la regla literal de la Tarea #4. `"unidad_dia"` es más estricto y elimina también
días tranquilos de parroquias activas, que sí son información válida de riesgo
bajo.

Esto también rige en la **predicción de la semana objetivo**:
`predecir_matriz_semana()` envía al modelo únicamente las filas elegibles y deja
el resto de la matriz en `NaN`. Predecir las G×T combinaciones y pintar después
en gris las no elegibles daría el mismo resultado visual, pero haría falsa la
afirmación de que no pasan por el modelo. Una unidad sin historial no reporta
nivel ni probabilidad: el clasificador no opina sobre ella.

## Salvaguarda de cobertura temporal

Meses no contiguos dejan semanas enteras vacías que contaminan los rezagos, las
medias móviles y los percentiles de la etiqueta. El notebook **se detiene** si
detecta semanas sin ningún evento. Es una salvaguarda de cobertura: dado el
volumen de tránsito en Guayaquil, una semana completa en cero sería
extremadamente sospechosa.

## Inconsistencia del reporte que hay que resolver (no es código)

RNF1 de la Tarea #3 dice que la clasificación **debe alcanzar** un F1 macro
superior al 60 %. La Tarea #4 declara vigentes RF1–RNF4, pero más adelante ya
dice que el 60 % es una meta *inicial de referencia* y que el criterio principal
es superar consistentemente los baselines.

Esa revisión quedó escrita el 27 de julio, **antes de tener ningún resultado
real**: no es un ajuste retroactivo, y hay que decirlo así.

> RNF1 se precisa respecto de la Tarea #3: el F1 macro de 0.60 se conserva como
> referencia aspiracional proveniente de literatura en contextos distintos,
> mientras que el criterio experimental principal de éxito es superar de forma
> consistente los baselines calculados sobre el mismo dataset. Esta precisión
> quedó establecida durante el diseño (Tarea #4), previamente a la obtención de
> resultados.

Si el F1 real queda por debajo de 0.60, **no escribir "cumplimos porque cambiamos
RNF1"**, sino:

> El umbral aspiracional original de 0.60 no se alcanzó; sin embargo, la
> evaluación se realiza también contra los baselines definidos sobre el mismo
> dataset, criterio establecido durante el diseño.

Los requisitos funcionales también se actualizan, no se esconden: RF3 pasa a
asignar cada registro a su parroquia y día de la semana; RF4 construye la
etiqueta por parroquia–día; RF5 predice el nivel de cada parroquia–día de la
semana siguiente; RF6 ofrece filtros por parroquia y día; RF8 exporta las
combinaciones críticas. La rúbrica del proyecto final exige expresamente
"nombre y descripción actualizada del problema" y modelos finales, así que este
es el lugar correcto para hacerlo.

## Correspondencia con los modelos de la Tarea #4

Las clases conservan los nombres del diagrama de clases de diseño:
`Preprocesador`, `ParticionadorTemporal`, `ConstructorEtiquetas`,
`GeneradorCaracteristicas`, `NodoArbol`, `ArbolDecisionPropio`,
`BosqueAleatorioPropio`, `EvaluadorMetricas`, `MatrizRiesgo`. La interfaz de
consulta es `app.py`.

Diagramas que hay que actualizar en el reporte final: la representación
computacional (desaparecen UTM y grilla, aparece `Cod_Parroquia`) y el diagrama
de clases del dominio (`CeldaGeografica` → `Parroquia`, `FranjaHoraria` →
`DiaSemana`). Casos de uso, escenarios, paquetes, componentes, algoritmo del
bosque, secuencia y estados quedan prácticamente iguales: cambian etiquetas, no
estructura.

## Plan restante

1. Descargar los meses **contiguos** de julio 2021 a diciembre 2025.
2. Correr `00_inspeccionar_csv.py` y confirmar que las columnas no cambian en
   ningún mes del periodo.
3. Ejecutar el notebook completo **y guardarlo con las salidas**.
4. `streamlit run app.py` y capturas para el reporte.
5. Escribir las secciones 6, 7 y 8 con los números de `salidas/metricas.json`.
6. Póster con el template de Canvas y armar `Grupo10_P2.zip`.

Partición temporal según lo fijado en la Tarea #3, sin cambios: entrenamiento
julio 2021 – diciembre 2024, validación enero – junio 2025, prueba julio –
diciembre 2025. Un notebook con `execution_count = None` no acredita ninguna
medición.
