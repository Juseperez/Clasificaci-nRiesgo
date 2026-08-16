"""
Componente 3 - InterfazConsulta (UC3).

Interfaz grafica propia. No es consola ni API: se ejecuta con

    streamlit run app.py

y lee los artefactos que dejo el notebook en salidas/. No reentrena nada.

El dataset abierto del ECU 911 no publica coordenadas, asi que no hay geometria
que dibujar: el mapa de calor es la matriz parroquia x dia de la semana, que es
exactamente la representacion sobre la que trabaja el modelo.

UC3 escenario principal   : matriz de riesgo con nivel y probabilidad, filtros
                            por parroquia y dia, exportacion del reporte.
UC3 escenario alternativo : parroquia sin incidentes historicos, que se muestra
                            en gris como "sin datos suficientes" sin pasar por
                            el modelo.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de emergencias de transito - Guayaquil",
                   layout="wide", initial_sidebar_state="expanded")

RUTA = "salidas"
COLORES = {"Alto": "#d7191c", "Medio": "#fdae61", "Bajo": "#2c7fb8",
           "Sin datos suficientes": "#bdbdbd"}
TEXTO = {"Alto": "#ffffff", "Medio": "#000000", "Bajo": "#ffffff",
         "Sin datos suficientes": "#666666"}
DIAS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]


@st.cache_data
def cargar():
    df = None
    for ext, lector in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = os.path.join(RUTA, "matriz_riesgo" + ext)
        if os.path.exists(p):
            df = lector(p)
            break
    met = None
    p = os.path.join(RUTA, "metricas.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            met = json.load(f)
    return df, met


def _texto_unidad(met):
    """La unidad se lee de los artefactos; no se escribe fija en la interfaz."""
    r = (met or {}).get("representacion", {})
    esp = r.get("unidad_espacial", "unidad administrativa")
    tmp = r.get("unidad_temporal", "dia de la semana")
    return f"{esp} x {tmp}"


df, met = cargar()

st.title("Clasificacion del nivel de riesgo de emergencias de transito")
st.caption(f"Canton Guayaquil - {_texto_unidad(met)} - datos historicos del "
           "SIS ECU 911. Grupo #10, CCPG1044.")

if df is None:
    st.error("No se encontro `salidas/matriz_riesgo.parquet` ni "
             "`salidas/matriz_riesgo.csv`. Corre primero el notebook "
             "`proyecto.ipynb` para generar la matriz de riesgo.")
    st.stop()

# ---------------------------------------------------------------- filtros ---
st.sidebar.header("Filtros")
unidades = sorted(df["unidad"].unique().tolist())
sel_u = st.sidebar.multiselect("Parroquia", unidades, default=unidades)
sel_d = st.sidebar.multiselect("Dia de la semana", DIAS, default=DIAS)
sel_n = st.sidebar.multiselect(
    "Nivel de riesgo", ["Alto", "Medio", "Bajo", "Sin datos suficientes"],
    default=["Alto", "Medio", "Bajo", "Sin datos suficientes"])
prob_min = st.sidebar.slider("Probabilidad minima de la clase predicha",
                             0.0, 1.0, 0.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "La probabilidad mostrada es **la probabilidad de la clase predicha**, es "
    "decir la confianza del clasificador en el nivel asignado. No es la "
    "probabilidad de que ocurra una emergencia.")
st.sidebar.warning(
    "Las parroquias en gris no tienen incidentes historicos en la ventana de "
    "entrenamiento. No pasan por el modelo y su riesgo **no esta estimado**: la "
    "ausencia de registros puede deberse a falta de cobertura y no equivale a "
    "riesgo bajo.")
st.sidebar.caption(
    "El dataset abierto del ECU 911 publica la ubicacion hasta parroquia y la "
    "fecha a nivel de dia, sin coordenadas ni hora. Por eso la unidad de "
    "analisis es la parroquia y el dia de la semana, y no una celda geografica "
    "ni una franja horaria.")

d = df[df["unidad"].isin(sel_u) & df["dia_semana"].isin(sel_d) & df["nivel"].isin(sel_n)]
d = d[(d["prob_clase_predicha"].fillna(0) >= prob_min) |
      (d["nivel"] == "Sin datos suficientes")]

# --------------------------------------------------------------- cabecera ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Semana objetivo", str(df["semana_objetivo"].iloc[0]))
c2.metric("Combinaciones mostradas", f"{len(d):,}")
c3.metric("Riesgo alto", f"{int((d['nivel'] == 'Alto').sum()):,}")
prom = d.loc[d["nivel"] != "Sin datos suficientes", "prob_clase_predicha"].mean()
c4.metric("Probabilidad media", "n/d" if pd.isna(prom) else f"{prom:.2f}")

tab_mapa, tab_tabla, tab_metricas = st.tabs(
    ["Mapa de riesgo", "Detalle", "Desempeno del modelo"])

# ------------------------------------------------------- matriz de riesgo ---
with tab_mapa:
    if len(d) == 0:
        st.warning("Ningun resultado con los filtros actuales.")
    else:
        st.subheader("Nivel de riesgo por parroquia y dia de la semana")
        dias_vis = [x for x in DIAS if x in sel_d]
        niveles = d.pivot_table(index="unidad", columns="dia_semana",
                                values="nivel", aggfunc="first")
        probas = d.pivot_table(index="unidad", columns="dia_semana",
                               values="prob_clase_predicha", aggfunc="first")
        niveles = niveles.reindex(columns=dias_vis)
        probas = probas.reindex(columns=dias_vis)
        orden = {"Alto": 0, "Medio": 1, "Bajo": 2, "Sin datos suficientes": 3}
        niveles = niveles.loc[
            niveles.apply(lambda r: min(orden.get(v, 9) for v in r.dropna())
                          if r.notna().any() else 9, axis=1).sort_values().index]
        probas = probas.reindex(index=niveles.index)

        matriz = niveles.copy()
        for i in niveles.index:
            for j in niveles.columns:
                n = niveles.loc[i, j]
                if pd.isna(n):
                    matriz.loc[i, j] = ""
                elif n == "Sin datos suficientes":
                    matriz.loc[i, j] = "sin datos"
                else:
                    p = probas.loc[i, j]
                    matriz.loc[i, j] = f"{n}  ({p:.2f})" if pd.notna(p) else str(n)

        def pintar(_):
            est = niveles.copy()
            for i in niveles.index:
                for j in niveles.columns:
                    n = niveles.loc[i, j]
                    if pd.isna(n):
                        est.loc[i, j] = ""
                    else:
                        est.loc[i, j] = (f"background-color: {COLORES[n]}; "
                                         f"color: {TEXTO[n]}; text-align: center")
            return est

        st.dataframe(matriz.style.apply(pintar, axis=None),
                     use_container_width=True)
        st.caption("Cada casilla de la matriz muestra el nivel y, entre parentesis, "
                   "la probabilidad de la clase predicha. Las casillas grises no "
                   "pasaron por el modelo.")

        st.subheader("Distribucion de niveles")
        conteo = (d["nivel"].value_counts()
                  .reindex(["Alto", "Medio", "Bajo", "Sin datos suficientes"])
                  .dropna())
        st.bar_chart(conteo)

# ------------------------------------------------------------------ tabla ---
with tab_tabla:
    cols = ["unidad", "canton", "dia_semana", "nivel", "prob_clase_predicha",
            "p_bajo", "p_medio", "p_alto"]
    cols = [c for c in cols if c in d.columns]
    vista = d[cols].sort_values(["nivel", "prob_clase_predicha"],
                                ascending=[True, False])
    st.dataframe(vista, use_container_width=True, height=460)
    st.download_button(
        "Exportar reporte de parroquias y dias criticos (CSV)",
        vista.to_csv(index=False).encode("utf-8"),
        file_name=f"reporte_riesgo_{df['semana_objetivo'].iloc[0]}.csv",
        mime="text/csv")

# --------------------------------------------------------------- metricas ---
with tab_metricas:
    if met is None:
        st.info("Corre el notebook para generar `salidas/metricas.json`.")
    else:
        if "representacion" in met:
            st.subheader("Representacion del problema")
            r = met["representacion"]
            a, b, cc = st.columns(3)
            a.metric("Unidad espacial", str(r.get("unidad_espacial", "n/d")))
            b.metric("Unidad temporal", str(r.get("unidad_temporal", "n/d")))
            cc.metric("|T|", str(r.get("T", "n/d")))
            if r.get("nota"):
                st.info(r["nota"])

        if "etiqueta" in met:
            st.subheader("Regla de la etiqueta efectivamente usada")
            e = met["etiqueta"]
            ub, ua = e.get("umbral_bajo", "n/d"), e.get("umbral_alto", "n/d")
            vec = isinstance(ub, list) or isinstance(ua, list)
            a, b, cc = st.columns(3)
            a.metric("Modo adoptado", str(e.get("modo_usado", "n/d")))
            b.metric("Umbral bajo",
                     f"{len(ub)} umbrales por dia" if isinstance(ub, list) else f"<= {ub}")
            cc.metric("Umbral alto",
                      f"{len(ua)} umbrales por dia" if isinstance(ua, list) else f"> {ua}")
            if vec:
                with st.expander("Ver los umbrales de cada dia"):
                    n = len(ub if isinstance(ub, list) else ua)
                    st.dataframe(pd.DataFrame({
                        "dia": [DIAS[k % 7] for k in range(n)],
                        "umbral bajo (<=)": ub if isinstance(ub, list) else [ub] * n,
                        "umbral alto (>)": ua if isinstance(ua, list) else [ua] * n,
                    }), use_container_width=True)
            if e.get("corresponde_a_percentiles_solicitados") is False:
                st.warning(
                    "Los umbrales no son los percentiles 60/90: son cortes enteros "
                    "adoptados porque los peldanos anteriores degeneraban. La "
                    "definicion de la etiqueta difiere de la propuesta original.")
            elif e.get("base_de_calculo") not in (None, "todos"):
                st.info(f"Percentiles 60/90 calculados sobre la base "
                        f"'{e['base_de_calculo']}', no sobre la distribucion completa.")
            if e.get("proporciones_pct"):
                st.caption(f"Proporciones de clase (bajo / medio / alto): "
                           f"{e['proporciones_pct']} %")

        if "comparativa" in met:
            st.subheader("Modelo propio frente a los baselines (conjunto de prueba)")
            st.dataframe(pd.DataFrame(met["comparativa"]), use_container_width=True)

        if "por_unidad" in met:
            st.subheader("Desempeno por parroquia")
            st.caption("La cabecera cantonal concentra la mayor parte de los "
                       "eventos: esta tabla muestra si el desempeno agregado se "
                       "sostiene unidad por unidad. Leer el F1 macro junto con n "
                       "y la distribucion de clases.")
            st.dataframe(pd.DataFrame(met["por_unidad"]), use_container_width=True)

        if "importancias" in met:
            st.subheader("Importancia de variables del bosque propio")
            imp = pd.DataFrame(met["importancias"]).sort_values(
                "importancia", ascending=False)
            st.bar_chart(imp.set_index("variable"))

        for clave, titulo in (("alcance", "Alcance"),
                              ("hiperparametros", "Hiperparametros"),
                              ("entorno", "Entorno de ejecucion")):
            if clave in met:
                st.subheader(titulo)
                st.json(met[clave])
