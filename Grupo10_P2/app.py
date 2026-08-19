"""
Componente 3 - InterfazConsulta (UC3) + Componente 4 - Inferencia (UC4).

Interfaz grafica propia. Se ejecuta con

    streamlit run app.py

Lee los artefactos oficiales de salidas/ (matriz_riesgo.parquet, metricas.json,
los PNG del notebook) para la corrida OFICIAL, y usa src/inferencia.py para
generar predicciones nuevas a partir de un CSV mensual subido por el usuario,
con el bosque YA ENTRENADO. No reentrena nada, no importa sklearn, no escribe
nunca modelo.joblib / metricas.json / matriz_riesgo.parquet.

Cinco areas: Resumen, Nueva prediccion, Matriz y detalle, Desempeno,
Metodologia. El tema visual (fondo navy, tipografia, radios) vive en
.streamlit/config.toml; el CSS de aqui solo agrega movimiento contenido y la
firma tipografica (lecturas numericas en monoespaciada tabular). Donde hace
falta color explicito fuera del tema (matriz de riesgo, resaltados de tabla,
chips de estado) se fija fondo+texto juntos para que sea legible sin importar
el tema del usuario.
"""
import json
import os
import tempfile

import pandas as pd
import streamlit as st

import src.inferencia as inf

st.set_page_config(page_title="Riesgo de transito - Guayaquil",
                   layout="wide", initial_sidebar_state="expanded",
                   page_icon="🚦")

RUTA = "salidas"
COLORES = {"Alto": "#d7191c", "Medio": "#fdae61", "Bajo": "#2c7fb8",
           "Sin datos suficientes": "#bdbdbd"}
TEXTO = {"Alto": "#ffffff", "Medio": "#000000", "Bajo": "#ffffff",
         "Sin datos suficientes": "#666666"}
ESTADO = {"ok": "#1e7d4f", "error": "#a4292c"}
ESTADO_TEXTO = {"ok": "#ffffff", "error": "#ffffff"}

TEMA = {
    "superficie": "#12182A",
    "linea": "#232B45",
    "acento": "#C9A24B",
    "mono": "Consolas, 'Cascadia Mono', 'Courier New', monospace",
}
DIAS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
NIVELES_ORDEN = ["Alto", "Medio", "Bajo", "Sin datos suficientes"]
ORDEN_NIVEL = {"Alto": 0, "Medio": 1, "Bajo": 2, "Sin datos suficientes": 3}

GRAFICOS_NOTEBOOK = [
    ("exploratorio.png", "Analisis exploratorio"),
    ("convergencia.png", "Convergencia del ensamble"),
    ("calibracion.png", "Curva de calibracion"),
    ("importancias.png", "Importancia de variables"),
]


# --------------------------------------------------------------- estilos ---
def _inyectar_estilos():
    t = TEMA
    st.markdown(f"""
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stMainBlockContainer"] {{
            animation: fadeIn 0.4s ease-out;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        .st-key-kpi_strip, .st-key-kpi_strip_desempeno, .st-key-mini_panel,
        .st-key-ficha_sidebar, .st-key-pipeline_panel, .st-key-particion_panel {{
            background-color: {t['superficie']};
            border-radius: 0.5rem;
            padding: 1.4rem 1.2rem;
            transition: background-color 0.2s ease;
        }}
        .st-key-kpi_strip [data-testid="stColumn"]:not(:first-child),
        .st-key-kpi_strip_desempeno [data-testid="stColumn"]:not(:first-child),
        .st-key-mini_panel [data-testid="stColumn"]:not(:first-child),
        .st-key-particion_panel [data-testid="stColumn"]:not(:first-child) {{
            border-left: 1px solid {t['linea']};
            padding-left: 1.4rem;
        }}
        .st-key-kpi_strip [data-testid="stColumn"]:not(:last-child),
        .st-key-kpi_strip_desempeno [data-testid="stColumn"]:not(:last-child),
        .st-key-mini_panel [data-testid="stColumn"]:not(:last-child),
        .st-key-particion_panel [data-testid="stColumn"]:not(:last-child) {{
            padding-right: 1.4rem;
        }}
        [data-testid="stMetricValue"] {{
            font-family: {t['mono']};
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.01em;
        }}
        [data-testid="stMetricLabel"] {{
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            opacity: 0.75;
        }}
        [data-testid="stTabs"] button {{ transition: color 0.15s ease; }}
        </style>
        """, unsafe_allow_html=True)


def _chip(texto, color_key):
    st.markdown(
        f'<span style="background-color:{ESTADO[color_key]};'
        f'color:{ESTADO_TEXTO[color_key]};padding:3px 12px;border-radius:12px;'
        f'font-size:0.85rem;font-weight:600;display:inline-block;'
        f'margin:2px 6px 2px 0">{texto}</span>', unsafe_allow_html=True)


def _leyenda_colores():
    chips = "".join(
        f'<span style="background-color:{COLORES[n]};color:{TEXTO[n]};'
        f'padding:3px 12px;border-radius:12px;margin-right:8px;'
        f'font-size:0.85rem;font-weight:600;display:inline-block;'
        f'margin-bottom:4px">{n}</span>'
        for n in NIVELES_ORDEN)
    st.markdown(chips, unsafe_allow_html=True)


# --------------------------------------------------------------------- IO ---
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


@st.cache_resource
def cargar_modelo_congelado():
    """Cachea el bosque ya cargado entre reruns (no lo reentrena ni lo altera)."""
    return inf.cargar_modelo()


def _texto_unidad(met):
    r = (met or {}).get("representacion", {})
    return f"{r.get('unidad_espacial', 'unidad')} x {r.get('unidad_temporal', 'dia')}"


def _texto_umbrales(met):
    """Umbrales de la etiqueta, leidos en vivo de metricas.json (nunca
    hardcodeados). Solo cubre el caso de umbral global unico."""
    e = (met or {}).get("etiqueta", {})
    ub, ua = e.get("umbral_bajo"), e.get("umbral_alto")
    if isinstance(ub, list) or isinstance(ua, list) or ub is None or ua is None:
        return None
    ub, ua = float(ub), float(ua)
    medio_desde = ub + 1 if ub == int(ub) else ub
    fmt = lambda x: f"{int(x)}" if x == int(x) else f"{x:g}"
    return f"Bajo ≤ {fmt(ub)} · Medio {fmt(medio_desde)}–{fmt(ua)} · Alto > {fmt(ua)}"


def _matriz_coloreada(d):
    """Pivot parroquia x dia, coloreado. Reutilizado por Resumen, Matriz/Detalle
    y Nueva prediccion -- la unica logica de pintado de toda la interfaz."""
    dias_vis = [x for x in DIAS if x in d["dia_semana"].unique()]
    niveles = d.pivot_table(index="unidad", columns="dia_semana",
                            values="nivel", aggfunc="first").reindex(columns=dias_vis)
    probas = d.pivot_table(index="unidad", columns="dia_semana",
                           values="prob_clase_predicha", aggfunc="first").reindex(columns=dias_vis)
    niveles = niveles.loc[
        niveles.apply(lambda r: min(ORDEN_NIVEL.get(v, 9) for v in r.dropna())
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
                est.loc[i, j] = ("" if pd.isna(n) else
                                 f"background-color: {COLORES[n]}; color: {TEXTO[n]}; text-align: center")
        return est

    st.dataframe(matriz.style.apply(pintar, axis=None), use_container_width=True)


def _top_riesgo(d, n=6):
    top = (d[d["nivel"] != "Sin datos suficientes"]
          .sort_values(["nivel", "prob_clase_predicha"],
                      key=lambda s: s.map(ORDEN_NIVEL) if s.name == "nivel" else s,
                      ascending=[True, False])
          .head(n)[["unidad", "dia_semana", "nivel", "prob_clase_predicha"]])
    st.dataframe(top, hide_index=True, use_container_width=True,
                column_config={
                    "unidad": "Parroquia", "dia_semana": "Dia", "nivel": "Nivel",
                    "prob_clase_predicha": st.column_config.ProgressColumn(
                        "Probabilidad", min_value=0.0, max_value=1.0, format="%.2f"),
                })


# ======================================================================== #
_inyectar_estilos()
df, met = cargar()

with st.sidebar.container(key="ficha_sidebar"):
    st.markdown("**Grupo #10 · Paralelo P2**")
    st.caption("CCPG1044 — IA · ESPOL · Random Forest propio (sin sklearn en la prediccion)")

st.title("Riesgo de transito en Guayaquil")
st.caption("Predicción por parroquia y día de la semana")

if df is None:
    st.error("No se encontro `salidas/matriz_riesgo.parquet`. Corre `proyecto.ipynb` primero.")
    st.stop()

_alcance = (met or {}).get("alcance", {})

tab_resumen, tab_pred, tab_matriz, tab_desempeno, tab_metodo = st.tabs(
    ["Resumen", "Nueva predicción", "Matriz y detalle", "Desempeño", "Metodología"])

# ============================================================== RESUMEN ===
with tab_resumen:
    with st.container(key="kpi_strip"):
        c1, c2, c3, c4, c5 = st.columns(5, gap="large")
        c1.metric("Semana objetivo", str(df["semana_objetivo"].iloc[0]))
        c2.metric("Parroquias evaluadas", f"{_alcance.get('G_elegibles', df['unidad'].nunique())}")
        c3.metric("Combinaciones evaluadas", f"{len(df):,}")
        c4.metric("Riesgo Alto", f"{int((df['nivel'] == 'Alto').sum()):,}")
        prom = df.loc[df["nivel"] != "Sin datos suficientes", "prob_clase_predicha"].mean()
        c5.metric("Probabilidad promedio", "n/d" if pd.isna(prom) else f"{prom:.2f}")

    st.caption("La IA estima el nivel de riesgo de la próxima semana utilizando "
              "información histórica del ECU 911.")

    st.subheader("Matriz de riesgo")
    _leyenda_colores()
    _matriz_coloreada(df)

    col_izq, col_der = st.columns(2, gap="large")
    with col_izq:
        st.caption("Distribución de niveles")
        st.bar_chart(df["nivel"].value_counts().reindex(NIVELES_ORDEN).dropna(), height=220)
    with col_der:
        st.caption("Combinaciones de mayor riesgo")
        _top_riesgo(df)

# ======================================================= NUEVA PREDICCIÓN ==
with tab_pred:
    st.subheader("Actualizar predicción")
    st.caption("Sube el CSV mensual del ECU 911")

    archivo = st.file_uploader("Archivo CSV", type=["csv"], label_visibility="collapsed")

    if archivo is not None:
        tmp_path = os.path.join(tempfile.gettempdir(), f"ecu911_upload_{archivo.name}")
        with open(tmp_path, "wb") as f:
            f.write(archivo.getbuffer())

        modelo_cong = cargar_modelo_congelado()
        with st.spinner("Validando archivo..."):
            historial_actual = inf.cargar_historial()
            df_valido, mensaje_val = inf.validar_csv(tmp_path, modelo_cong["meta"], historial_actual)

        if df_valido is not None:
            _chip("✓ Formato válido", "ok")
            _chip("✓ Datos de Guayaquil", "ok")
            _chip("✓ Tránsito y Movilidad", "ok")
            _chip("✓ Mes disponible", "ok")
            st.caption(mensaje_val)

            if st.button("Generar nueva predicción", type="primary"):
                st.info("Modelo congelado · sin reentrenamiento")
                try:
                    with st.spinner("Generando predicción con el modelo congelado..."):
                        resultado = inf.predecir_desde_csv_nuevo(tmp_path)
                except inf.CSVInvalido as e:
                    resultado = {"ok": False, "mensaje": str(e)}
                st.session_state["resultado_inferencia"] = resultado
        else:
            _chip("✕ Archivo rechazado", "error")
            st.caption(mensaje_val)

    resultado = st.session_state.get("resultado_inferencia")
    if resultado:
        st.divider()
        if resultado["ok"]:
            st.success(f"Predicción generada para la semana **{resultado['semana_objetivo']}** "
                      f"({resultado['n_registros_nuevos']:,} registros nuevos incorporados)")
            st.caption("Predicción experimental — no sustituye ni sobrescribe la corrida oficial.")

            r1, r2, r3 = st.columns(3, gap="large")
            r1.metric("Semana objetivo", resultado["semana_objetivo"])
            resumen_niveles = resultado["resumen"] or {}
            r2.metric("Riesgo Alto", resumen_niveles.get("Alto", 0))
            r3.metric("Registros en el historial", f"{resultado['n_registros_historial']:,}")

            st.subheader("Matriz de riesgo (nueva predicción)")
            _leyenda_colores()
            _matriz_coloreada(resultado["matriz"])

            col_izq, col_der = st.columns(2, gap="large")
            with col_izq:
                st.caption("Combinaciones de mayor riesgo")
                _top_riesgo(resultado["matriz"])
            with col_der:
                st.caption("Probabilidades por clase")
                cols_p = [c for c in ["unidad", "dia_semana", "p_bajo", "p_medio", "p_alto"]
                         if c in resultado["matriz"].columns]
                st.dataframe(resultado["matriz"][cols_p], hide_index=True,
                            use_container_width=True, height=260)
        else:
            _chip("✕ No se generó la predicción", "error")
            st.caption(resultado["mensaje"])

# =================================================== MATRIZ Y DETALLE ===
with tab_matriz:
    st.sidebar.header("Filtros")
    unidades = sorted(df["unidad"].unique().tolist())
    sel_u = st.sidebar.multiselect("Parroquia", unidades, default=unidades)
    sel_d = st.sidebar.multiselect("Día de la semana", DIAS, default=DIAS)
    sel_n = st.sidebar.multiselect("Nivel de riesgo", NIVELES_ORDEN, default=NIVELES_ORDEN)
    prob_min = st.sidebar.slider("Probabilidad mínima", 0.0, 1.0, 0.0, 0.05)

    _excluidas = _alcance.get("unidades_excluidas_sin_historial")
    st.sidebar.caption(
        ("\"Sin datos suficientes\" = sin historial de entrenamiento; no pasa "
         "por el modelo." +
         (f" ({_excluidas} de {_alcance.get('G_total', '?')} parroquias.)"
          if _excluidas is not None else "")))

    d = df[df["unidad"].isin(sel_u) & df["dia_semana"].isin(sel_d) & df["nivel"].isin(sel_n)]
    d = d[(d["prob_clase_predicha"].fillna(0) >= prob_min) | (d["nivel"] == "Sin datos suficientes")]

    if len(d) == 0:
        st.warning("Ningún resultado con los filtros actuales.")
    else:
        st.subheader("Matriz de riesgo")
        _leyenda_colores()
        _matriz_coloreada(d)

        _umbrales_txt = _texto_umbrales(met)
        with st.expander("¿Por qué el riesgo Alto se concentra en Guayaquil?"):
            if _umbrales_txt:
                st.caption(f"Umbrales: {_umbrales_txt}")
            st.markdown(
                "El umbral que separa Medio de Alto es un número absoluto de "
                "incidentes por día, igual para las seis parroquias. La cabecera "
                "cantonal tiene una escala histórica muy superior a las demás, "
                "así que el riesgo Alto queda concentrado ahí — no es evidencia "
                "de un rendimiento uniforme en todo el cantón.")

        st.divider()
        st.subheader("Detalle")
        cols = ["unidad", "canton", "dia_semana", "nivel", "prob_clase_predicha",
                "p_bajo", "p_medio", "p_alto"]
        cols = [c for c in cols if c in d.columns]
        vista = d[cols].sort_values(["nivel", "prob_clase_predicha"], ascending=[True, False])
        st.dataframe(vista, use_container_width=True, height=420, hide_index=True,
                    column_config={
                        "unidad": "Parroquia", "canton": "Cantón", "dia_semana": "Día",
                        "nivel": "Nivel",
                        "prob_clase_predicha": st.column_config.ProgressColumn(
                            "Prob. clase predicha", min_value=0.0, max_value=1.0, format="%.2f"),
                        "p_bajo": st.column_config.ProgressColumn(
                            "P(bajo)", min_value=0.0, max_value=1.0, format="%.2f"),
                        "p_medio": st.column_config.ProgressColumn(
                            "P(medio)", min_value=0.0, max_value=1.0, format="%.2f"),
                        "p_alto": st.column_config.ProgressColumn(
                            "P(alto)", min_value=0.0, max_value=1.0, format="%.2f"),
                    })
        st.download_button("Exportar CSV", vista.to_csv(index=False).encode("utf-8"),
                          file_name=f"reporte_riesgo_{df['semana_objetivo'].iloc[0]}.csv",
                          mime="text/csv")

# ========================================================= DESEMPEÑO ===
with tab_desempeno:
    if met is None or "comparativa" not in met:
        st.info("Corre el notebook para generar `salidas/metricas.json`.")
    else:
        comp = pd.DataFrame(met["comparativa"])
        propio = comp[comp["modelo"].str.contains("propio", case=False, na=False)]

        if not propio.empty:
            fila = propio.iloc[0]
            with st.container(key="kpi_strip_desempeno"):
                k1, k2, k3, k4, k5 = st.columns(5, gap="large")
                k1.metric("Accuracy", f"{fila['exactitud']:.4f}")
                k2.metric("F1 macro", f"{fila['F1 macro']:.4f}")
                k3.metric("F1 Bajo", f"{fila.get('F1 bajo', float('nan')):.4f}")
                k4.metric("F1 Medio", f"{fila.get('F1 medio', float('nan')):.4f}")
                k5.metric("F1 Alto", f"{fila.get('F1 alto', float('nan')):.4f}")

        st.subheader("Bosque propio vs. baseline sklearn")
        comp_sk = comp[comp["modelo"].str.contains("propio|sklearn", case=False, na=False)]
        st.bar_chart(comp_sk.set_index("modelo")["F1 macro"], height=220)
        with st.expander("Ver comparación completa (todos los baselines)"):
            st.dataframe(comp, use_container_width=True, hide_index=True)

        if "importancias" in met:
            st.subheader("Importancia de variables")
            imp = pd.DataFrame(met["importancias"]).sort_values("importancia", ascending=False)
            st.bar_chart(imp.set_index("variable"), height=260)

        if "por_unidad" in met:
            st.subheader("Desempeño por parroquia")
            df_unidad = pd.DataFrame(met["por_unidad"])
            if "clases_presentes" in df_unidad.columns:
                def _resaltar_pocas_clases(fila):
                    if fila.get("clases_presentes", 3) < 3:
                        return ["background-color: #fff3cd; color: #000000"] * len(fila)
                    return [""] * len(fila)
                st.dataframe(df_unidad.style.apply(_resaltar_pocas_clases, axis=1),
                             use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_unidad, use_container_width=True, hide_index=True)
            with st.expander("¿Por qué Guayaquil aparece con F1 macro bajo pese a acertar siempre?"):
                st.markdown(
                    "Cuando `clases_presentes` < 3 (fila resaltada), el F1 macro "
                    "promedia también las clases ausentes (F1=0): no es comparable "
                    "con una fila de 3 clases. Guayaquil casi siempre es Alto en "
                    "prueba, así que su `exactitud` — no su F1 macro aislado — es "
                    "la lectura correcta de su desempeño.")

        if "busqueda_validacion" in met and met["busqueda_validacion"]:
            with st.expander("Cómo seleccionamos los hiperparámetros"):
                st.caption("Rejilla evaluada solo contra validación, antes de tocar prueba.")
                bv = pd.DataFrame(met["busqueda_validacion"]).sort_values(
                    "f1_macro", ascending=False).reset_index(drop=True)
                hp = met.get("hiperparametros") or {}

                def _resaltar_ganador(fila):
                    gano = (fila.get("profundidad") == hp.get("profundidadMax")
                           and fila.get("potencia_pesos") == hp.get("potenciaPesos"))
                    return ["background-color: #d4edda; color: #000000" if gano else ""] * len(fila)

                st.dataframe(bv.style.apply(_resaltar_ganador, axis=1),
                            use_container_width=True, hide_index=True)

        with st.expander("Gráficos adicionales del notebook"):
            col_a, col_b = st.columns(2)
            for i, (nombre, titulo) in enumerate(GRAFICOS_NOTEBOOK):
                ruta = os.path.join(RUTA, nombre)
                with (col_a if i % 2 == 0 else col_b):
                    if os.path.exists(ruta):
                        st.image(ruta, caption=titulo, use_container_width=True)

        with st.expander("Detalles técnicos de la corrida"):
            fa, fb, fc = st.columns(3)
            for col, (clave, titulo) in zip(
                    (fa, fb, fc),
                    (("alcance", "Alcance"), ("hiperparametros", "Hiperparámetros"),
                     ("entorno", "Entorno"))):
                if clave in met:
                    with col:
                        st.markdown(f"**{titulo}**")
                        st.json(met[clave])

# ========================================================= METODOLOGÍA ===
with tab_metodo:
    st.subheader("Pipeline")
    with st.container(key="pipeline_panel"):
        st.markdown(
            "**ECU 911** → **Preprocesamiento** → **Tensor parroquia × día × "
            "semana** → **11 características** → **Train / Validación / Test** "
            "→ **Random Forest propio** → **Predicción** → **Nivel Bajo / Medio "
            "/ Alto**")

    hp = (met or {}).get("hiperparametros", {})
    t1, t2, t3, t4, t5, t6 = st.columns(6, gap="large")
    t1.metric("Árboles", hp.get("nArboles", "n/d"))
    t2.metric("Profundidad", hp.get("profundidadMax", "n/d"))
    t3.metric("Criterio", "Gini")
    t4.metric("Muestreo", "Bootstrap")
    t5.metric("Características", "11")
    t6.metric("Semilla", hp.get("semilla", "n/d"))

    st.info("El dataset abierto del ECU 911 no publica coordenadas ni hora del "
           "incidente. Por eso el modelo trabaja a nivel de parroquia y día.")

    st.subheader("Partición temporal")
    st.markdown("**Partición temporal, no aleatoria.**")
    with st.container(key="particion_panel"):
        p1, p2, p3 = st.columns(3, gap="large")
        with p1:
            st.markdown("**TRAIN**")
            st.caption("Jul 2021 – Dic 2024")
            st.metric("Instancias", f"{_alcance.get('n_train', 'n/d'):,}"
                      if isinstance(_alcance.get("n_train"), int) else "n/d")
        with p2:
            st.markdown("**VALIDACIÓN**")
            st.caption("Ene 2025 – Jun 2025")
            st.metric("Instancias", f"{_alcance.get('n_val', 'n/d'):,}"
                      if isinstance(_alcance.get("n_val"), int) else "n/d")
        with p3:
            st.markdown("**TEST**")
            st.caption("Jul 2025 – Dic 2025")
            st.metric("Instancias", f"{_alcance.get('n_test', 'n/d'):,}"
                      if isinstance(_alcance.get("n_test"), int) else "n/d")

    st.subheader("Limitaciones")
    l1, l2, l3 = st.columns(3, gap="large")
    with l1:
        st.markdown("**Sin coordenadas ni hora**")
        st.caption("Resolución: parroquia × día de la semana.")
    with l2:
        st.markdown("**Concentración espacial**")
        st.caption("La cabecera cantonal concentra la gran mayoría de eventos.")
    with l3:
        st.markdown("**Enero 2024**")
        st.caption("Caída anómala de registros en la fuente; se conservó sin imputar.")

    with st.expander("Ver metodología completa"):
        st.markdown(
            "1. **Partición temporal** cronológica, sin mezclar semanas.\n"
            "2. **Elegibilidad** por parroquia, calculada solo con entrenamiento.\n"
            "3. **Umbrales de etiqueta** (P60/P90) ajustados solo con "
            "entrenamiento y congelados para validación y prueba.\n"
            "4. **Bins de discretización** ajustados solo con entrenamiento.\n"
            "5. **Características** (rezagos, medias móviles, densidad "
            "histórica) con información estrictamente anterior a la semana "
            "objetivo: `dia_semana`, `es_fin_de_semana`, `mes`, `es_feriado`, "
            "`rezago_1..4`, `media_movil_4`, `media_movil_12`, "
            "`densidad_historica_parroquia`.\n"
            "6. **Random Forest propio en NumPy** — CART, Gini ponderado, "
            "bootstrap, selección aleatoria de atributos por nodo, "
            "ponderación de clases, agregación de probabilidades. "
            "`scikit-learn` solo aparece como referencia de comparación, "
            "nunca predice.\n"
            "7. **Selección de hiperparámetros** solo contra validación, y "
            "**una única evaluación** sobre prueba.")
