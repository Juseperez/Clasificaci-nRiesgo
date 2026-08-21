"""
Componente 3 - InterfazConsulta (UC3) + Componente 4 - Inferencia (UC4).

Interfaz grafica propia. Se ejecuta con

    streamlit run app.py

Lee los artefactos oficiales de salidas/ (matriz_riesgo.parquet, metricas.json,
los PNG del notebook) para la corrida OFICIAL, y usa src/inferencia.py para
generar predicciones nuevas a partir de un CSV mensual subido por el usuario,
con el modelo operativo YA ENTRENADO. No reentrena nada y no escribe
nunca modelo.joblib / metricas.json / matriz_riesgo.parquet.

Cinco areas: Resumen, Nueva prediccion, Matriz y detalle, Desempeno,
Metodologia. El tema visual (fondo navy, tipografia, radios) vive en
.streamlit/config.toml; el CSS de aqui solo agrega movimiento contenido y la
firma tipografica (lecturas numericas en monoespaciada tabular). Donde hace
falta color explicito fuera del tema (matriz de riesgo, resaltados de tabla,
chips de estado) se fija fondo+texto juntos para que sea legible sin importar
el tema del usuario.
"""
import html
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

        /* KPIs propios. No dependen de st.metric, por lo que nunca usan elipsis. */
        .kpi-grid-custom {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 1px;
            background: {t['linea']};
            border: 1px solid {t['linea']};
            border-radius: 0.55rem;
            overflow: hidden;
            margin-bottom: 0.9rem;
        }}
        .kpi-card-custom {{
            min-width: 0;
            background: {t['superficie']};
            padding: 1.05rem 1rem 1.15rem 1rem;
        }}
        .kpi-label-custom {{
            text-transform: uppercase;
            font-size: clamp(0.70rem, 0.85vw, 0.82rem);
            letter-spacing: 0.035em;
            opacity: 0.78;
            line-height: 1.25;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: normal;
            word-break: normal;
            min-height: 2.15em;
        }}
        .kpi-value-custom {{
            font-family: {t['mono']};
            font-variant-numeric: tabular-nums;
            font-size: clamp(1.35rem, 1.8vw, 2rem);
            font-weight: 700;
            line-height: 1.12;
            margin-top: 0.30rem;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
        }}

        /* Fallback para cualquier st.metric secundario que permanezca en expanders. */
        [data-testid="stMetric"] {{ min-width: 0 !important; }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] p {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            max-width: none !important;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {{
            overflow: visible !important;
            text-overflow: clip !important;
            max-width: none !important;
        }}
        @media (max-width: 900px) {{
            .kpi-grid-custom {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
        }}
        @media (max-width: 600px) {{
            .kpi-grid-custom {{ grid-template-columns: 1fr 1fr; }}
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


def _kpi_grid(items):
    """Renderiza KPIs responsivos sin truncar etiquetas ni fechas."""
    cards = []
    for etiqueta, valor in items:
        cards.append(
            '<div class="kpi-card-custom">'
            f'<div class="kpi-label-custom">{html.escape(str(etiqueta))}</div>'
            f'<div class="kpi-value-custom">{html.escape(str(valor))}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="kpi-grid-custom">' + ''.join(cards) + '</div>',
        unsafe_allow_html=True,
    )


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
    """Cachea el paquete del modelo congelado entre reruns; no reentrena ni altera artefactos."""
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
    st.caption("CCPG1044 — IA · ESPOL")

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
    prom = df.loc[df["nivel"] != "Sin datos suficientes", "prob_clase_predicha"].mean()
    _kpi_grid([
        ("Semana objetivo", str(df["semana_objetivo"].iloc[0])),
        ("Parroquias evaluadas", _alcance.get("G_elegibles", df["unidad"].nunique())),
        ("Combinaciones evaluadas", f"{len(df):,}"),
        ("Riesgo alto", f"{int((df['nivel'] == 'Alto').sum()):,}"),
        ("Confianza promedio", "n/d" if pd.isna(prom) else f"{prom:.2f}"),
    ])

    # Muestra qué estimador produce realmente la matriz oficial sin romper la vista si
    # el artefacto de modelo no está disponible.
    try:
        _paq_resumen = cargar_modelo_congelado()
        _modelo_resumen = str(_paq_resumen.get("modelo_nombre", "modelo congelado")).title()
    except Exception:
        _modelo_resumen = str((met or {}).get("modelo_mejor_validacion", "modelo congelado")).title()

    st.caption(
        f"Modelo operativo: {_modelo_resumen} · La IA estima el nivel de riesgo de la próxima "
        "semana utilizando información histórica del ECU 911. La confianza mostrada es la "
        "probabilidad de la clase predicha, no la probabilidad absoluta de que ocurra un accidente."
    )

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

            resumen_niveles = resultado["resumen"] or {}
            _kpi_grid([
                ("Semana objetivo", resultado["semana_objetivo"]),
                ("Riesgo Alto", resumen_niveles.get("Alto", 0)),
                ("Registros en el historial", f"{resultado['n_registros_historial']:,}"),
            ])

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
    prob_min = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.05)
    st.sidebar.caption(
        "Filtro visual sobre la probabilidad de la clase predicha. "
        "No cambia ni reentrena el modelo."
    )

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
    if met is None:
        st.info("Corre el notebook para generar `salidas/metricas.json`.")
    else:
        comp_val = pd.DataFrame(met.get("comparacion_familias_validacion", []))
        comp_test = pd.DataFrame(met.get("comparacion_familias_test", []))
        modelo_seleccionado = str(met.get("modelo_mejor_validacion", "extra trees"))

        # Verificación cruzada: metricas.json y modelo.joblib deben apuntar al mismo modelo.
        try:
            _paq_perf = cargar_modelo_congelado()
            _modelo_joblib = str(_paq_perf.get("modelo_nombre", modelo_seleccionado))
        except Exception:
            _modelo_joblib = modelo_seleccionado

        if _modelo_joblib.lower() != modelo_seleccionado.lower():
            st.warning(
                f"Inconsistencia de artefactos: métricas seleccionan {modelo_seleccionado.title()} "
                f"pero modelo.joblib contiene {_modelo_joblib.title()}. Regenera los artefactos."
            )

        # Busca el mismo modelo sin depender de mayúsculas/minúsculas.
        fila_val = pd.DataFrame()
        fila_test = pd.DataFrame()
        if not comp_val.empty and "modelo" in comp_val.columns:
            fila_val = comp_val[
                comp_val["modelo"].astype(str).str.lower() == modelo_seleccionado.lower()
            ]
        if not comp_test.empty and "modelo" in comp_test.columns:
            fila_test = comp_test[
                comp_test["modelo"].astype(str).str.lower() == modelo_seleccionado.lower()
            ]

        nombre_visible = modelo_seleccionado.title()
        st.subheader(f"Modelo seleccionado: {nombre_visible}")
        st.caption(
            "La familia se seleccionó exclusivamente con F1 Macro sobre validación "
            "temporal. El conjunto de prueba se utilizó después solo para medir desempeño final."
        )

        if not fila_test.empty:
            ft = fila_test.iloc[0]
            f1_val_sel = (
                float(fila_val.iloc[0]["F1 macro validacion"])
                if not fila_val.empty and "F1 macro validacion" in fila_val.columns
                else float("nan")
            )
            _kpi_grid([
                ("F1 validación", "n/d" if pd.isna(f1_val_sel) else f"{f1_val_sel:.4f}"),
                ("Accuracy test", f"{float(ft['exactitud']):.4f}"),
                ("F1 Macro test", f"{float(ft['F1 macro']):.4f}"),
                ("F1 Medio", f"{float(ft.get('F1 medio', float('nan'))):.4f}"),
                ("F1 Alto", f"{float(ft.get('F1 alto', float('nan'))):.4f}"),
                ("Brier", f"{float(ft.get('Brier', float('nan'))):.4f}"),
            ])
        else:
            st.warning(
                "`metricas.json` no contiene todavía la comparación final de familias. "
                "Vuelve a ejecutar `proyecto.ipynb` para regenerar los artefactos."
            )

        # Comparación que realmente decidió el modelo operativo.
        if not comp_val.empty:
            st.subheader("Comparación de los cinco modelos")
            st.caption("F1 Macro en validación temporal — criterio utilizado para seleccionar el modelo.")
            graf_val = comp_val.copy()
            if "F1 macro validacion" in graf_val.columns:
                graf_val = graf_val.sort_values("F1 macro validacion", ascending=False)
                st.bar_chart(
                    graf_val.set_index("modelo")["F1 macro validacion"],
                    height=250,
                )

        if not comp_val.empty and not comp_test.empty:
            cols_val = [c for c in ["modelo", "F1 macro validacion", "tiempo ajuste s"] if c in comp_val.columns]
            cols_test = [c for c in [
                "modelo", "exactitud", "F1 macro", "F1 bajo", "F1 medio",
                "F1 alto", "recall alto", "Brier"
            ] if c in comp_test.columns]

            tabla = comp_val[cols_val].merge(comp_test[cols_test], on="modelo", how="left")
            if "F1 macro validacion" in tabla.columns:
                tabla = tabla.sort_values("F1 macro validacion", ascending=False).reset_index(drop=True)

            def _resaltar_modelo_operativo(fila):
                if str(fila.get("modelo", "")).lower() == modelo_seleccionado.lower():
                    return ["background-color: #d4edda; color: #000000"] * len(fila)
                return [""] * len(fila)

            st.dataframe(
                tabla.style.apply(_resaltar_modelo_operativo, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                "Extra Trees se mantiene como modelo operativo porque ganó en validación. "
                "Aunque otro modelo pueda obtener una métrica ligeramente mayor en test, "
                "cambiar la selección después de observar test introduciría sesgo de selección."
            )

        # Todo lo siguiente pertenece específicamente al bosque desarrollado por el grupo.
        comp = pd.DataFrame(met.get("comparativa", []))
        if not comp.empty:
            with st.expander("Análisis complementario del Bosque Propio"):
                st.caption(
                    "Este bloque documenta la implementación propia en NumPy y sus benchmarks. "
                    "No corresponde al modelo operativo de Streamlit, que es Extra Trees."
                )

                propio = comp[comp["modelo"].str.contains("propio", case=False, na=False)]
                if not propio.empty:
                    fila = propio.iloc[0]
                    _kpi_grid([
                        ("Accuracy", f"{fila['exactitud']:.4f}"),
                        ("F1 Macro", f"{fila['F1 macro']:.4f}"),
                        ("F1 Bajo", f"{fila.get('F1 bajo', float('nan')):.4f}"),
                        ("F1 Medio", f"{fila.get('F1 medio', float('nan')):.4f}"),
                        ("F1 Alto", f"{fila.get('F1 alto', float('nan')):.4f}"),
                    ])

                comp_sk = comp[comp["modelo"].str.contains("propio|sklearn", case=False, na=False)]
                if not comp_sk.empty:
                    st.markdown("**Bosque Propio vs. Random Forest de referencia**")
                    st.bar_chart(comp_sk.set_index("modelo")["F1 macro"], height=220)

                with st.expander("Ver baselines y benchmark completo"):
                    st.dataframe(comp, use_container_width=True, hide_index=True)

        if "importancias" in met:
            with st.expander("Importancia de variables — Bosque Propio"):
                st.caption(
                    "Importancia interna por reducción de impureza del Bosque Propio. "
                    "No representa causalidad ni la importancia calculada por Extra Trees."
                )
                imp = pd.DataFrame(met["importancias"]).sort_values("importancia", ascending=False)
                st.bar_chart(imp.set_index("variable"), height=260)

        if "por_unidad" in met:
            with st.expander("Desempeño por parroquia — Bosque Propio"):
                st.caption(
                    "Este análisis por unidad fue calculado con las predicciones del Bosque Propio "
                    "y se conserva como diagnóstico complementario."
                )
                df_unidad = pd.DataFrame(met["por_unidad"])
                if "clases_presentes" in df_unidad.columns:
                    def _resaltar_pocas_clases(fila):
                        if fila.get("clases_presentes", 3) < 3:
                            return ["background-color: #fff3cd; color: #000000"] * len(fila)
                        return [""] * len(fila)

                    st.dataframe(
                        df_unidad.style.apply(_resaltar_pocas_clases, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(df_unidad, use_container_width=True, hide_index=True)

                st.markdown(
                    "**Lectura:** cuando una parroquia no contiene las tres clases en prueba, "
                    "su F1 Macro aislado no es directamente comparable con el de una unidad que sí las contiene."
                )

        if "busqueda_validacion" in met and met["busqueda_validacion"]:
            with st.expander("Ajuste interno de hiperparámetros — Bosque Propio"):
                st.caption(
                    "Rejilla del Bosque Propio evaluada únicamente en validación. "
                    "Es distinta de la comparación final entre las cinco familias."
                )
                bv = pd.DataFrame(met["busqueda_validacion"]).sort_values(
                    "f1_macro", ascending=False
                ).reset_index(drop=True)
                hp = met.get("hiperparametros") or {}

                def _resaltar_ganador(fila):
                    gano = (
                        fila.get("profundidad") == hp.get("profundidadMax")
                        and fila.get("potencia_pesos") == hp.get("potenciaPesos")
                    )
                    return ["background-color: #d4edda; color: #000000" if gano else ""] * len(fila)

                st.dataframe(
                    bv.style.apply(_resaltar_ganador, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Gráficos adicionales del notebook"):
            st.caption(
                "Exploratorio = datos. Convergencia, calibración e importancias = análisis del Bosque Propio."
            )
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
                (("alcance", "Alcance"), ("hiperparametros", "Bosque Propio"), ("entorno", "Entorno")),
            ):
                if clave in met:
                    with col:
                        st.markdown(f"**{titulo}**")
                        st.json(met[clave])


# ========================================================= METODOLOGÍA ===
with tab_metodo:
    # El paquete congelado es exactamente el mismo que utiliza src.inferencia.py.
    try:
        paquete_modelo = cargar_modelo_congelado()
    except Exception:
        paquete_modelo = {}

    nombre_operativo = str(
        paquete_modelo.get("modelo_nombre", (met or {}).get("modelo_mejor_validacion", "Extra Trees"))
    )

    st.subheader("Pipeline")
    with st.container(key="pipeline_panel"):
        st.markdown(
            "**ECU 911** → **Preprocesamiento** → **Tensor parroquia × día × semana** "
            "→ **11 características** → **Train / Validación / Test** "
            f"→ **Comparación de 5 modelos** → **{nombre_operativo.title()}** "
            "→ **Predicción** → **Nivel Bajo / Medio / Alto**"
        )

    config_operativa = paquete_modelo.get("configuracion_modelo", {}) or {}

    comp_val_met = pd.DataFrame((met or {}).get("comparacion_familias_validacion", []))
    f1_val_operativo = None
    if not comp_val_met.empty and "modelo" in comp_val_met.columns:
        _fv = comp_val_met[
            comp_val_met["modelo"].astype(str).str.lower() == nombre_operativo.lower()
        ]
        if not _fv.empty and "F1 macro validacion" in _fv.columns:
            f1_val_operativo = float(_fv.iloc[0]["F1 macro validacion"])

    _kpi_grid([
        ("Modelo", nombre_operativo.title()),
        ("Árboles", config_operativa.get("n_estimators", 300)),
        ("Profundidad", config_operativa.get("max_depth", 16)),
        ("Mín. hoja", config_operativa.get("min_samples_leaf", 5)),
        ("Max features", config_operativa.get("max_features", "sqrt")),
        ("F1 validación", "n/d" if f1_val_operativo is None else f"{f1_val_operativo:.4f}"),
    ])

    st.caption(
        "Extra Trees es el modelo operativo seleccionado por validación temporal. "
        "El Random Forest propio permanece como implementación desarrollada por el grupo, "
        "benchmark y evidencia de comprensión algorítmica."
    )

    st.info(
        "El dataset abierto del ECU 911 no publica coordenadas ni hora del incidente. "
        "Por eso la representación final trabaja a nivel de parroquia y día de la semana."
    )

    st.subheader("Partición temporal")
    st.markdown("**Partición temporal, no aleatoria.**")
    with st.container(key="particion_panel"):
        p1, p2, p3 = st.columns(3, gap="large")
        with p1:
            st.markdown("**TRAIN**")
            st.caption("Jul 2021 – Dic 2024")
            st.metric(
                "Instancias",
                f"{_alcance.get('n_train', 'n/d'):,}"
                if isinstance(_alcance.get("n_train"), int)
                else "n/d",
            )
        with p2:
            st.markdown("**VALIDACIÓN**")
            st.caption("Ene 2025 – Jun 2025")
            st.metric(
                "Instancias",
                f"{_alcance.get('n_val', 'n/d'):,}"
                if isinstance(_alcance.get("n_val"), int)
                else "n/d",
            )
        with p3:
            st.markdown("**TEST**")
            st.caption("Jul 2025 – Dic 2025")
            st.metric(
                "Instancias",
                f"{_alcance.get('n_test', 'n/d'):,}"
                if isinstance(_alcance.get("n_test"), int)
                else "n/d",
            )

    st.caption(
        "Train ajusta modelos y transformaciones · Validación selecciona familia e hiperparámetros · "
        "Test se reserva para la evaluación final."
    )

    st.subheader("Limitaciones")
    l1, l2, l3 = st.columns(3, gap="large")
    with l1:
        st.markdown("**Sin coordenadas ni hora**")
        st.caption("Resolución verificable: parroquia × día de la semana.")
    with l2:
        st.markdown("**Concentración espacial**")
        st.caption("La cabecera cantonal concentra la gran mayoría de eventos.")
    with l3:
        st.markdown("**Enero 2024**")
        st.caption("Caída anómala de registros en la fuente; se conservó sin imputar.")

    with st.expander("Ver metodología completa"):
        st.markdown(
            "1. **Datos y preprocesamiento:** lectura de los CSV del ECU 911, normalización, "
            "filtrado de Tránsito y Movilidad y cantón Guayaquil, y control de semanas incompletas.\n"
            "2. **Representación:** tensor `c[g,t,s]` con parroquia × día de la semana × semana.\n"
            "3. **Partición temporal:** train, validación y test conservan el orden cronológico; "
            "no se utiliza una división aleatoria.\n"
            "4. **Etiquetado:** umbrales de riesgo ajustados únicamente con entrenamiento mediante "
            "la estrategia adaptativa global → positivos → por día → discreto.\n"
            "5. **Características:** 11 variables de calendario e historia construidas con información "
            "anterior al horizonte: `dia_semana`, `es_fin_de_semana`, `mes`, `es_feriado`, "
            "`rezago_1..4`, `media_movil_4`, `media_movil_12` y `densidad_historica_parroquia`.\n"
            "6. **Bosque Propio:** implementación en NumPy de CART, Gini ponderado, bootstrap, "
            "selección aleatoria de atributos, pesos de clase y agregación de probabilidades.\n"
            "7. **Selección multimodelo:** se comparan Bosque Propio, Extra Trees, Gradient Boosting, "
            "Regresión Logística y MLP bajo las mismas particiones y características.\n"
            "8. **Criterio de selección:** F1 Macro en validación temporal. Extra Trees obtuvo el mayor "
            "valor y quedó congelado como modelo operativo.\n"
            "9. **Evaluación final:** test se utiliza después de congelar las configuraciones y no se "
            "emplea para cambiar el modelo seleccionado.\n"
            "10. **Inferencia:** `src/inferencia.py` valida un CSV mensual nuevo, actualiza el historial "
            "reconstruible, vuelve a generar las 11 características y utiliza el **mismo Extra Trees "
            "congelado**, sin reentrenar ni sobrescribir los artefactos oficiales."
        )

    with st.expander("¿Qué hace la opción Confianza mínima?"):
        st.markdown(
            "Es un **filtro visual** de la pestaña *Matriz y detalle*. Una celda como `Medio (0.77)` "
            "significa que el clasificador asignó la clase Medio con una confianza de 0.77. "
            "Si se fija una confianza mínima de 0.70, se ocultan de la vista las predicciones cuya "
            "confianza en la clase elegida sea menor que 0.70. **No cambia la predicción, no modifica "
            "los umbrales y no reentrena el modelo.** La cifra tampoco representa la probabilidad "
            "absoluta de que ocurra un accidente; es la probabilidad asignada a la clase predicha."
        )