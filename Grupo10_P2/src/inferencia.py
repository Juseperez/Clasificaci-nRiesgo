"""
Componente 4 (nuevo, 17-ago) - Inferencia sobre CSV mensual nuevo con el
modelo YA ENTRENADO.

Esta capa NO reentrena nada. No modifica ningun artefacto oficial. La unica
escritura que hace, y solo si el CSV pasa todas las validaciones, es
actualizar `salidas/historial_filtrado.parquet`: el historial acumulado de
registros filtrados (Transito y Movilidad, Guayaquil) que hace falta para
calcular `media_movil_12` y `densidad_historica_parroquia` de la semana
nueva. Ese parquet NO es uno de los artefactos oficiales del proyecto (no es
`matriz_riesgo.parquet`); es cache propio de esta capa, reconstruible desde
`datos/` en cualquier momento.

Congelados, nunca se escriben ni se recalculan aqui (auditoria del 17-ago):
    salidas/modelo.joblib, salidas/metricas.json, salidas/matriz_riesgo.parquet
    src/bosque.py, src/caracteristicas.py, src/preprocesamiento.py, src/metricas.py
    umbrales (P60/P90), hiperparametros, semilla, mask_elegible,
    particion train/validacion/test

Por que no se llama a ParticionadorTemporal.dividir(): la particion completa
tambien expone val/test, que esta capa no debe tocar ni redefinir. En su
lugar, `semanas_train_congeladas()` reimplementa la MISMA formula de dos
lineas (fecha <= corte_train, con burn-in) pero restringida por construccion
a las primeras `meta_congelada["S"]` semanas del tensor original: ninguna
semana agregada por inferencia puede calificar como train, sin depender de
recomputar la particion.

Por que no se recalculan los bordes de discretizacion "a mano": SI se
recalculan, pero llamando a `GeneradorCaracteristicas.construirX()` sin
tocarla, con las MISMAS semanas_train (siempre las mismas 172, por lo de
arriba). La auditoria del 17-ago verifico que asi los bordes salen
identicos a los del entrenamiento original, con predicciones bit-exactas
contra `matriz_riesgo.parquet`.

Por que no se reutilizan Preprocesador.asignarUnidadDia()/agregarConteos():
esas funciones derivan el origen de semanas y el catalogo de unidades del
propio CSV recibido (`origen = df["ts"].min()`, `unidades = sort(unique(df))`).
Sobre un CSV mensual suelto eso da un origen y una geometria de fila
distintos a los del modelo congelado, y el tensor resultante no se puede
alinear con `mask_elegible`. `reconstruir_tensor()` hace la misma agregacion
(mismo bincount, mismo descarte de semana final incompleta -- reutilizando
`Preprocesador._ultima_semana_completa`, sin reimplementarlo) pero contra el
origen y el catalogo YA CONGELADOS en `modelo.joblib`.

Flujo:
    CSV nuevo
      -> validar_csv()            columnas, fecha, Guayaquil, Transito,
                                   parroquia conocida, sin fechas futuras,
                                   mes no cargado (evita re-cargar el mismo
                                   archivo/mes dos veces)
      -> incorporar_historial()   concatena al historial completo
      -> reconstruir_tensor()     bincount contra origen/catalogo congelados
      -> GeneradorCaracteristicas.construirX()   sin tocar
      -> predecir_matriz_semana() bosque congelado, sin tocar
      -> MatrizRiesgo             sin tocar
"""
import glob
import os

import numpy as np
import pandas as pd

from src.preprocesamiento import Preprocesador
from src.caracteristicas import GeneradorCaracteristicas, BURN_IN
from src.riesgo import predecir_matriz_semana, MatrizRiesgo

RUTA_MODELO = os.path.join("salidas", "modelo.joblib")
RUTA_HISTORIAL = os.path.join("salidas", "historial_filtrado.parquet")
RUTA_DATOS_SEMILLA = "datos"
CORTE_TRAIN = "2024-12-31"    # debe coincidir con el usado para entrenar el bosque congelado

_COLS_HISTORIAL = ["ts", "cod_parroquia", "parroquia", "canton", "archivo"]


class CSVInvalido(Exception):
    """Fallo interno inesperado (no una validacion normal de negocio)."""


# --------------------------------------------------------------- utilidad --
def _unidad_desde_cod(cod_parroquia):
    """
    Mismo transform que Preprocesador.asignarUnidadDia() para
    nivel_espacial='parroquia': float -> Int64 -> str, para que 90150.0
    quede "90150" y coincida con el catalogo congelado en modelo.joblib.
    """
    return cod_parroquia.astype(float).astype("Int64").astype(str)


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo congelado (bosque, umbrales, nombres, mask_elegible,
    meta). No lo modifica ni lo reentrena."""
    import joblib
    return joblib.load(ruta)


# ---------------------------------------------------------------- historial --
def historial_vacio():
    return pd.DataFrame({c: pd.Series(dtype=object) for c in _COLS_HISTORIAL})


def cargar_historial(ruta_historial=RUTA_HISTORIAL, ruta_datos_semilla=RUTA_DATOS_SEMILLA,
                     construir_si_falta=True):
    """
    Carga el historial filtrado acumulado. Si no existe todavia, lo
    construye UNA VEZ a partir de los CSV originales en datos/ (los mismos
    54 archivos con los que se entreno el modelo), reutilizando
    Preprocesador.consolidar + filtrarAlcance -- nunca asignarUnidadDia ni
    agregarConteos. Tarda ~2 min la primera vez; despues queda cacheado en
    disco y esta funcion solo lee el parquet.
    """
    if os.path.exists(ruta_historial):
        return pd.read_parquet(ruta_historial)
    if not construir_si_falta:
        return historial_vacio()

    pre = Preprocesador(categoria="TRANSITO", alcance=("canton", "GUAYAQUIL"),
                        nivel_espacial="parroquia")
    rutas = sorted(glob.glob(os.path.join(ruta_datos_semilla, "*.csv")))
    df = pre.consolidar(rutas)
    df = pre.filtrarAlcance(df)
    cols = [c for c in _COLS_HISTORIAL if c in df.columns]
    hist = df[cols].reset_index(drop=True)
    guardar_historial(hist, ruta_historial)
    return hist


def guardar_historial(historial, ruta_historial=RUTA_HISTORIAL):
    carpeta = os.path.dirname(ruta_historial)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    historial.to_parquet(ruta_historial, index=False)
    return ruta_historial


# --------------------------------------------------------------- validacion --
def validar_csv(ruta_csv, meta_congelada, historial_actual, hoy=None):
    """
    Reutiliza Preprocesador.consolidar + filtrarAlcance (sin tocarlas) y
    aplica encima las validaciones propias de esta capa.

    Devuelve (df_valido, mensaje) si el CSV se acepta, o (None, mensaje) si
    se rechaza. En el caso de rechazo NO se toca el historial ni se corre
    inferencia -- el llamador debe verificar `df_valido is None`.
    """
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.now().normalize()
    rutas = [ruta_csv] if isinstance(ruta_csv, str) else list(ruta_csv)

    pre = Preprocesador(categoria="TRANSITO", alcance=("canton", "GUAYAQUIL"),
                        nivel_espacial="parroquia")
    try:
        df = pre.consolidar(rutas)
        df = pre.filtrarAlcance(df)
    except (FileNotFoundError, ValueError) as e:
        return None, f"CSV rechazado: {e}"

    if len(df) == 0:
        return None, ("CSV rechazado: no quedaron registros de Transito y "
                      "Movilidad en Guayaquil despues de filtrar.")

    # -- fechas futuras -------------------------------------------------
    futuras = df["ts"] > hoy
    if futuras.any():
        return None, (f"CSV rechazado: contiene {int(futuras.sum())} registro(s) "
                      f"con fecha futura, posterior a hoy ({hoy.date()}).")

    # -- parroquia fuera del catalogo congelado --------------------------
    df = df.copy()
    df["_unidad"] = _unidad_desde_cod(df["cod_parroquia"])
    catalogo = set(str(u) for u in meta_congelada["unidades"])
    desconocidas = sorted(set(df.loc[~df["_unidad"].isin(catalogo), "_unidad"].unique()))
    if desconocidas:
        return None, (f"CSV rechazado: codigo(s) de parroquia fuera del catalogo "
                      f"congelado de {len(catalogo)} parroquias: {desconocidas}. "
                      "No se agregan unidades nuevas al modelo.")

    # -- mes ya cargado ---------------------------------------------------
    meses_nuevo = set(df["ts"].dt.to_period("M").astype(str))
    if len(historial_actual):
        meses_cargados = set(
            pd.to_datetime(historial_actual["ts"]).dt.to_period("M").astype(str))
    else:
        meses_cargados = set()
    if meses_nuevo and meses_nuevo <= meses_cargados:
        return None, (f"CSV rechazado: el/los mes(es) {sorted(meses_nuevo)} ya "
                      "estan en el historial acumulado. Sube un archivo de un "
                      "mes que todavia no se haya cargado.")

    # -- NO se deduplica por fila -----------------------------------------
    # El dataset del ECU 911 no trae un identificador de incidente: dos
    # filas identicas (misma fecha, misma parroquia, mismo subtipo) pueden
    # ser dos choques distintos ese mismo dia, exactamente igual que en
    # Preprocesador.agregarConteos() (que tampoco deduplica: cada fila es un
    # evento). Deduplicar aqui por fila borraria incidentes reales. El
    # riesgo real que hay que evitar -- re-subir el mismo archivo/mes dos
    # veces -- ya lo bloquea el chequeo de "mes ya cargado" de arriba.

    # -- mes incompleto: advertencia, no rechazo --------------------------
    # Un CSV puede legitimamente cubrir solo una parte de un mes (se sube a
    # mitad de mes, o el mes real termino antes). No se rechaza: el
    # descarte de la semana final incompleta ya lo resuelve
    # reconstruir_tensor(), reutilizando la misma regla que usa el
    # Preprocesador original.
    dias_cubiertos = df["ts"].dt.normalize().nunique()
    aviso_incompleto = ""
    if dias_cubiertos < 25:
        aviso_incompleto = (f" [aviso: el archivo cubre solo {dias_cubiertos} "
                            "dia(s) distintos, podria ser un mes incompleto; "
                            "la semana final sin cerrar se descartara "
                            "automaticamente]")

    mensaje = f"CSV valido: {len(df):,} registro(s) nuevo(s)" + aviso_incompleto
    return df, mensaje


def incorporar_historial(historial_actual, df_valido):
    """
    Concatena el CSV ya validado al historial acumulado.

    No deduplica por fila (ver comentario en validar_csv): la proteccion
    contra re-cargar el mismo mes dos veces es el chequeo de "mes ya
    cargado" en validar_csv, que ya impide llegar aqui con datos repetidos.

    Devuelve (historial_actualizado, 0) -- el segundo valor se conserva por
    compatibilidad de firma, siempre 0 porque no hay deduplicacion. No
    escribe nada en disco -- ver guardar_historial().
    """
    cols = [c for c in _COLS_HISTORIAL if c in df_valido.columns]
    nuevo = df_valido[cols].reset_index(drop=True)
    if len(historial_actual):
        combinado = pd.concat([historial_actual[cols], nuevo], ignore_index=True)
    else:
        combinado = nuevo
    return combinado.reset_index(drop=True), 0


# ----------------------------------------------------------- reconstruccion --
def semanas_train_congeladas(meta_congelada, corte_train=CORTE_TRAIN):
    """
    Indices de semana de ENTRENAMIENTO, derivados solo de las fechas del
    tensor original congelado (las primeras meta_congelada["S"] semanas).
    Nunca llama a ParticionadorTemporal.dividir(): por construccion, ninguna
    semana agregada por esta capa (siempre posterior a S0) puede calificar
    como train, sin necesidad de recomputar la particion completa (que
    tambien expondria val/test).
    """
    S0 = meta_congelada["S"]
    fechas0 = pd.to_datetime(pd.Series(meta_congelada["fecha_semana"][:S0]))
    usable = np.arange(S0) >= BURN_IN
    tr = usable & (fechas0 <= pd.Timestamp(corte_train)).values
    return np.nonzero(tr)[0]


def reconstruir_tensor(historial, meta_congelada):
    """
    Reemplaza Preprocesador.asignarUnidadDia()+agregarConteos(): agrega el
    historial acumulado contra el ORIGEN y el CATALOGO DE UNIDADES
    congelados en modelo.joblib (nunca derivados del historial), y descarta
    la semana final incompleta con la MISMA regla que usa el Preprocesador
    original (se reutiliza el staticmethod, no se reimplementa).

    Devuelve (c, meta_extendido). meta_extendido es una copia de
    meta_congelada con solo S, fecha_semana, ultima_fecha_observada y
    semanas_incompletas_descartadas actualizados: unidades, origen, G, T,
    nombre_unidad y canton_unidad quedan exactamente los del modelo
    congelado.
    """
    origen = pd.Timestamp(meta_congelada["origen_semanas"])
    unidades = [str(u) for u in meta_congelada["unidades"]]
    idx_u = {u: i for i, u in enumerate(unidades)}
    G, T = meta_congelada["G"], meta_congelada["T"]

    h = historial.copy()
    h["ts"] = pd.to_datetime(h["ts"])
    if "_unidad" not in h.columns:
        h["_unidad"] = _unidad_desde_cod(h["cod_parroquia"])

    # Defensa en profundidad: validar_csv ya deberia haber rechazado
    # cualquier parroquia fuera del catalogo antes de llegar aqui.
    conocidas = h["_unidad"].isin(idx_u)
    if not conocidas.all():
        h = h.loc[conocidas].copy()

    g = h["_unidad"].map(idx_u).values.astype(np.int64)
    t = h["ts"].dt.dayofweek.values.astype(np.int64)
    semana = (h["ts"].dt.normalize() - origen).dt.days // 7
    if (semana < 0).any():
        raise CSVInvalido(
            f"Hay registros con fecha anterior al origen congelado ({origen.date()}); "
            "no se pueden ubicar en el tensor.")
    s = semana.values.astype(np.int64)
    S = int(s.max()) + 1

    plano = np.bincount(g * (T * S) + t * S + s, minlength=G * T * S)
    c = plano.reshape(G, T, S).astype(np.int32)

    ultima_fecha = h["ts"].max().normalize()
    s_max = Preprocesador._ultima_semana_completa(origen, ultima_fecha)
    incompletas = S - 1 - s_max
    if incompletas > 0:
        c = c[:, :, :s_max + 1]
        S = s_max + 1

    meta_ext = dict(meta_congelada)
    meta_ext["S"] = S
    meta_ext["fecha_semana"] = [origen + pd.Timedelta(weeks=int(i)) for i in range(S)]
    meta_ext["ultima_fecha_observada"] = ultima_fecha
    meta_ext["semanas_incompletas_descartadas"] = int(incompletas)
    return c, meta_ext


# --------------------------------------------------------------------- API --
def predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=RUTA_MODELO,
                             ruta_historial=RUTA_HISTORIAL,
                             ruta_datos_semilla=RUTA_DATOS_SEMILLA,
                             hoy=None, guardar=True):
    """
    Punto de entrada unico de esta capa.

    Devuelve un dict:
        ok, mensaje, semana_objetivo, matriz (DataFrame, esquema identico a
        MatrizRiesgo.a_dataframe()), resumen (dict nivel->conteo),
        n_registros_nuevos, n_registros_historial.

    Si ok=False: el historial en disco y todos los artefactos congelados
    quedan exactamente como estaban. No se ejecuta ninguna inferencia.
    """
    modelo = cargar_modelo(ruta_modelo)
    meta_congelada = modelo["meta"]
    bosque = modelo["bosque"]
    mask_congelada = modelo["mask_elegible"]
    umbrales = modelo["umbrales"]
    nombres_esperados = modelo["nombres"]

    historial = cargar_historial(ruta_historial, ruta_datos_semilla)

    df_valido, mensaje = validar_csv(ruta_csv, meta_congelada, historial, hoy=hoy)
    if df_valido is None:
        return {"ok": False, "mensaje": mensaje, "semana_objetivo": None,
                "matriz": None, "resumen": None, "n_registros_nuevos": 0,
                "n_registros_historial": len(historial)}

    historial_actualizado, _n_dup_hist = incorporar_historial(historial, df_valido)

    sem_tr = semanas_train_congeladas(meta_congelada)
    c, meta_ext = reconstruir_tensor(historial_actualizado, meta_congelada)

    gc_ = GeneradorCaracteristicas()
    X, nombres = gc_.construirX(c, meta_ext, sem_tr, semanas_futuras=1,
                                mask_elegible=mask_congelada)
    if nombres != nombres_esperados:
        raise CSVInvalido("Las features generadas no coinciden con las del "
                          "modelo congelado; abortando sin guardar el historial.")

    G, T = meta_congelada["G"], meta_congelada["T"]
    S_obs = c.shape[2]
    i_sig = gc_.indices_semana_futura(G, T, S_obs, k=0)
    proba, _n_predichas = predecir_matriz_semana(bosque, X, i_sig, mask_congelada)

    mr = MatrizRiesgo(meta_ext, gc_.fecha_semana_futura(0), umbrales=umbrales)
    mr.desde_predicciones(proba, mascara_sin_datos=~mask_congelada)
    matriz = mr.a_dataframe()

    if guardar:
        guardar_historial(historial_actualizado, ruta_historial)

    return {
        "ok": True,
        "mensaje": mensaje,
        "semana_objetivo": str(gc_.fecha_semana_futura(0).date()),
        "matriz": matriz,
        "resumen": mr.resumen(),
        "n_registros_nuevos": int(len(df_valido)),
        "n_registros_historial": int(len(historial_actualizado)),
    }
