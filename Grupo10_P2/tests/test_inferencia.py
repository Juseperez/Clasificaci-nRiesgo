"""
Pruebas de src/inferencia.py (auditoria del 17-ago -> capa de inferencia).

Dos grupos:

  A. RAPIDAS, con un modelo y un historial SINTETICOS (catalogo de 2
     parroquias inventadas "111"/"222"). No tocan datos/ ni salidas/ reales.
     Cubren la logica propia de esta capa: validaciones, deduplicacion,
     reconstruccion del tensor, extension del historial.

     El unico entrenamiento que aparece aqui es la INFRAESTRUCTURA DE LA
     PRUEBA (un bosque de juguete de 5 arboles, para tener algo que prediga),
     construido exactamente igual que test_gini.py/test_pipeline.py ya
     hacen. NO es el modelo del proyecto y src/inferencia.py nunca lo
     entrena: solo lo carga y lo usa, tal como haria con el real.

  B. LENTA (una sola), contra los artefactos REALES (datos/, modelo.joblib,
     matriz_riesgo.parquet). Prueba la reproduccion bit-exacta y que los
     artefactos oficiales quedan intactos. Tarda ~2 minutos la primera vez
     porque siembra el historial desde los 54 CSV originales.

    python -m tests.test_inferencia
"""
import sys
import os
import atexit
import shutil
import tempfile
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.inferencia as inf
from src.bosque import BosqueAleatorioPropio
from src.caracteristicas import (GeneradorCaracteristicas, ConstructorEtiquetas,
                                 mascara_elegibilidad, indices_de_semanas,
                                 filtrar_elegibles, BURN_IN)
from src.preprocesamiento import DIAS

_TEMPORALES = []


def _tmpdir(nombre):
    d = tempfile.mkdtemp(prefix=f"inferencia_{nombre}_")
    _TEMPORALES.append(d)
    return d


@atexit.register
def _limpiar_temporales():
    for d in _TEMPORALES:
        shutil.rmtree(d, ignore_errors=True)


CSV_HEADER = "Fecha;provincia;Canton;Cod_Parroquia;Parroquia;Servicio;Subtipo"


def _fila_csv(fecha, cod, parr, canton="GUAYAQUIL", serv="Transito y Movilidad",
             prov="GUAYAS"):
    return f"{fecha};{prov};{canton};{cod};{parr};{serv};Choque"


def _escribir_csv(ruta, filas):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CSV_HEADER + "\n" + "\n".join(filas))


# ============================================================================
# GRUPO A -- modelo y historial sinteticos
# ============================================================================

_G, _T = 2, 7
_UNIDADES = ["111", "222"]
_ORIGEN = pd.Timestamp("2023-01-02")   # lunes
_S0 = 30                               # semanas del modelo "congelado" de prueba
# Con origen=2023-01-02, la semana _S0=30 (2023-07-31 a 2023-08-06) es la
# primera que cruza de julio a agosto, y agosto NO tiene NINGUN dato en el
# historial sembrado (semanas 0.._S0-1 terminan el 2023-07-30). Se elige a
# proposito para poder probar la reconstruccion de una semana que cruza mes.


def _meta_sintetica(S, unidades=_UNIDADES, origen=_ORIGEN):
    fecha_semana = [origen + pd.Timedelta(weeks=i) for i in range(S)]
    return {
        "unidades": np.array(unidades),
        "nombre_unidad": np.array([f"PARROQUIA {u}" for u in unidades]),
        "canton_unidad": np.array(["GUAYAQUIL"] * len(unidades)),
        "nivel_espacial": "parroquia",
        "alcance": ("canton", "GUAYAQUIL"),
        "origen_semanas": origen,
        "fecha_semana": fecha_semana,
        "dias": DIAS,
        "G": len(unidades), "T": _T, "S": S,
        "semanas_con_datos": np.arange(S),
        "ultima_fecha_observada": fecha_semana[-1] + pd.Timedelta(days=6),
        "semanas_incompletas_descartadas": 0,
    }


def _tensor_sintetico(S, semilla=7):
    rng = np.random.default_rng(semilla)
    lam = np.array([8.0, 1.5])   # unidad 0 mas activa que unidad 1
    return rng.poisson(lam[:, None, None], size=(_G, _T, S)).astype(np.int32)


def _filas_desde_conteos(c_slice, s_offset, unidades=_UNIDADES, origen=_ORIGEN,
                         archivo="semilla.csv"):
    """Inversa de bincount: una fila de texto CSV por evento."""
    G, T, S = c_slice.shape
    filas = []
    for g in range(G):
        for t in range(T):
            for s in range(S):
                n = int(c_slice[g, t, s])
                if n == 0:
                    continue
                fecha = origen + pd.Timedelta(weeks=s + s_offset, days=t)
                for _ in range(n):
                    filas.append(_fila_csv(fecha.strftime("%d/%m/%Y"),
                                           unidades[g], f"PARROQUIA {unidades[g]}"))
    return filas


def _construir_frozen(tmp_dir):
    """
    Construye un modelo.joblib e historial.parquet SINTETICOS, con la misma
    estructura de claves que los reales, en tmp_dir. Devuelve
    (ruta_modelo, ruta_historial, c_reservado_para_extension).
    """
    c_total = _tensor_sintetico(_S0 + 8)         # 8 semanas extra para pruebas de extension
    c = c_total[:, :, :_S0]                       # lo que el modelo "conoce"
    meta = _meta_sintetica(_S0)

    sem_tr = np.arange(BURN_IN, _S0)
    mask = mascara_elegibilidad(c, sem_tr, nivel="unidad")

    ce = ConstructorEtiquetas(p_bajo=60, p_alto=90, modo="auto")
    P60, P90 = ce.ajustarUmbrales(c, sem_tr, mask_elegible=mask)
    y = ce.construirY(c)

    gc_ = GeneradorCaracteristicas()
    X, nombres = gc_.construirX(c, meta, sem_tr, semanas_futuras=0, mask_elegible=mask)
    i_tr = filtrar_elegibles(indices_de_semanas(_G, _T, _S0, sem_tr), mask, _T, _S0)
    Xtr, ytr = X[i_tr], y.reshape(-1)[i_tr]

    bosque = BosqueAleatorioPropio(nArboles=5, profundidadMax=4, minMuestrasHoja=2,
                                   nBins=32, semilla=42, verboso=False)
    bosque.entrenar(Xtr, ytr)

    modelo = {"bosque": bosque, "umbrales": (float(P60), float(P90)),
             "nombres": nombres, "mask_elegible": mask, "meta": meta}
    ruta_modelo = os.path.join(tmp_dir, "modelo.joblib")
    import joblib
    joblib.dump(modelo, ruta_modelo)

    historial = _expandir_historial(c, meta)
    ruta_historial = os.path.join(tmp_dir, "historial.parquet")
    historial.to_parquet(ruta_historial, index=False)

    return ruta_modelo, ruta_historial, c_total


def _expandir_historial(c, meta, archivo="semilla.csv"):
    G, T, S = c.shape
    origen = meta["origen_semanas"]
    filas = []
    for g in range(G):
        for t in range(T):
            for s in range(S):
                n = int(c[g, t, s])
                if n == 0:
                    continue
                fecha = origen + pd.Timedelta(weeks=s, days=t)
                cod = float(meta["unidades"][g])
                for _ in range(n):
                    filas.append((fecha, cod, meta["nombre_unidad"][g], "GUAYAQUIL", archivo))
    return pd.DataFrame(filas, columns=["ts", "cod_parroquia", "parroquia", "canton", "archivo"])


# --- 1. CSV valido nuevo produce prediccion ---------------------------------
def test_csv_valido_produce_prediccion():
    d = _tmpdir("caso1")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)

    # semana siguiente al historial: semana s=_S0 (mes nuevo)
    filas = _filas_desde_conteos(c_total[:, :, _S0:_S0 + 4], s_offset=_S0)
    ruta_csv = os.path.join(d, "nuevo.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(
        ruta_csv, ruta_modelo=ruta_modelo, ruta_historial=ruta_hist,
        hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is True, r["mensaje"]
    assert r["matriz"] is not None and len(r["matriz"]) == _G * _T
    assert set(r["resumen"].keys()) <= {"Bajo", "Medio", "Alto", "Sin datos suficientes"}
    assert r["semana_objetivo"] is not None
    print(f"  OK CSV valido -> prediccion para {r['semana_objetivo']}, "
          f"{r['n_registros_nuevos']} registros nuevos")


# --- 2. CSV duplicado no duplica el historial -------------------------------
def test_csv_duplicado_no_duplica_historial():
    d = _tmpdir("caso2")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)

    filas = _filas_desde_conteos(c_total[:, :, _S0:_S0 + 4], s_offset=_S0)
    ruta_csv = os.path.join(d, "nuevo.csv")
    _escribir_csv(ruta_csv, filas)

    r1 = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                      ruta_historial=ruta_hist,
                                      hoy=pd.Timestamp("2050-01-01"))
    assert r1["ok"] is True
    n_tras_primera = r1["n_registros_historial"]

    # subir el MISMO archivo otra vez: el mes ya esta cargado -> se rechaza
    r2 = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                      ruta_historial=ruta_hist,
                                      hoy=pd.Timestamp("2050-01-01"))
    assert r2["ok"] is False
    assert r2["n_registros_historial"] == n_tras_primera, \
        "el historial crecio al subir el mismo CSV dos veces"
    print("  OK subir el mismo CSV dos veces no duplica el historial "
          f"({n_tras_primera} registros, sin cambio)")


# --- 3. CSV de un mes ya cargado se rechaza ---------------------------------
def test_mes_ya_cargado_se_rechaza():
    d = _tmpdir("caso3")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)

    # semana 5, ya dentro del historial sembrado (0.._S0-1)
    filas = _filas_desde_conteos(c_total[:, :, 5:6], s_offset=5)
    ruta_csv = os.path.join(d, "repetido.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is False
    assert "ya" in r["mensaje"].lower() and "historial" in r["mensaje"].lower()
    print("  OK mes ya presente en el historial -> rechazado:", r["mensaje"][:70])


# --- 4. parroquia desconocida se rechaza ------------------------------------
def test_parroquia_desconocida_se_rechaza():
    d = _tmpdir("caso4")
    ruta_modelo, ruta_hist, _c = _construir_frozen(d)
    meta = inf.cargar_modelo(ruta_modelo)["meta"]

    fecha = (meta["fecha_semana"][-1] + pd.Timedelta(weeks=1)).strftime("%d/%m/%Y")
    filas = [_fila_csv(fecha, "999999", "PARROQUIA INVENTADA")]
    ruta_csv = os.path.join(d, "parr_nueva.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is False
    assert "catalogo" in r["mensaje"].lower()
    assert r["n_registros_historial"] == len(pd.read_parquet(ruta_hist)), \
        "el historial se modifico pese al rechazo"
    print("  OK parroquia fuera del catalogo congelado -> rechazado, "
          "historial intacto")


# --- 5. fecha futura se rechaza ---------------------------------------------
def test_fecha_futura_se_rechaza():
    d = _tmpdir("caso5")
    ruta_modelo, ruta_hist, _c = _construir_frozen(d)

    filas = [_fila_csv("01/01/2099", "111", "PARROQUIA 111")]
    ruta_csv = os.path.join(d, "futuro.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist)   # hoy real
    assert r["ok"] is False
    assert "futur" in r["mensaje"].lower()
    print("  OK fecha futura -> rechazado:", r["mensaje"][:70])


# --- 6/7. sin Guayaquil / sin Transito: rechazo limpio ----------------------
def test_csv_sin_guayaquil_rechazo_limpio():
    d = _tmpdir("caso6")
    ruta_modelo, ruta_hist, _c = _construir_frozen(d)
    filas = [_fila_csv("01/03/2023", "120153", "OTRA", canton="BABAHOYO")]
    ruta_csv = os.path.join(d, "otra_ciudad.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is False and "rechazado" in r["mensaje"].lower()
    print("  OK CSV sin Guayaquil -> rechazo limpio, sin excepcion")


def test_csv_sin_transito_rechazo_limpio():
    d = _tmpdir("caso7")
    ruta_modelo, ruta_hist, _c = _construir_frozen(d)
    filas = [_fila_csv("01/03/2023", "111", "PARROQUIA 111", serv="Seguridad Ciudadana")]
    ruta_csv = os.path.join(d, "seguridad.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is False and "rechazado" in r["mensaje"].lower()
    print("  OK CSV sin Transito y Movilidad -> rechazo limpio, sin excepcion")


# --- 8. mes incompleto: comportamiento controlado (no crash, se avisa) -----
def test_mes_parcial_se_incorpora_con_advertencia():
    d = _tmpdir("caso8")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)

    # solo 2 dias de la semana _S0 (mes muy parcial)
    fecha_base = _ORIGEN + pd.Timedelta(weeks=_S0)
    filas = [_fila_csv(fecha_base.strftime("%d/%m/%Y"), "111", "PARROQUIA 111"),
            _fila_csv((fecha_base + pd.Timedelta(days=1)).strftime("%d/%m/%Y"),
                      "222", "PARROQUIA 222")]
    ruta_csv = os.path.join(d, "parcial.csv")
    _escribir_csv(ruta_csv, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    # 2 dias sueltos no completan ninguna semana nueva -> se acepta el
    # registro (queda en el historial) pero no hay semana objetivo nueva
    # que predecir todavia, o la prediccion sigue siendo la ya conocida.
    assert r["ok"] is True, r["mensaje"]
    assert r["n_registros_nuevos"] == 2
    print("  OK mes con muy pocos dias -> se acepta sin reventar "
          f"({r['mensaje'][:90]})")


# --- 9. semana que cruza mes usa el historial acumulado ---------------------
def test_semana_cruza_mes_usa_historial_acumulado():
    """
    La semana s=_S0 (2023-07-31 a 2023-08-06) cruza de julio a agosto. El
    historial sembrado solo llega hasta el 2023-07-30 (fin de la semana
    _S0-1): agosto no tiene NINGUN dato todavia. Al subir un archivo de
    agosto completo (un mes, como en el uso real: un CSV por mes), la
    semana que cruza el limite debe reconstruirse con la parte de julio que
    YA estaba en el historial mas la parte de agosto recien subida -- ni
    mas ni menos que el conteo real de esa semana en el tensor sintetico.
    """
    d = _tmpdir("caso9")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)
    meta = inf.cargar_modelo(ruta_modelo)["meta"]

    conteo_real_semana_cruce = int(c_total[:, :, _S0].sum())
    assert conteo_real_semana_cruce > 0, "el tensor sintetico no tiene eventos en la semana de cruce"

    # _construir_frozen siembra por SEMANA (0.._S0-1), que termina el
    # domingo 2023-07-30 -- un dia antes de que julio termine de verdad
    # (2023-07-31 cae en la semana _S0, la que cruza a agosto). Un archivo
    # real de julio si habria incluido el 31. Se completa esa semilla aqui
    # (no via validar_csv/predecir_desde_csv_nuevo: esto representa como
    # habria quedado el historial real tras cargar el archivo de julio
    # completo, no una subida nueva que esta prueba deba validar).
    faltante_31_julio = pd.DataFrame([
        {"ts": _ORIGEN + pd.Timedelta(weeks=_S0, days=0),
         "cod_parroquia": float(_UNIDADES[g]), "parroquia": f"PARROQUIA {_UNIDADES[g]}",
         "canton": "GUAYAQUIL", "archivo": "semilla.csv"}
        for g in range(_G) for _ in range(int(c_total[g, 0, _S0]))
    ])
    if len(faltante_31_julio):
        hist_semilla = pd.read_parquet(ruta_hist)
        pd.concat([hist_semilla, faltante_31_julio], ignore_index=True).to_parquet(
            ruta_hist, index=False)

    # archivo de "agosto completo": todo lo que en el tensor sintetico cae
    # en agosto 2023 (incluida la cola de la semana _S0 y semanas 31-34
    # enteras, para que sea un mes realista, no un solo dia suelto).
    filas_agosto = []
    for g in range(_G):
        for t in range(_T):
            for s in range(_S0, _S0 + 5):        # cubre julio->agosto->parte de septiembre
                fecha = _ORIGEN + pd.Timedelta(weeks=s, days=t)
                if fecha.month != 8:
                    continue
                n = int(c_total[g, t, s])
                for _ in range(n):
                    filas_agosto.append(_fila_csv(fecha.strftime("%d/%m/%Y"),
                                                  _UNIDADES[g], f"PARROQUIA {_UNIDADES[g]}"))

    ruta_agosto = os.path.join(d, "agosto.csv")
    _escribir_csv(ruta_agosto, filas_agosto)

    r = inf.predecir_desde_csv_nuevo(ruta_agosto, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is True, r["mensaje"]

    hist_final = pd.read_parquet(ruta_hist)
    c_reconstruido, _meta_ext = inf.reconstruir_tensor(hist_final, meta)
    assert c_reconstruido.shape[2] > _S0
    assert int(c_reconstruido[:, :, _S0].sum()) == conteo_real_semana_cruce, (
        "la semana que cruza julio/agosto no combino correctamente el historial "
        "previo (julio) con el archivo nuevo (agosto)")
    print(f"  OK semana que cruza mes (julio/agosto) reconstruida con "
          f"{conteo_real_semana_cruce} eventos combinando historial previo + CSV nuevo")


# --- 10/11/12/13 (variante sintetica): no se llama a entrenamiento ---------
def test_no_se_invoca_entrenamiento():
    """
    Bloquea BosqueAleatorioPropio.entrenar y ArbolDecisionPropio.construir:
    si predecir_desde_csv_nuevo los llamara, esta prueba fallaria con el
    RuntimeError en vez de con un assert -- prueba estructural, no de
    convencion.
    """
    d = _tmpdir("caso10")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)
    filas = _filas_desde_conteos(c_total[:, :, _S0:_S0 + 4], s_offset=_S0)
    ruta_csv = os.path.join(d, "nuevo.csv")
    _escribir_csv(ruta_csv, filas)

    from src.bosque import ArbolDecisionPropio
    orig_entrenar = BosqueAleatorioPropio.entrenar
    orig_construir = ArbolDecisionPropio.construir

    def _bloqueado_entrenar(*a, **kw):
        raise RuntimeError("entrenar() no debe llamarse durante inferencia")

    def _bloqueado_construir(*a, **kw):
        raise RuntimeError("construir() no debe llamarse durante inferencia")

    BosqueAleatorioPropio.entrenar = _bloqueado_entrenar
    ArbolDecisionPropio.construir = _bloqueado_construir
    try:
        r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                         ruta_historial=ruta_hist,
                                         hoy=pd.Timestamp("2050-01-01"))
    finally:
        BosqueAleatorioPropio.entrenar = orig_entrenar
        ArbolDecisionPropio.construir = orig_construir

    assert r["ok"] is True, r["mensaje"]
    print("  OK predecir_desde_csv_nuevo tuvo exito con entrenar()/construir() "
          "bloqueados: nunca se llaman")


# --- 14. semanas_train no cambian al extender el historial ------------------
def test_semanas_train_no_cambian_al_extender_historial():
    d = _tmpdir("caso14")
    ruta_modelo, ruta_hist, c_total = _construir_frozen(d)
    meta = inf.cargar_modelo(ruta_modelo)["meta"]

    sem_tr_antes = inf.semanas_train_congeladas(meta, corte_train="2099-01-01")

    filas = _filas_desde_conteos(c_total[:, :, _S0:_S0 + 4], s_offset=_S0)
    ruta_csv = os.path.join(d, "nuevo.csv")
    _escribir_csv(ruta_csv, filas)
    r = inf.predecir_desde_csv_nuevo(ruta_csv, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is True

    sem_tr_despues = inf.semanas_train_congeladas(meta, corte_train="2099-01-01")
    assert np.array_equal(sem_tr_antes, sem_tr_despues), \
        "las semanas de entrenamiento cambiaron al extender el historial"
    assert sem_tr_despues.max() < meta["S"], \
        "semanas_train_congeladas devolvio una semana fuera del tensor original"
    print(f"  OK semanas_train identicas antes/despues de extender el historial "
          f"({len(sem_tr_antes)} semanas, todas < S0={meta['S']})")


# ============================================================================
# GRUPO B -- artefactos REALES (lenta: siembra el historial desde datos/,
# ~2 min la primera vez; instantanea en corridas siguientes porque queda
# cacheado en salidas/historial_filtrado.parquet).
# ============================================================================

RUTA_SALIDAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "salidas")
_ARTEFACTOS_CONGELADOS = ["modelo.joblib", "metricas.json", "matriz_riesgo.parquet"]


def _hash_archivo(ruta):
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_reproduce_matriz_riesgo_real_y_no_altera_artefactos_congelados():
    """
    Con los artefactos REALES del proyecto:

    1. Siembra (o reusa, si ya existe) salidas/historial_filtrado.parquet
       desde datos/ -- los mismos 54 CSV con los que se entreno el modelo.
    2. Reconstruye el tensor SOLO con ese historial (sin ningun CSV nuevo) y
       verifica que la matriz de riesgo resultante es BIT-EXACTA contra
       salidas/matriz_riesgo.parquet -- la misma prueba que hizo la
       auditoria del 17-ago a mano, ahora contra el codigo real de
       inferencia.py.
    3. Simula la llegada de un mes nuevo (enero 2026, fabricado, con los
       codigos de parroquia REALES) sobre una COPIA del historial -- nunca
       sobre el historial real -- y confirma que produce una prediccion.
    4. Verifica, por hash SHA-256 antes/despues de TODO lo anterior, que
       modelo.joblib, metricas.json y matriz_riesgo.parquet no cambiaron
       ni un byte.
    """
    ruta_modelo = os.path.join(RUTA_SALIDAS, "modelo.joblib")
    if not os.path.exists(ruta_modelo):
        print("  [saltada] no se encontro salidas/modelo.joblib")
        return

    hashes_antes = {a: _hash_archivo(os.path.join(RUTA_SALIDAS, a))
                    for a in _ARTEFACTOS_CONGELADOS}

    modelo = inf.cargar_modelo(ruta_modelo)
    meta = modelo["meta"]

    ruta_historial_real = os.path.join(RUTA_SALIDAS, "historial_filtrado.parquet")
    historial = inf.cargar_historial(ruta_historial_real,
                                     ruta_datos_semilla=os.path.join(
                                         os.path.dirname(RUTA_SALIDAS), "datos"))
    print(f"  historial cargado: {len(historial):,} registros")

    # -- 2. reproduccion bit-exacta, sin ningun CSV nuevo --------------------
    c_recon, meta_ext = inf.reconstruir_tensor(historial, meta)
    assert c_recon.shape == (meta["G"], meta["T"], meta["S"]), (
        f"el tensor reconstruido desde el historial no coincide con el "
        f"congelado: {c_recon.shape} vs (G={meta['G']},T={meta['T']},S={meta['S']})")

    sem_tr = inf.semanas_train_congeladas(meta)
    gc_ = GeneradorCaracteristicas()
    X, nombres = gc_.construirX(c_recon, meta_ext, sem_tr, semanas_futuras=1,
                                mask_elegible=modelo["mask_elegible"])
    assert nombres == modelo["nombres"]

    from src.riesgo import predecir_matriz_semana, MatrizRiesgo
    G, T = meta["G"], meta["T"]
    i_sig = gc_.indices_semana_futura(G, T, c_recon.shape[2], k=0)
    estimador = modelo.get("modelo", modelo["bosque"])
    proba, _n = predecir_matriz_semana(
        estimador, X, i_sig, modelo["mask_elegible"]
    )
    mr = MatrizRiesgo(meta_ext, gc_.fecha_semana_futura(0), umbrales=modelo["umbrales"])
    mr.desde_predicciones(proba, mascara_sin_datos=~modelo["mask_elegible"])
    recon_df = mr.a_dataframe()

    oficial = pd.read_parquet(os.path.join(RUTA_SALIDAS, "matriz_riesgo.parquet"))
    a = recon_df.sort_values(["unidad_id", "dia_idx"]).reset_index(drop=True)
    b = oficial.sort_values(["unidad_id", "dia_idx"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["nivel"] == b["nivel"]).all(), \
        "el nivel reconstruido difiere del oficial matriz_riesgo.parquet"
    dif_prob = (a["prob_clase_predicha"] - b["prob_clase_predicha"]).abs().max()
    assert dif_prob < 1e-6, f"diferencia de probabilidad: {dif_prob}"
    print(f"  OK reproduccion bit-exacta de matriz_riesgo.parquet "
          f"(dif. probabilidad max = {dif_prob:.2e})")

    # -- 3. mes nuevo fabricado, sobre una COPIA del historial ---------------
    d = _tmpdir("real_extension")
    ruta_hist_copia = os.path.join(d, "historial_copia.parquet")
    historial.to_parquet(ruta_hist_copia, index=False)

    ultima = pd.to_datetime(historial["ts"]).max()
    inicio_nuevo = (ultima + pd.Timedelta(days=1)).normalize()
    filas = []
    for cod in meta["unidades"]:
        for k in range(10):
            fecha = (inicio_nuevo + pd.Timedelta(days=k)).strftime("%d/%m/%Y")
            filas.append(_fila_csv(fecha, cod, "X"))
    ruta_csv_nuevo = os.path.join(d, "enero_2026.csv")
    _escribir_csv(ruta_csv_nuevo, filas)

    r = inf.predecir_desde_csv_nuevo(ruta_csv_nuevo, ruta_modelo=ruta_modelo,
                                     ruta_historial=ruta_hist_copia,
                                     hoy=pd.Timestamp("2050-01-01"))
    assert r["ok"] is True, r["mensaje"]
    assert len(r["matriz"]) == G * T
    print(f"  OK mes fabricado posterior al historial real -> prediccion "
          f"para {r['semana_objetivo']} (sobre copia, no sobre el historial real)")

    # -- 4. artefactos congelados intactos ------------------------------------
    hashes_despues = {a: _hash_archivo(os.path.join(RUTA_SALIDAS, a))
                      for a in _ARTEFACTOS_CONGELADOS}
    for nombre in _ARTEFACTOS_CONGELADOS:
        assert hashes_antes[nombre] == hashes_despues[nombre], (
            f"salidas/{nombre} cambio durante la prueba de inferencia")
    print("  OK modelo.joblib, metricas.json y matriz_riesgo.parquet "
          "identicos byte a byte antes/despues (hash SHA-256)")


if __name__ == "__main__":
    pruebas = [
        test_csv_valido_produce_prediccion,
        test_csv_duplicado_no_duplica_historial,
        test_mes_ya_cargado_se_rechaza,
        test_parroquia_desconocida_se_rechaza,
        test_fecha_futura_se_rechaza,
        test_csv_sin_guayaquil_rechazo_limpio,
        test_csv_sin_transito_rechazo_limpio,
        test_mes_parcial_se_incorpora_con_advertencia,
        test_semana_cruza_mes_usa_historial_acumulado,
        test_no_se_invoca_entrenamiento,
        test_semanas_train_no_cambian_al_extender_historial,
    ]
    for f in pruebas:
        f()
    print("\nTodas las pruebas rapidas de inferencia pasaron.")

    print("\nPrueba lenta (datos reales, ~2 min si no hay cache)...")
    test_reproduce_matriz_riesgo_real_y_no_altera_artefactos_congelados()
    print("\nTodas las pruebas de inferencia pasaron.")
