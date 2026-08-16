"""
Pruebas de la representacion parroquia x dia de semana.

Las 19 pruebas anteriores siguen pasando porque operan sobre el tensor
c[g, t, s], cuya forma no cambio al migrar de celda/franja a parroquia/dia. Por
eso mismo NO validan la nueva interpretacion: estas si.

    python -m tests.test_representacion
"""
import sys
import os
import atexit
import shutil
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocesamiento import Preprocesador, T_DIAS, DIAS as DIAS_T, normalizar
from src.caracteristicas import (GeneradorCaracteristicas, Discretizador,
                                 FERIADOS_FIJOS, pascua, feriados_del_anio)
from src.riesgo import predecir_matriz_semana, MatrizRiesgo

CSV = """Fecha;provincia;Canton;Cod_Parroquia;Parroquia;Servicio;Subtipo
{filas}"""


def _escribir_csv(tmp, filas):
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(CSV.format(filas="\n".join(filas)))


def _fila(fecha, canton="GUAYAQUIL", cod=90150, parr="CABECERA",
          serv="Tránsito y Movilidad", prov="GUAYAS"):
    return f"{fecha};{prov};{canton};{cod};{parr};{serv};Choque"


_TEMPORALES = []


def _tmpdir(nombre):
    """
    Directorio temporal UNICO por llamada.

    Una ruta fija como /tmp/_test_t7 no es hermetica: si queda un archivo de
    otra ejecucion, o de otro usuario, la prueba falla con PermissionError por
    algo que no tiene que ver con lo que intenta verificar. mkdtemp crea un
    directorio nuevo, propiedad de quien corre la suite, y se limpia al salir.
    """
    d = tempfile.mkdtemp(prefix=f"ecu911_{nombre}_")
    _TEMPORALES.append(d)
    return d


@atexit.register
def _limpiar_temporales():
    for d in _TEMPORALES:
        shutil.rmtree(d, ignore_errors=True)


# --- estructura del tensor --------------------------------------------------
def test_T_es_siete_dias_de_semana():
    """El dataset no publica hora: la unidad temporal es el dia, |T| = 7."""
    assert T_DIAS == 7
    d = _tmpdir("t7")
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026"), _fila("11/1/2026")])
    c, meta = Preprocesador(descartar_semanas_incompletas=False).ejecutar(d)
    assert c.shape[1] == 7, c.shape
    assert meta["T"] == 7
    print("  OK |T| = 7 dias de semana (no 56 franjas de 3 h)")


def test_lunes_es_cero_y_domingo_es_seis():
    d = _tmpdir("dow")
    # 5/1/2026 es lunes; 11/1/2026 es domingo
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026"), _fila("11/1/2026")])
    c, meta = Preprocesador(descartar_semanas_incompletas=False).ejecutar(d)
    assert pd.Timestamp("2026-01-05").dayofweek == 0
    assert pd.Timestamp("2026-01-11").dayofweek == 6
    assert c[:, 0, :].sum() == 1, "el lunes debe caer en t=0"
    assert c[:, 6, :].sum() == 1, "el domingo debe caer en t=6"
    assert meta["dias"][0] == "Lunes" and meta["dias"][6] == "Domingo"
    print("  OK lunes -> t=0, domingo -> t=6")


def test_la_semana_agrupa_de_lunes_a_domingo():
    d = _tmpdir("sem")
    # lunes 5 y domingo 11 son la MISMA semana; lunes 12 es la siguiente
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026"), _fila("11/1/2026"),
                                 _fila("12/1/2026")])
    c, meta = Preprocesador(descartar_semanas_incompletas=False).ejecutar(d)
    assert c[:, :, 0].sum() == 2, "lunes y domingo de la misma semana"
    assert c[:, :, 1].sum() == 1, "el lunes siguiente abre semana nueva"
    assert pd.Timestamp(meta["fecha_semana"][0]).dayofweek == 0
    print("  OK la semana va de lunes a domingo y el origen es un lunes")


# --- filtros ----------------------------------------------------------------
def test_filtro_exacto_de_canton_y_categoria():
    d = _tmpdir("filtro")
    _escribir_csv(f"{d}/a.csv", [
        _fila("5/1/2026"),                                        # si
        _fila("5/1/2026", canton="DURAN", cod=90250),             # otro canton
        _fila("5/1/2026", serv="Seguridad Ciudadana"),            # otra categoria
        _fila("6/1/2026", canton="GUAYAQUIL", cod=90152, parr="PROGRESO"),
    ])
    c, meta = Preprocesador(alcance=("canton", "GUAYAQUIL"),
                            descartar_semanas_incompletas=False).ejecutar(d)
    assert int(c.sum()) == 2, f"debian quedar 2 registros, quedaron {int(c.sum())}"
    assert meta["G"] == 2, "dos parroquias distintas de Guayaquil"
    print("  OK filtro exacto: canton GUAYAQUIL + categoria transito")


def test_alcance_por_provincia_incluye_otros_cantones():
    d = _tmpdir("prov")
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026"),
                                 _fila("5/1/2026", canton="DURAN", cod=90250)])
    c, _ = Preprocesador(alcance=("provincia", "GUAYAS"),
                         descartar_semanas_incompletas=False).ejecutar(d)
    assert int(c.sum()) == 2
    c2, _ = Preprocesador(alcance=("canton", "GUAYAQUIL"),
                          descartar_semanas_incompletas=False).ejecutar(d)
    assert int(c2.sum()) == 1
    print("  OK el alcance es un parametro: canton vs provincia")


def test_no_falla_por_ausencia_de_coordenadas_y_hora():
    """El dataset real no trae latitud, longitud ni hora."""
    d = _tmpdir("sincoord")
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026")])
    c, meta = Preprocesador(descartar_semanas_incompletas=False).ejecutar(d)
    assert c.sum() == 1
    assert "centro_utm" not in meta and "pares_ixiy" not in meta
    print("  OK el pipeline no requiere coordenadas ni hora")


def test_detecta_semanas_sin_datos():
    """Meses no contiguos dejan semanas vacias que envenenan rezagos y umbrales."""
    d = _tmpdir("huecos")
    _escribir_csv(f"{d}/a.csv", [_fila("5/1/2026"), _fila("2/3/2026")])
    c, meta = Preprocesador(descartar_semanas_incompletas=False).ejecutar(d)
    con_datos = meta["semanas_con_datos"]
    assert len(con_datos) == 2 and meta["S"] > 5, (len(con_datos), meta["S"])
    assert (c.sum(axis=(0, 1)) == 0).sum() == meta["S"] - 2
    print(f"  OK detecta {meta['S'] - 2} semanas vacias entre meses no contiguos")


# --- calendario por (t, s) --------------------------------------------------
def _gen(fechas_lunes, T=7):
    G, S = 1, len(fechas_lunes)
    c = np.ones((G, T, S), dtype=np.int32)
    meta = {"fecha_semana": fechas_lunes, "S": S}
    gc = GeneradorCaracteristicas()
    X, nom = gc.construirX(c, meta, np.arange(S), semanas_futuras=0)
    return X.reshape(G, T, gc.S_total, -1), nom


def test_mes_cambia_dentro_de_una_semana():
    """
    Regresion: mes y es_feriado se calculaban con la fecha del lunes y se
    replicaban a los siete dias. La semana del lunes 30-nov-2026 termina en
    diciembre, y 6 de sus 7 dias quedaban con el mes equivocado.
    """
    lunes = [pd.Timestamp("2026-11-30"), pd.Timestamp("2026-12-07"),
             pd.Timestamp("2026-12-14")]
    X, nom = _gen(lunes)
    mes = X[0, :, 0, nom.index("mes")]
    assert mes[0] != mes[1], "el 30-nov debe separarse del 1-dic"
    assert len(set(mes[1:].tolist())) == 1, "del martes al domingo todos son diciembre"
    print("  OK el mes se calcula por dia: 30-nov se separa de diciembre")


def test_feriado_marca_solo_el_dia_exacto():
    """El 2 y 3 de noviembre son feriados; el resto de esa semana no lo es."""
    lunes = [pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-09")]
    X, nom = _gen(lunes)
    fer = X[0, :, 0, nom.index("es_feriado")]
    assert fer[0] == fer[1], "lunes 2 y martes 3 son feriado"
    assert fer[0] != fer[2], "el miercoles 4 NO es feriado"
    assert len(set(fer[2:].tolist())) == 1
    assert (11, 2) in FERIADOS_FIJOS
    print("  OK el feriado marca solo su fecha, no la semana completa")


def test_calendario_incluye_feriados_moviles():
    """
    Carnaval y Viernes Santo son moviles: se derivan de la Pascua, no se
    tabulan. Domingos de Pascua verificados contra el calendario oficial.
    """
    esperado = {2021: "2021-04-04", 2022: "2022-04-17", 2023: "2023-04-09",
                2024: "2024-03-31", 2025: "2025-04-20"}
    for anio, fecha in esperado.items():
        assert pascua(anio) == pd.Timestamp(fecha), (anio, pascua(anio))

    f2025 = feriados_del_anio(2025)
    assert pd.Timestamp("2025-03-03") in f2025, "Carnaval lunes 2025"
    assert pd.Timestamp("2025-03-04") in f2025, "Carnaval martes 2025"
    assert pd.Timestamp("2025-04-18") in f2025, "Viernes Santo 2025"
    assert len(f2025) == len(FERIADOS_FIJOS) + 3
    print(f"  OK Pascua correcta 2021-2025; {len(f2025)} feriados por anio "
          "(8 fijos + 3 moviles)")


def test_carnaval_marcado_en_las_caracteristicas():
    """Carnaval 2025 cae lunes 3 y martes 4 de marzo: deben salir como feriado."""
    lunes = [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-10")]
    X, nom = _gen(lunes)
    fer = X[0, :, 0, nom.index("es_feriado")]
    assert fer[0] == fer[1], "lunes y martes de Carnaval son feriado"
    assert fer[0] != fer[2], "el miercoles de ceniza no lo es"
    print("  OK Carnaval 2025 (3 y 4 de marzo) queda marcado como feriado")


# --- discretizacion ---------------------------------------------------------
def test_bins_no_colapsan_variables_de_baja_cardinalidad():
    """
    Regresion: el discretizador usaba side="right" con bordes = valores unicos
    menos el maximo, de modo que los dos valores mas altos caian en el mismo
    bin. Variables binarias como es_feriado quedaban CONSTANTES.
    """
    for valores in ([0, 1], [11, 12], [0, 1, 2], [0, 1, 2, 3]):
        v = np.array(valores)
        disc = Discretizador()
        bordes = disc.ajustar_columna(v)
        bins = Discretizador.aplicar(v, bordes)
        assert len(set(bins.tolist())) == len(valores), (valores, bins.tolist())
    print("  OK los bins conservan todos los valores distintos (side='left')")


def test_variables_binarias_no_quedan_constantes():
    lunes = [pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=i) for i in range(6)]
    X, nom = _gen(lunes)
    fds = X[0, :, :, nom.index("es_fin_de_semana")]
    assert len(np.unique(fds)) == 2, "es_fin_de_semana debe tomar dos valores"
    dia = X[0, :, 0, nom.index("dia_semana")]
    assert len(set(dia.tolist())) == 7, "los 7 dias deben ser distinguibles"
    print("  OK es_fin_de_semana tiene 2 valores y dia_semana los 7")


# --- caracteristicas espaciales ---------------------------------------------
def test_no_hay_caracteristicas_de_vecindad_artificial():
    """
    Pertenecer al mismo canton no implica adyacencia geografica: Puna es una
    isla y Tenguel esta al extremo sur. Se prescinde de vecindad en lugar de
    fabricar una proximidad indefendible.
    """
    lunes = [pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=i) for i in range(4)]
    _, nom = _gen(lunes)
    prohibidas = {"vecinos_rezago_1", "vecinos_media_movil_4", "n_vecinos",
                  "densidad_historica_canton", "centroide_x", "centroide_y"}
    assert not (prohibidas & set(nom)), prohibidas & set(nom)
    assert any(n.startswith("densidad_historica_") for n in nom)
    print(f"  OK {len(nom)} variables, ninguna de vecindad artificial ni de coordenadas")


def test_descarta_semanas_incompletas_al_final():
    """
    Regresion: si los datos terminan a mitad de semana, los dias posteriores no
    existen en la fuente pero el tensor los contaba como cero, generando
    etiquetas "bajo" falsas que contaminaban la evaluacion y desplazaban la
    semana objetivo una semana hacia adelante.

    Caso: datos hasta el miercoles 31/12/2025. La semana del lunes 29/12 solo
    tiene 3 de sus 7 dias, asi que debe descartarse y la ultima semana completa
    pasa a ser la del 22/12.
    """
    d = _tmpdir("incompleta")
    filas = []
    # cinco semanas completas, del 24/11 al 28/12
    for dia in pd.date_range("2025-11-24", "2025-12-28"):
        filas.append(_fila(dia.strftime("%d/%m/%Y")))
    # semana parcial: solo lunes, martes y miercoles
    for dia in pd.date_range("2025-12-29", "2025-12-31"):
        filas.append(_fila(dia.strftime("%d/%m/%Y")))
    _escribir_csv(f"{d}/a.csv", filas)

    c, meta = Preprocesador().ejecutar(d)
    assert meta["semanas_incompletas_descartadas"] == 1, meta["semanas_incompletas_descartadas"]
    ultima = pd.Timestamp(meta["fecha_semana"][-1])
    assert ultima == pd.Timestamp("2025-12-22"), ultima
    # los 3 dias del 29, 30 y 31 no deben aparecer en el tensor
    assert int(c.sum()) == 35, int(c.sum())
    print("  OK la semana parcial del 29/12 se descarta; ultima completa 22/12")


def test_semana_completa_al_borde_no_se_descarta():
    """Si los datos terminan justo un domingo, ninguna semana se descarta."""
    d = _tmpdir("borde")
    filas = [_fila(x.strftime("%d/%m/%Y"))
             for x in pd.date_range("2025-12-01", "2025-12-28")]
    _escribir_csv(f"{d}/a.csv", filas)
    c, meta = Preprocesador().ejecutar(d)
    assert meta["semanas_incompletas_descartadas"] == 0
    assert pd.Timestamp(meta["fecha_semana"][-1]) == pd.Timestamp("2025-12-22")
    assert int(c.sum()) == 28
    print("  OK datos que terminan en domingo: no se descarta ninguna semana")


# --- prediccion de la semana objetivo ---------------------------------------
class _BosqueEspia:
    """Registra cuantas filas recibe realmente predecirProba."""

    def __init__(self):
        self.filas_vistas = 0

    def predecirProba(self, X):
        self.filas_vistas += len(X)
        p = np.full((len(X), 3), 1 / 3, dtype=np.float32)
        return p


def test_solo_las_unidades_elegibles_pasan_por_el_modelo():
    """
    La Tarea #4 dice que las unidades sin historial no ingresan al modelo.
    Predecir las G*T combinaciones y pintar despues en gris las no elegibles
    daria el mismo resultado visual, pero haria falsa esa afirmacion.
    """
    G, T = 6, 7
    mask = np.zeros((G, T), dtype=bool)
    mask[:4] = True                      # 4 unidades elegibles de 6
    X = np.zeros((G * T, 3), dtype=np.uint8)
    indices = np.arange(G * T)

    espia = _BosqueEspia()
    proba, n = predecir_matriz_semana(espia, X, indices, mask)

    assert n == int(mask.sum()) == 4 * T, (n, mask.sum())
    assert espia.filas_vistas == n, (espia.filas_vistas, n)
    assert espia.filas_vistas != G * T, "no debe predecir toda la matriz"
    assert np.isnan(proba[~mask.reshape(-1)]).all(), "las no elegibles quedan en NaN"
    assert not np.isnan(proba[mask.reshape(-1)]).any()
    print(f"  OK el modelo ve {n} filas elegibles, no las {G*T} de la matriz completa")


def test_las_unidades_sin_datos_no_muestran_probabilidad():
    """Una unidad que no paso por el modelo no tiene distribucion estimada."""
    G, T = 3, 7
    mask = np.zeros((G, T), dtype=bool)
    mask[0] = True
    meta = {"G": G, "T": T, "unidades": np.array(["a", "b", "c"]),
            "nombre_unidad": np.array(["A", "B", "C"]),
            "canton_unidad": np.array(["X", "X", "X"]), "dias": DIAS_T}
    proba, _ = predecir_matriz_semana(_BosqueEspia(), np.zeros((G * T, 3), np.uint8),
                                      np.arange(G * T), mask)
    mr = MatrizRiesgo(meta, pd.Timestamp("2026-01-05")).desde_predicciones(
        proba, mascara_sin_datos=~mask)
    df = mr.a_dataframe()
    sd = df[df["nivel"] == "Sin datos suficientes"]
    assert len(sd) == (G - 1) * T
    for col in ("prob_clase_predicha", "p_bajo", "p_medio", "p_alto"):
        assert sd[col].isna().all(), f"{col} deberia ser NaN en las unidades sin datos"
    print("  OK las unidades sin datos no reportan ninguna probabilidad")


if __name__ == "__main__":
    import io
    import contextlib
    grupos = [
        ("Estructura del tensor (parroquia x dia x semana)",
         [test_T_es_siete_dias_de_semana, test_lunes_es_cero_y_domingo_es_seis,
          test_la_semana_agrupa_de_lunes_a_domingo]),
        ("Filtros y tolerancia a la fuente real",
         [test_filtro_exacto_de_canton_y_categoria,
          test_alcance_por_provincia_incluye_otros_cantones,
          test_no_falla_por_ausencia_de_coordenadas_y_hora,
          test_detecta_semanas_sin_datos]),
        ("Calendario por (t, s), no por semana",
         [test_mes_cambia_dentro_de_una_semana,
          test_feriado_marca_solo_el_dia_exacto,
          test_calendario_incluye_feriados_moviles,
          test_carnaval_marcado_en_las_caracteristicas]),
        ("Discretizacion",
         [test_bins_no_colapsan_variables_de_baja_cardinalidad,
          test_variables_binarias_no_quedan_constantes]),
        ("Caracteristicas espaciales",
         [test_no_hay_caracteristicas_de_vecindad_artificial]),
        ("Cobertura temporal",
         [test_descarta_semanas_incompletas_al_final,
          test_semana_completa_al_borde_no_se_descarta]),
        ("Prediccion de la semana objetivo",
         [test_solo_las_unidades_elegibles_pasan_por_el_modelo,
          test_las_unidades_sin_datos_no_muestran_probabilidad]),
    ]
    for titulo, fs in grupos:
        print(f"\n{titulo}")
        for f in fs:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                f()
            print([l for l in buf.getvalue().splitlines() if l.startswith("  OK")][0])
    print("\nTodas las pruebas de representacion pasaron.")