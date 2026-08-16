"""
Pruebas de regresion de los fallos detectados en la revision del 15 de agosto.

    python -m tests.test_pipeline

Cada prueba corresponde a un fallo concreto, de modo que si alguien vuelve a
introducirlo el error aparece de inmediato.
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.caracteristicas import (GeneradorCaracteristicas, ConstructorEtiquetas,
                                 mascara_elegibilidad, indices_de_semanas,
                                 filtrar_elegibles)


def _tensor_exacto(G, T, S, props=(0.85, 0.05, 0.10), semilla=0):
    """
    Tensor con la composicion EXACTA pedida, no muestreada. Necesario cuando el
    test depende del valor concreto del percentil: con 85/5/10 exactos,
    np.percentile devuelve P60 = 0 y P90 = 1.1, que es el caso que se quiere
    verificar. Muestrear haria que el percentil saltara entre 1.1 y 2 segun la
    realizacion.
    """
    n = G * T * S
    k = [int(round(f * n)) for f in props[:-1]]
    k.append(n - sum(k))
    v = np.repeat(np.arange(len(k), dtype=np.int32), k)
    np.random.default_rng(semilla).shuffle(v)
    return v.reshape(G, T, S)


def _meta(G, T, S):
    return {"fecha_semana": [pd.Timestamp("2021-07-05") + pd.Timedelta(weeks=i)
                             for i in range(S)],
            "G": G, "T": T, "S": S}


# --- Fallo 1: se predecia la ultima semana observada, no la siguiente --------
def test_semana_futura_usa_rezagos_correctos():
    G, T, S = 3, 4, 20
    c = (np.arange(G * T * S, dtype=np.int32).reshape(G, T, S)) % 7
    gc = GeneradorCaracteristicas()
    gc.construirX(c, _meta(G, T, S), np.arange(12, 16), semanas_futuras=1)

    cpad = np.concatenate([c, np.zeros((G, T, 1), dtype=c.dtype)], axis=2)
    for k in (1, 2, 3, 4):
        r = GeneradorCaracteristicas._rezago(cpad, k)
        assert np.array_equal(r[:, :, S], c[:, :, S - k]), f"rezago_{k} incorrecto"
    mm = GeneradorCaracteristicas._media_movil(cpad, 4)
    assert np.allclose(mm[:, :, S], c[:, :, S - 4:S].mean(axis=2), atol=1e-5)
    assert gc.S_total == S + 1
    print("  OK la fila s=S usa c[S-1..S-4] como rezagos")


def test_semana_futura_no_depende_del_valor_desconocido():
    """Ninguna caracteristica de s=S puede leer c[S]: ese es el valor a predecir."""
    G, T, S = 3, 4, 20
    c = np.random.default_rng(0).integers(0, 5, (G, T, S)).astype(np.int32)
    a = np.concatenate([c, np.zeros((G, T, 1), dtype=c.dtype)], axis=2)
    b = a.copy()
    b[:, :, S] = 999

    for k in (1, 2, 3, 4):
        assert np.array_equal(GeneradorCaracteristicas._rezago(a, k)[:, :, S],
                              GeneradorCaracteristicas._rezago(b, k)[:, :, S])
    for w in (4, 12):
        assert np.allclose(GeneradorCaracteristicas._media_movil(a, w)[:, :, S],
                           GeneradorCaracteristicas._media_movil(b, w)[:, :, S])
    assert np.allclose(GeneradorCaracteristicas._densidad_acumulada(a)[:, :, S],
                       GeneradorCaracteristicas._densidad_acumulada(b)[:, :, S])
    print("  OK alterar c[S] no cambia ninguna caracteristica de s=S (sin fuga)")


def test_fecha_de_la_semana_futura():
    G, T, S = 2, 8, 15
    m = _meta(G, T, S)
    gc = GeneradorCaracteristicas()
    gc.construirX(np.ones((G, T, S), dtype=np.int32), m, np.arange(12, 14), semanas_futuras=1)
    d = (gc.fecha_semana_futura(0) - m["fecha_semana"][-1]).days
    assert d == 7, d
    idx = gc.indices_semana_futura(G, T, S)
    assert len(idx) == G * T and (idx % gc.S_total == S).all()
    print(f"  OK la semana objetivo es 7 dias posterior a la ultima observada")


# --- Fallo 3: la escalera de umbrales habia desaparecido --------------------
def test_escalera_se_detiene_en_global_si_alcanza():
    """Distribucion sana: debe quedarse en el primer peldano."""
    rng = np.random.default_rng(0)
    c = rng.poisson(6, (40, 7, 60)).astype(np.int32)
    ce = ConstructorEtiquetas()
    ce.ajustarUmbrales(c, np.arange(12, 45))
    assert ce.modo_usado == "global", ce.modo_usado
    assert ce.corresponde_a_percentiles is True
    print(f"  OK distribucion sana -> modo '{ce.modo_usado}', masa acumulada "
          f"{ce.masa_acumulada_bajo:.1f}%/{ce.masa_acumulada_alto:.1f}%")


def test_escalera_recorre_y_declara_cuando_hay_cero_inflacion():
    """Con 93% de ceros ningun percentil funciona: debe llegar a 'discreto' y avisar."""
    rng = np.random.default_rng(1)
    c = rng.poisson(0.08, (40, 7, 60)).astype(np.int32)
    ce = ConstructorEtiquetas()
    ce.ajustarUmbrales(c, np.arange(12, 45))

    modos = [d[0] for d in ce.diagnostico]
    assert modos[0] == "global" and "positivos" in modos, modos
    assert ce.modo_usado == "discreto", ce.modo_usado
    assert ce.corresponde_a_percentiles is False
    assert ce.masa_acumulada_alto > 95, ce.masa_acumulada_alto

    y = ce.construirY(c)
    cnt, _ = ConstructorEtiquetas.distribucion(y)
    assert (cnt > 0).all(), cnt
    print(f"  OK escalera recorrida {modos} -> '{ce.modo_usado}'; corte superior con "
          f"masa acumulada {ce.masa_acumulada_alto:.1f}%, marcado como NO percentil")


def test_por_dia_se_rechaza_si_algun_dia_pierde_la_clase_media():
    """
    El modo por_dia pasaba el chequeo por tener tres clases en el AGREGADO,
    aunque dentro de varios dias P60 == P90 y ahi todo c >= 1 saltaba a "alto",
    de modo que la etiqueta significaba cosas distintas segun el dia.

    Se construyen 2 dias con actividad y 5 casi vacios sobre |T| = 7.
    """
    rng = np.random.default_rng(11)
    G, T, S = 300, 7, 80
    lam = np.where(np.isin(np.arange(T), np.arange(2)), 0.9, 0.03)
    c = rng.poisson(lam[None, :, None] * np.ones((G, T, S))).astype(np.int32)
    sem_tr = np.arange(12, 60)

    ce = ConstructorEtiquetas(modo="por_dia")
    try:
        ce.ajustarUmbrales(c, sem_tr)
        fallo = ce.modo_usado == "por_dia"
    except ValueError:
        fallo = False
    assert not fallo, "por_dia no debe aceptarse con dias sin clase media"

    d = [x for x in ce.diagnostico if x[0] == "por_dia"][0]
    colapsadas = d[3]
    assert colapsadas is not None and colapsadas > 0, d
    print(f"  OK por_dia rechazado: {colapsadas} dias con P60=P90 "
          "quedarian sin clase media")


def test_alto_mas_frecuente_que_medio_no_invalida_percentiles_correctos():
    """
    Regresion del sobre-ajuste introducido al corregir por_dia: se habia
    puesto props[alto] > props[medio] como criterio de RECHAZO. Es incorrecto.

    Con 85% de ceros, 5% de unos y 10% de doses, np.percentile da P60=0 y
    P90=1.1, y las clases quedan 85/5/10. La clase alta es mas frecuente que la
    media, pero no hay inversion: "alto" es exactamente c > 1.1, el extremo
    superior. Ese etiquetado es legitimo y debe aceptarse.
    """
    c = _tensor_exacto(200, 7, 70)
    ce = ConstructorEtiquetas()
    ce.ajustarUmbrales(c, np.arange(70))

    assert ce.modo_usado == "global", f"debio aceptarse en global, no en {ce.modo_usado}"
    assert ce.corresponde_a_percentiles is True
    p = ce.proporciones
    assert p[2] > p[1], "el caso de prueba debe tener alto > medio"
    assert ce.alto_supera_medio is True, "debe quedar como diagnostico"
    assert abs(p[0] - 0.85) < 0.01 and abs(p[2] - 0.10) < 0.01, p
    print(f"  OK global aceptado con alto {p[2]*100:.1f}% > medio {p[1]*100:.1f}%: "
          "en una discreta eso no invalida P60/P90")


def test_corresponde_a_percentiles_es_por_construccion_no_por_masa_acumulada():
    """
    El cuantil de una discreta es una funcion escalon: con 85% de ceros, el
    valor 0 es a la vez el percentil 60, el 70 y el 80. Comprobar la condicion
    "es un percentil" mirando F(u) daba falsos negativos. Ahora los tres
    peldanos que usan np.percentile son percentiles por construccion, y solo
    'discreto' no lo es; F(u) se conserva aparte como masa acumulada.
    """
    c = _tensor_exacto(200, 7, 70)
    ce = ConstructorEtiquetas()
    ce.ajustarUmbrales(c, np.arange(70))
    assert ce.corresponde_a_percentiles is True
    assert ce.masa_acumulada_bajo > 80, ce.masa_acumulada_bajo
    print(f"  OK P60 valido con masa acumulada F(u)={ce.masa_acumulada_bajo:.1f}% "
          "(no es contradiccion en una discreta)")

    # y 'discreto' si queda marcado como no percentil
    c2 = np.random.default_rng(1).poisson(0.08, (300, 7, 80)).astype(np.int32)
    ce2 = ConstructorEtiquetas()
    ce2.ajustarUmbrales(c2, np.arange(12, 60))
    assert ce2.modo_usado == "discreto"
    assert ce2.corresponde_a_percentiles is False
    print("  OK 'discreto' si queda marcado como NO percentil")


def test_base_de_calculo_se_declara_por_modo():
    """
    Regresion: _fijar calculaba el percentil real contra todos los conteos
    incluidos los ceros, incluso en el modo 'positivos', cuyos umbrales son
    P60/P90 de los conteos > 0. Eso hacia imprimir "NO son percentiles" cuando
    si lo eran, y podia meter una afirmacion falsa en el reporte.
    """
    rng = np.random.default_rng(9)
    act = rng.random((200, 7, 1)) < 0.10
    c = rng.poisson(np.where(act, 40.0, 0.0) * np.ones((200, 7, 70))).astype(np.int32)
    ce = ConstructorEtiquetas()
    ce.ajustarUmbrales(c, np.arange(12, 50))

    assert ce.modo_usado == "positivos", ce.modo_usado
    assert ce.base_referencia == "positivos"
    assert ce.corresponde_a_percentiles is True
    assert ce.masa_acumulada_bajo > 90, ce.masa_acumulada_bajo
    print(f"  OK modo positivos: base declarada '{ce.base_referencia}', "
          f"masa acumulada {ce.masa_acumulada_bajo:.1f}%/{ce.masa_acumulada_alto:.1f}% "
          "sobre todos los conteos")


def test_umbrales_por_dia_producen_vector():
    ce = ConstructorEtiquetas(modo="por_dia", max_dias_colapsados=7)
    rng = np.random.default_rng(2)
    c = rng.poisson(np.linspace(1, 12, 7)[None, :, None] * np.ones((30, 7, 60))).astype(np.int32)
    ce.ajustarUmbrales(c, np.arange(12, 45))
    assert ce.por_dia and np.asarray(ce.P60).shape == (7,)
    y = ce.construirY(c)
    assert y.shape == c.shape and (np.bincount(y.ravel(), minlength=3) > 0).all()
    print("  OK el modo por_dia produce un umbral por dia y tres clases")


# --- Fallo 4: unidades sin historial entraban al entrenamiento ----------------
def test_elegibilidad_excluye_unidades_que_aparecen_despues_del_entrenamiento():
    G, T, S = 5, 4, 30
    c = np.zeros((G, T, S), dtype=np.int32)
    c[0:3] = 2                       # unidades activas desde el inicio
    c[3, :, 25:] = 5                 # aparece solo en prueba
    # la unidad 4 no tiene nada nunca
    sem_tr = np.arange(12, 20)

    m = mascara_elegibilidad(c, sem_tr, nivel="unidad")
    assert m[0].all() and m[1].all() and m[2].all()
    assert not m[3].any(), "la unidad que aparece despues del entrenamiento debe excluirse"
    assert not m[4].any()

    gc = GeneradorCaracteristicas()
    gc.construirX(c, _meta(G, T, S), sem_tr, semanas_futuras=1)
    i_tr = indices_de_semanas(G, T, gc.S_total, sem_tr)
    i_ok = filtrar_elegibles(i_tr, m, T, gc.S_total)
    assert len(i_ok) == 3 * T * len(sem_tr), (len(i_ok), 3 * T * len(sem_tr))

    g = i_ok // (T * gc.S_total)
    assert set(np.unique(g).tolist()) == {0, 1, 2}
    print(f"  OK entrenamiento pasa de {len(i_tr)} a {len(i_ok)} filas; "
          "las unidades 3 y 4 no ingresan al modelo")


def test_elegibilidad_unidad_dia_es_mas_estricta():
    G, T, S = 4, 8, 30
    c = np.zeros((G, T, S), dtype=np.int32)
    c[:, 0:3, :] = 1                 # solo tres dias tienen actividad
    sem_tr = np.arange(12, 20)
    m_unidad = mascara_elegibilidad(c, sem_tr, nivel="unidad")
    m_cf = mascara_elegibilidad(c, sem_tr, nivel="unidad_dia")
    assert m_unidad.sum() == G * T
    assert m_cf.sum() == G * 3
    print(f"  OK nivel 'unidad' deja {m_unidad.sum()} y 'unidad_dia' deja {m_cf.sum()}")


if __name__ == "__main__":
    pruebas = [
        ("Fallo 1 - prediccion de la semana siguiente",
         [test_semana_futura_usa_rezagos_correctos,
          test_semana_futura_no_depende_del_valor_desconocido,
          test_fecha_de_la_semana_futura]),
        ("Fallo 3 - escalera de umbrales y su declaracion",
         [test_escalera_se_detiene_en_global_si_alcanza,
          test_escalera_recorre_y_declara_cuando_hay_cero_inflacion,
          test_por_dia_se_rechaza_si_algun_dia_pierde_la_clase_media,
          test_alto_mas_frecuente_que_medio_no_invalida_percentiles_correctos,
          test_corresponde_a_percentiles_es_por_construccion_no_por_masa_acumulada,
          test_base_de_calculo_se_declara_por_modo,
          test_umbrales_por_dia_producen_vector]),
        ("Fallo 4 - elegibilidad de unidades sin historial",
         [test_elegibilidad_excluye_unidades_que_aparecen_despues_del_entrenamiento,
          test_elegibilidad_unidad_dia_es_mas_estricta]),
    ]
    import io
    import contextlib
    for titulo, fs in pruebas:
        print(f"\n{titulo}")
        for f in fs:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                f()
            print([l for l in buf.getvalue().splitlines() if l.startswith("  OK")][0])
    print("\nTodas las pruebas de regresion pasaron.")
