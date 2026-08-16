"""
Verificacion via 1: pruebas unitarias del criterio de division contra casos
resueltos a mano. El caso principal es el que ya quedo reportado en la Tarea #4.

    python -m tests.test_gini      (desde la raiz del proyecto)
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bosque import ArbolDecisionPropio, BosqueAleatorioPropio, giniPonderado


def test_pesos_de_clase():
    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint8)
    w = BosqueAleatorioPropio.pesosDeClase(y)
    esperado = np.array([6 / 9, 1.0, 2.0])
    assert np.allclose(w, esperado), w
    print(f"  OK pesos w_k = n/(3 n_k) = {np.round(w, 4)}")


def test_gini_nodo():
    """Nodo {0,0,0,1,1,2}: los pesos igualan las proporciones -> Gini = 2/3."""
    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint8)
    w = BosqueAleatorioPropio.pesosDeClase(y)
    W = np.bincount(y, weights=w[y], minlength=3)
    assert np.allclose(W, [2.0, 2.0, 2.0]), W
    g = float(giniPonderado(W))
    assert abs(g - 2 / 3) < 1e-9, g
    print(f"  OK Gini del nodo = {g:.6f}  (esperado 0.666667)")


def test_gini_hijos():
    """Corte que separa {0,0,0} de {1,1,2}: hijos 0 y 0.5, promedio 0.3333."""
    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint8)
    w = BosqueAleatorioPropio.pesosDeClase(y)

    izq, der = y[:3], y[3:]
    W_izq = np.bincount(izq, weights=w[izq], minlength=3)
    W_der = np.bincount(der, weights=w[der], minlength=3)
    g_izq, g_der = float(giniPonderado(W_izq)), float(giniPonderado(W_der))
    assert abs(g_izq - 0.0) < 1e-9, g_izq
    assert abs(g_der - 0.5) < 1e-9, g_der

    prom = (W_izq.sum() * g_izq + W_der.sum() * g_der) / (W_izq.sum() + W_der.sum())
    assert abs(prom - 1 / 3) < 1e-9, prom
    print(f"  OK Gini hijos = ({g_izq:.4f}, {g_der:.4f}), "
          f"promedio ponderado por peso total = {prom:.6f}  (esperado 0.333333)")


def test_mejor_corte_encuentra_la_separacion():
    """El buscador de cortes debe elegir exactamente esa division."""
    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint8)
    w = BosqueAleatorioPropio.pesosDeClase(y)
    Xb = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.uint8)

    arbol = ArbolDecisionPropio(minMuestrasHoja=1, nBins=32,
                                rng=np.random.default_rng(0))
    corte = arbol.mejorCorte(Xb, y, w, np.arange(6, dtype=np.int32), [0])
    assert corte is not None
    j, u, ganancia = corte
    assert j == 0 and u == 0, corte
    esperado = (2 / 3 - 1 / 3) * 6.0
    assert abs(ganancia - esperado) < 1e-9, (ganancia, esperado)
    print(f"  OK mejorCorte -> atributo {j}, umbral bin {u}, "
          f"reduccion ponderada {ganancia:.6f}  (esperado {esperado:.6f})")


def test_nodo_puro_no_se_divide():
    y = np.zeros(20, dtype=np.uint8)
    w = np.ones(3)
    Xb = np.arange(20, dtype=np.uint8).reshape(-1, 1) % 32
    arbol = ArbolDecisionPropio(minMuestrasHoja=1, rng=np.random.default_rng(0))
    assert arbol.mejorCorte(Xb, y, w, np.arange(20, dtype=np.int32), [0]) is None
    print("  OK un nodo puro no produce corte")


def test_arbol_separable_es_perfecto():
    """Problema linealmente separable: el arbol debe clasificarlo sin error."""
    rng = np.random.default_rng(7)
    n = 3000
    Xb = rng.integers(0, 32, size=(n, 4)).astype(np.uint8)
    y = np.where(Xb[:, 0] < 10, 0, np.where(Xb[:, 1] < 16, 1, 2)).astype(np.uint8)
    w = BosqueAleatorioPropio.pesosDeClase(y)

    arbol = ArbolDecisionPropio(profundidadMax=6, minMuestrasHoja=1, mAtributos=4,
                                rng=np.random.default_rng(1))
    arbol.construir(Xb, y, w)
    pred = arbol.predecirProba(Xb).argmax(axis=1)
    acc = (pred == y).mean()
    assert acc > 0.99, acc
    print(f"  OK arbol sobre problema separable: exactitud {acc:.4f}")


def test_probabilidades_suman_uno():
    rng = np.random.default_rng(3)
    Xb = rng.integers(0, 32, size=(2000, 6)).astype(np.uint8)
    y = rng.integers(0, 3, size=2000).astype(np.uint8)
    bosque = BosqueAleatorioPropio(nArboles=5, profundidadMax=4, verboso=False,
                                   semilla=1).entrenar(Xb, y)
    p = bosque.predecirProba(Xb)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert (p >= 0).all()
    print("  OK f(x) entrega un vector de probabilidad valido (suma 1, no negativo)")


if __name__ == "__main__":
    print("Pruebas del criterio de division (casos resueltos a mano):")
    for f in [test_pesos_de_clase, test_gini_nodo, test_gini_hijos,
              test_mejor_corte_encuentra_la_separacion, test_nodo_puro_no_se_divide,
              test_arbol_separable_es_perfecto, test_probabilidades_suman_uno]:
        f()
    print("\nTodas las pruebas pasaron.")
