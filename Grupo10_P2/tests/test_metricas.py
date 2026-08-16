"""
Auditoria 16-ago: EvaluadorMetricas contra la matriz de confusion real de la
corrida oficial (bosque propio, conjunto de prueba, n=1050).

No se reimplementa el calculo de metricas aqui: se reconstruyen los vectores
(y, yPred) que producen exactamente esa matriz de confusion y se llama a las
funciones reales de src/metricas.py. Antes de esta prueba, ese modulo no tenia
ninguna cobertura y toda cifra publicada en README/metricas.json dependia de
el sin verificacion automatica.

    python -m tests.test_metricas
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metricas import EvaluadorMetricas


def _construir_desde_matriz(M):
    """Genera (y, yPred) cuya matriz de confusion es exactamente M (filas=real)."""
    y, yPred = [], []
    for real, fila in enumerate(M):
        for pred, n in enumerate(fila):
            y.extend([real] * int(n))
            yPred.extend([pred] * int(n))
    return np.array(y, dtype=np.uint8), np.array(yPred, dtype=np.uint8)


# Matriz de confusion del bosque propio sobre el conjunto de prueba, tal como
# quedo reportada en README.md #10 y salidas/metricas.json (corrida oficial).
M_BOSQUE_PROPIO = np.array([
    [581,  94,   0],
    [104,  96,   0],
    [  0,   0, 175],
])


def test_matriz_confusion_se_reconstruye_igual():
    y, yPred = _construir_desde_matriz(M_BOSQUE_PROPIO)
    assert y.size == 1050, y.size          # n_test de la corrida oficial
    M = EvaluadorMetricas.matrizConfusion(y, yPred)
    assert np.array_equal(M, M_BOSQUE_PROPIO), M
    print("  OK matrizConfusion reproduce la matriz publicada (n=1050)")


def test_metricas_del_bosque_propio_coinciden_con_lo_publicado():
    y, yPred = _construir_desde_matriz(M_BOSQUE_PROPIO)
    _, _, f1 = EvaluadorMetricas.precisionRecallF1(y, yPred)

    assert round(EvaluadorMetricas.exactitud(y, yPred), 4) == 0.8114
    assert np.allclose(np.round(f1, 4), [0.8544, 0.4923, 1.0000])
    assert round(EvaluadorMetricas.f1Macro(y, yPred), 4) == 0.7822
    print("  OK accuracy 0.8114, F1 [bajo/medio/alto] = [0.8544, 0.4923, 1.0000], "
          "F1 macro 0.7822 (coincide con README y metricas.json)")


if __name__ == "__main__":
    for f in (test_matriz_confusion_se_reconstruye_igual,
              test_metricas_del_bosque_propio_coinciden_con_lo_publicado):
        f()
    print("\nTodas las pruebas de metricas.py pasaron.")
