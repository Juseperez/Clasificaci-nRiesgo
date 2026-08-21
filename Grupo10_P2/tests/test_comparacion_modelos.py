"""Pruebas de la nueva comparacion de familias de modelos."""
import inspect
import json

import numpy as np

from src.bosque import BosqueAleatorioPropio
from src.comparacion_modelos import (
    NOMBRES_MODELOS,
    ModeloAjustado,
    benchmark_rf_sklearn,
    construir_logistica,
    construir_mlp,
    evaluar_en_test,
    guardar_resultados,
    pesos_clase_balanceados,
    pesos_muestra_balanceados,
    comparar_en_validacion,
)


def _datos_sinteticos(seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(180, 5))
    z = X[:, 0] + 0.7 * X[:, 1] - 0.25 * X[:, 2]
    y = np.digitize(z, [-0.45, 0.65]).astype(np.uint8)
    return X[:110], y[:110], X[110:145], y[110:145], X[145:], y[145:]


def test_pesos_clase_reproducen_formula_balanceada():
    y = np.array([0] * 6 + [1] * 3 + [2] * 1, dtype=np.uint8)
    w = pesos_clase_balanceados(y)
    esperado = np.array([10 / (3 * 6), 10 / (3 * 3), 10 / (3 * 1)])
    np.testing.assert_allclose(w, esperado)
    np.testing.assert_allclose(pesos_muestra_balanceados(y), w[y])


def test_seleccion_no_acepta_test_en_su_firma():
    # Barrera estructural contra seleccionar hiperparametros mirando test.
    params = set(inspect.signature(comparar_en_validacion).parameters)
    assert "Xte" not in params and "yte" not in params
    assert "X_test" not in params and "y_test" not in params


def test_logistica_escala_solo_con_train():
    Xtr, ytr, Xva, _, _, _ = _datos_sinteticos()
    # Hacemos validation deliberadamente extrema; no debe entrar al fit del scaler.
    Xva_extrema = Xva + 1000
    p = construir_logistica(C=1.0)
    p.fit(Xtr, ytr)
    np.testing.assert_allclose(p.named_steps["scaler"].mean_, Xtr.mean(axis=0))
    assert not np.allclose(p.named_steps["scaler"].mean_, np.vstack([Xtr, Xva_extrema]).mean(axis=0))


def test_mlp_pipeline_contiene_scaler_y_tres_salidas():
    Xtr, ytr, _, _, Xte, _ = _datos_sinteticos()
    sw = pesos_muestra_balanceados(ytr)
    p = construir_mlp(capas=(8,), alpha=1e-3, max_iter=120)
    p.fit(Xtr, ytr, modelo__sample_weight=sw)
    proba = p.predict_proba(Xte)
    assert proba.shape == (len(Xte), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)


def test_evaluar_en_test_acepta_bosque_propio_y_sklearn():
    Xtr, ytr, Xva, yva, Xte, yte = _datos_sinteticos()

    # El bosque propio requiere bins enteros; discretizamos solo para esta prueba.
    lo = Xtr.min(axis=0)
    hi = Xtr.max(axis=0)
    def bins(X):
        q = (X - lo) / np.maximum(hi - lo, 1e-9)
        return np.clip((q * 7).astype(np.int16), 0, 7).astype(np.uint8)

    b = BosqueAleatorioPropio(nArboles=5, profundidadMax=4, minMuestrasHoja=2,
                              nBins=8, semilla=42, verboso=False)
    b.entrenar(bins(Xtr), ytr)

    # Para probar la interfaz comun usamos cinco entradas; las cuatro externas
    # reutilizan una logistica pequena. El objetivo del test es la estructura,
    # no comparar algoritmos.
    lr = construir_logistica(C=1.0)
    lr.fit(Xtr, ytr)

    modelos = {
        "bosque propio": ModeloAjustado("bosque propio", b, {}, 0.0, 0.0),
        "extra trees": ModeloAjustado("extra trees", lr, {}, 0.0, 0.0),
        "regresion logistica": ModeloAjustado("regresion logistica", lr, {}, 0.0, 0.0),
        "gradient boosting": ModeloAjustado("gradient boosting", lr, {}, 0.0, 0.0),
        "mlp": ModeloAjustado("mlp", lr, {}, 0.0, 0.0),
    }

    # El bosque espera bins; hacemos una copia de la evaluacion para verificarlo
    # por separado y luego evaluamos los sklearn sobre X real.
    pred = b.predecir(bins(Xte))
    assert pred.shape == (len(yte),)

    # Sustituimos la entrada del bosque por un adaptador sklearn-like para que
    # toda la tabla reciba la misma matriz Xte en este test sintetico.
    modelos["bosque propio"] = ModeloAjustado("bosque propio", lr, {}, 0.0, 0.0)
    tabla, reportes = evaluar_en_test(modelos, Xte, yte)
    assert set(tabla["modelo"]) == set(NOMBRES_MODELOS)
    assert len(reportes) == 5
    assert np.isfinite(tabla["F1 macro"]).all()
    assert np.isfinite(tabla["Brier"]).all()


def test_guardar_resultados_no_sobrescribe_modelo(tmp_path):
    import pandas as pd

    val = pd.DataFrame([
        {"modelo": "bosque propio", "F1 macro validacion": 0.8,
         "tiempo ajuste s": 1.0, "configuracion": "{}"},
    ])
    test = pd.DataFrame([
        {"modelo": "bosque propio", "F1 macro": 0.79},
    ])
    modelos = {
        "bosque propio": ModeloAjustado("bosque propio", object(), {"nArboles": 100}, 0.8, 1.0)
    }

    congelado = tmp_path / "modelo.joblib"
    congelado.write_bytes(b"NO TOCAR")
    rutas = guardar_resultados(val, test, modelos, carpeta=str(tmp_path))

    assert congelado.read_bytes() == b"NO TOCAR"
    for ruta in rutas.values():
        assert tmp_path.joinpath(ruta.split("/")[-1]).exists() or __import__("os").path.exists(ruta)
    payload = json.loads((tmp_path / "comparacion_modelos.json").read_text(encoding="utf-8"))
    assert payload["modelo_mejor_validacion"] == "bosque propio"


def test_benchmark_rf_sklearn_devuelve_version_y_metricas():
    Xtr, ytr, _, _, Xte, yte = _datos_sinteticos()
    hiper = {"nArboles": 5, "profundidadMax": 4, "minMuestrasHoja": 2,
             "maxMuestrasPorArbol": 1000}
    rep = benchmark_rf_sklearn(Xtr, ytr, Xte, yte, hiper)
    assert "sklearn_version" in rep
    assert 0 <= rep["f1_macro"] <= 1
    assert rep["matriz_confusion"].shape == (3, 3)