"""
Comparacion reproducible de familias de clasificadores para el Componente 2.

Objetivo metodologico
---------------------
Comparar cinco familias sobre EXACTAMENTE las mismas X/y y la misma particion
cronologica del proyecto, sin utilizar el conjunto de prueba para seleccionar
modelo ni hiperparametros.

Familias principales:
  1. Bosque aleatorio propio (ya ajustado por el notebook del proyecto)
  2. Extra Trees
  3. Gradient Boosting
  4. Regresion logistica multinomial
  5. Perceptron multicapa (MLP)

RandomForest de scikit-learn se conserva fuera de estas cinco familias como
benchmark de implementacion del bosque propio.

El flujo correcto es:
  train -> ajuste de cada candidato
  validation -> seleccion/configuracion
  test -> evaluacion final, una vez congeladas las decisiones

Este modulo deliberadamente separa ``comparar_en_validacion`` de
``evaluar_en_test``. La funcion de seleccion NO recibe X_test ni y_test.
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metricas import EvaluadorMetricas


NOMBRES_MODELOS = (
    "bosque propio",
    "extra trees",
    "regresion logistica",
    "gradient boosting",
    "mlp",
)


@dataclass
class ModeloAjustado:
    """Modelo ya entrenado y congelado tras seleccion en validacion."""

    nombre: str
    estimador: Any
    configuracion: Dict[str, Any]
    f1_macro_validacion: float
    tiempo_ajuste_s: float


# ---------------------------------------------------------------------------
# Pesos y utilidades comunes
# ---------------------------------------------------------------------------
def pesos_clase_balanceados(y: np.ndarray, n_clases: int = 3) -> np.ndarray:
    """Reproduce w_k = n / (K*n_k), la potencia 1.0 del bosque definitivo."""
    y = np.asarray(y, dtype=np.int64)
    conteos = np.bincount(y, minlength=n_clases).astype(np.float64)
    n = float(conteos.sum())
    return np.where(conteos > 0, n / (n_clases * conteos), 0.0)


def pesos_muestra_balanceados(y: np.ndarray, n_clases: int = 3) -> np.ndarray:
    """Peso por fila obtenido a partir del peso de su clase."""
    w = pesos_clase_balanceados(y, n_clases=n_clases)
    return w[np.asarray(y, dtype=np.int64)]


def _predecir(estimador: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Interfaz comun para el bosque propio y los estimadores scikit-learn."""
    if hasattr(estimador, "predecir"):
        pred = np.asarray(estimador.predecir(X), dtype=np.uint8)
        proba = (np.asarray(estimador.predecirProba(X), dtype=float)
                 if hasattr(estimador, "predecirProba") else None)
        return pred, proba

    pred = np.asarray(estimador.predict(X), dtype=np.uint8)
    proba = (np.asarray(estimador.predict_proba(X), dtype=float)
             if hasattr(estimador, "predict_proba") else None)
    return pred, proba


def _f1_validacion(estimador: Any, Xva: np.ndarray, yva: np.ndarray) -> float:
    pred, _ = _predecir(estimador, Xva)
    return EvaluadorMetricas.f1Macro(np.asarray(yva), pred)


def _mejor_por_f1(candidatos: Iterable[Tuple[Any, Dict[str, Any], float]]) -> Tuple[Any, Dict[str, Any], float]:
    """Seleccion determinista: mayor F1; en empate conserva el primero."""
    mejor_modelo = None
    mejor_cfg: Dict[str, Any] = {}
    mejor_f1 = -np.inf
    for modelo, cfg, f1 in candidatos:
        if f1 > mejor_f1 + 1e-12:
            mejor_modelo, mejor_cfg, mejor_f1 = modelo, cfg, float(f1)
    if mejor_modelo is None:
        raise RuntimeError("No se genero ningun candidato de modelo.")
    return mejor_modelo, mejor_cfg, mejor_f1


# ---------------------------------------------------------------------------
# Constructores de pipelines. El StandardScaler queda DENTRO del pipeline,
# por lo que fit() se ejecuta solo con X_train.
# ---------------------------------------------------------------------------
def construir_logistica(C: float = 1.0, semilla: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(
            C=float(C),
            class_weight="balanced",
            max_iter=2_000,
            solver="lbfgs",
            random_state=semilla,
        )),
    ])


def construir_mlp(capas: Tuple[int, ...] = (64, 32), alpha: float = 1e-4,
                   semilla: int = 42, max_iter: int = 800) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", MLPClassifier(
            hidden_layer_sizes=tuple(capas),
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            learning_rate_init=0.001,
            max_iter=int(max_iter),
            early_stopping=False,  # la seleccion se hace con NUESTRA validacion temporal
            random_state=semilla,
        )),
    ])


# ---------------------------------------------------------------------------
# Busquedas pequenas y defendibles. Todas usan SOLO train + validation.
# ---------------------------------------------------------------------------
def buscar_extra_trees(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    profundidades: Sequence[Optional[int]] = (8, 16, None),
    min_hoja: Sequence[int] = (1, 5),
    n_estimators: int = 300,
    semilla: int = 42,
) -> ModeloAjustado:
    t0 = time.perf_counter()
    candidatos = []
    for prof, hoja in product(profundidades, min_hoja):
        m = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            max_depth=prof,
            min_samples_leaf=int(hoja),
            max_features="sqrt",
            class_weight="balanced",
            random_state=semilla,
            n_jobs=-1,
        )
        m.fit(Xtr, ytr)
        cfg = {
            "n_estimators": int(n_estimators),
            "max_depth": prof,
            "min_samples_leaf": int(hoja),
            "max_features": "sqrt",
            "class_weight": "balanced",
        }
        candidatos.append((m, cfg, _f1_validacion(m, Xva, yva)))

    m, cfg, f1 = _mejor_por_f1(candidatos)
    return ModeloAjustado("extra trees", m, cfg, f1, time.perf_counter() - t0)


def buscar_logistica(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Cs: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    semilla: int = 42,
) -> ModeloAjustado:
    t0 = time.perf_counter()
    candidatos = []
    for C in Cs:
        m = construir_logistica(C=C, semilla=semilla)
        # El scaler se ajusta exclusivamente aqui, con Xtr.
        m.fit(Xtr, ytr)
        cfg = {"C": float(C), "class_weight": "balanced", "scaler": "StandardScaler(train)"}
        candidatos.append((m, cfg, _f1_validacion(m, Xva, yva)))

    m, cfg, f1 = _mejor_por_f1(candidatos)
    return ModeloAjustado("regresion logistica", m, cfg, f1, time.perf_counter() - t0)


def buscar_gradient_boosting(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    n_estimators_grid: Sequence[int] = (100, 200),
    learning_rates: Sequence[float] = (0.05, 0.10),
    profundidades: Sequence[int] = (2, 3),
    semilla: int = 42,
) -> ModeloAjustado:
    t0 = time.perf_counter()
    sw = pesos_muestra_balanceados(ytr)
    candidatos = []
    for n_est, lr, prof in product(n_estimators_grid, learning_rates, profundidades):
        m = GradientBoostingClassifier(
            n_estimators=int(n_est),
            learning_rate=float(lr),
            max_depth=int(prof),
            random_state=semilla,
        )
        m.fit(Xtr, ytr, sample_weight=sw)
        cfg = {
            "n_estimators": int(n_est),
            "learning_rate": float(lr),
            "max_depth": int(prof),
            "sample_weight": "balanceado con train",
        }
        candidatos.append((m, cfg, _f1_validacion(m, Xva, yva)))

    m, cfg, f1 = _mejor_por_f1(candidatos)
    return ModeloAjustado("gradient boosting", m, cfg, f1, time.perf_counter() - t0)


def buscar_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    arquitecturas: Sequence[Tuple[int, ...]] = ((32,), (64, 32), (128, 64)),
    alphas: Sequence[float] = (1e-4, 1e-3),
    semilla: int = 42,
    max_iter: int = 800,
) -> ModeloAjustado:
    t0 = time.perf_counter()
    sw = pesos_muestra_balanceados(ytr)
    candidatos = []

    for capas, alpha in product(arquitecturas, alphas):
        m = construir_mlp(capas=capas, alpha=alpha, semilla=semilla, max_iter=max_iter)
        try:
            # En sklearn 1.9.0 MLPClassifier.fit acepta sample_weight.
            # El prefijo modelo__ lo envia solo al ultimo paso del Pipeline.
            m.fit(Xtr, ytr, modelo__sample_weight=sw)
        except TypeError as exc:
            raise RuntimeError(
                "La version de scikit-learn no acepta sample_weight en MLPClassifier. "
                "El proyecto fue verificado con scikit-learn 1.9.0."
            ) from exc

        cfg = {
            "hidden_layer_sizes": tuple(int(v) for v in capas),
            "alpha": float(alpha),
            "learning_rate_init": 0.001,
            "max_iter": int(max_iter),
            "sample_weight": "balanceado con train",
            "scaler": "StandardScaler(train)",
        }
        candidatos.append((m, cfg, _f1_validacion(m, Xva, yva)))

    m, cfg, f1 = _mejor_por_f1(candidatos)
    return ModeloAjustado("mlp", m, cfg, f1, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Comparacion principal. NOTA: no recibe test por diseno.
# ---------------------------------------------------------------------------
def comparar_en_validacion(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    bosque_propio: Any,
    hiper_bosque: Dict[str, Any],
    f1_val_bosque: Optional[float] = None,
    semilla: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ModeloAjustado]]:
    """
    Ajusta las cuatro familias externas usando train y las selecciona en val.

    X_test/y_test NO forman parte de la firma, por lo que esta etapa no puede
    seleccionar mirando el conjunto de prueba por accidente.
    """
    if f1_val_bosque is None:
        f1_val_bosque = _f1_validacion(bosque_propio, Xva, yva)

    modelos: Dict[str, ModeloAjustado] = {
        "bosque propio": ModeloAjustado(
            "bosque propio",
            bosque_propio,
            dict(hiper_bosque),
            float(f1_val_bosque),
            float(getattr(bosque_propio, "tiempo_entrenamiento", np.nan)),
        )
    }

    for ajustado in (
        buscar_extra_trees(Xtr, ytr, Xva, yva, semilla=semilla),
        buscar_logistica(Xtr, ytr, Xva, yva, semilla=semilla),
        buscar_gradient_boosting(Xtr, ytr, Xva, yva, semilla=semilla),
        buscar_mlp(Xtr, ytr, Xva, yva, semilla=semilla),
    ):
        modelos[ajustado.nombre] = ajustado

    filas = []
    for nombre in NOMBRES_MODELOS:
        m = modelos[nombre]
        filas.append({
            "modelo": m.nombre,
            "F1 macro validacion": round(float(m.f1_macro_validacion), 6),
            "tiempo ajuste s": round(float(m.tiempo_ajuste_s), 3),
            "configuracion": json.dumps(m.configuracion, ensure_ascii=False, default=str),
        })

    tabla = pd.DataFrame(filas).sort_values("F1 macro validacion", ascending=False, kind="stable")
    tabla = tabla.reset_index(drop=True)
    return tabla, modelos


def evaluar_en_test(
    modelos: Dict[str, ModeloAjustado],
    Xte: np.ndarray,
    yte: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Evalua configuraciones YA congeladas. No ajusta nada en test."""
    reportes: Dict[str, Dict[str, Any]] = {}
    filas = []

    for nombre in NOMBRES_MODELOS:
        ajustado = modelos[nombre]
        pred, proba = _predecir(ajustado.estimador, Xte)
        rep = EvaluadorMetricas.reporte(yte, pred, proba, titulo=nombre)
        reportes[nombre] = rep

        filas.append({
            "modelo": nombre,
            "exactitud": round(float(rep["exactitud"]), 6),
            "F1 macro": round(float(rep["f1_macro"]), 6),
            "F1 bajo": round(float(rep["f1_por_clase"][0]), 6),
            "F1 medio": round(float(rep["f1_por_clase"][1]), 6),
            "F1 alto": round(float(rep["f1_por_clase"][2]), 6),
            "recall alto": round(float(rep["recall_alto"]), 6),
            "Brier": round(float(rep["brier"]), 6) if "brier" in rep else np.nan,
            "AUC bajo": round(float(rep["auc_ovr"][0]), 6) if "auc_ovr" in rep else np.nan,
            "AUC medio": round(float(rep["auc_ovr"][1]), 6) if "auc_ovr" in rep else np.nan,
            "AUC alto": round(float(rep["auc_ovr"][2]), 6) if "auc_ovr" in rep else np.nan,
        })

    tabla = pd.DataFrame(filas).sort_values("F1 macro", ascending=False, kind="stable")
    return tabla.reset_index(drop=True), reportes


def benchmark_rf_sklearn(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    hiper_bosque: Dict[str, Any],
    semilla: int = 42,
) -> Dict[str, Any]:
    """Benchmark de implementacion; NO cuenta como sexta familia principal."""
    from sklearn.ensemble import RandomForestClassifier

    n_train = len(ytr)
    max_muestras = min(int(hiper_bosque.get("maxMuestrasPorArbol", n_train)), n_train)
    rf = RandomForestClassifier(
        n_estimators=int(hiper_bosque.get("nArboles", 100)),
        max_depth=int(hiper_bosque.get("profundidadMax", 16)),
        min_samples_leaf=int(hiper_bosque.get("minMuestrasHoja", 5)),
        max_features="sqrt",
        class_weight="balanced",
        max_samples=max_muestras,
        bootstrap=True,
        random_state=semilla,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    rf.fit(Xtr, ytr)
    tiempo = time.perf_counter() - t0
    rep = EvaluadorMetricas.reporte(
        yte, rf.predict(Xte), rf.predict_proba(Xte),
        "sklearn RandomForest (benchmark de implementacion)",
    )
    rep["tiempo_entrenamiento_s"] = tiempo
    rep["sklearn_version"] = sklearn.__version__
    return rep


def guardar_resultados(
    tabla_validacion: pd.DataFrame,
    tabla_test: pd.DataFrame,
    modelos: Dict[str, ModeloAjustado],
    carpeta: str = "salidas",
) -> Dict[str, str]:
    """Guarda solo resultados de comparacion; NO sobrescribe modelo.joblib."""
    os.makedirs(carpeta, exist_ok=True)
    ruta_val = os.path.join(carpeta, "comparacion_modelos_validacion.csv")
    ruta_test = os.path.join(carpeta, "comparacion_modelos_test.csv")
    ruta_json = os.path.join(carpeta, "comparacion_modelos.json")

    tabla_validacion.to_csv(ruta_val, index=False, encoding="utf-8-sig")
    tabla_test.to_csv(ruta_test, index=False, encoding="utf-8-sig")

    ganador = str(tabla_validacion.iloc[0]["modelo"])
    payload = {
        "criterio_seleccion": "F1 macro en validacion temporal",
        "modelo_mejor_validacion": ganador,
        "nota_test": "El test se evalua solo despues de congelar todas las configuraciones.",
        "entorno": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validacion": tabla_validacion.to_dict(orient="records"),
        "test": tabla_test.to_dict(orient="records"),
        "configuraciones": {
            nombre: modelo.configuracion for nombre, modelo in modelos.items()
        },
    }
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    return {"validacion": ruta_val, "test": ruta_test, "json": ruta_json}