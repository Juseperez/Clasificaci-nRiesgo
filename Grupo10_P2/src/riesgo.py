"""
Componente 3 (parte de datos) - Matriz de riesgo parroquia-dia.

Produce, para la semana objetivo, el nivel de riesgo y la probabilidad de la
clase predicha de cada parroquia-dia, y la deja en un formato que la interfaz
grafica pueda leer sin reentrenar.
"""
import numpy as np
import pandas as pd

NIVELES = {0: "Bajo", 1: "Medio", 2: "Alto"}
COLORES = {0: "#2c7fb8", 1: "#fdae61", 2: "#d7191c", -1: "#bdbdbd"}


class MatrizRiesgo:
    """
    Guarda nivel y probabilidad por (parroquia, dia de semana) para una semana objetivo.

    Las unidades sin incidentes historicos NO pasan por el modelo: se marcan con
    nivel -1 y se dibujan en gris como "sin datos suficientes" (riesgo no
    estimado), porque la ausencia de registros puede deberse a falta de
    cobertura y no equivale a riesgo bajo.
    """

    def __init__(self, meta, semana_objetivo, umbrales=None):
        self.meta = meta
        self.semana_objetivo = pd.Timestamp(semana_objetivo)
        self.umbrales = umbrales
        self.nivel = None
        self.probabilidad = None
        self.proba_clases = None

    def desde_predicciones(self, proba, mascara_sin_datos=None):
        """proba: (G*T, 3) en el orden (unidad, dia de semana)."""
        G, T = self.meta["G"], self.meta["T"]
        p = np.asarray(proba, dtype=np.float32).reshape(G, T, 3).copy()
        self.nivel = p.argmax(axis=2).astype(np.int8)
        self.probabilidad = p.max(axis=2)
        if mascara_sin_datos is not None:
            self.nivel[mascara_sin_datos] = -1
            self.probabilidad[mascara_sin_datos] = np.nan
            # Tambien las probabilidades por clase: una unidad que no paso por
            # el modelo no tiene distribucion estimada, y mostrar ceros daria a
            # entender que el clasificador si opino sobre ella.
            p[mascara_sin_datos] = np.nan
        self.proba_clases = p
        return self

    def a_dataframe(self):
        G, T = self.meta["G"], self.meta["T"]
        g, t = np.meshgrid(np.arange(G), np.arange(T), indexing="ij")
        dias = self.meta.get("dias", ["Lunes", "Martes", "Miercoles", "Jueves",
                                      "Viernes", "Sabado", "Domingo"])
        return pd.DataFrame({
            "unidad_id": np.asarray(self.meta["unidades"])[g].ravel(),
            "unidad": np.asarray(self.meta["nombre_unidad"])[g].ravel(),
            "canton": np.asarray(self.meta["canton_unidad"])[g].ravel(),
            "unidad_idx": g.ravel(),
            "dia_idx": t.ravel(),
            "dia_semana": [dias[k] for k in t.ravel()],
            "nivel_cod": self.nivel.ravel(),
            "nivel": [NIVELES.get(int(k), "Sin datos suficientes")
                      for k in self.nivel.ravel()],
            "prob_clase_predicha": self.probabilidad.ravel(),
            "p_bajo": self.proba_clases[:, :, 0].ravel(),
            "p_medio": self.proba_clases[:, :, 1].ravel(),
            "p_alto": self.proba_clases[:, :, 2].ravel(),
            "semana_objetivo": self.semana_objetivo.date().isoformat(),
        })

    def guardar(self, ruta="salidas/matriz_riesgo.parquet"):
        df = self.a_dataframe()
        try:
            df.to_parquet(ruta, index=False)
        except Exception:
            ruta = ruta.replace(".parquet", ".csv")
            df.to_csv(ruta, index=False)
        print(f"  matriz de riesgo guardada en {ruta} ({len(df):,} filas)")
        return ruta

    def resumen(self):
        v, k = np.unique(self.nivel.ravel(), return_counts=True)
        return {NIVELES.get(int(a), "Sin datos suficientes"): int(b) for a, b in zip(v, k)}


def predecir_matriz_semana(bosque, X, indices, mask_elegible):
    """
    Predice la semana objetivo pasando por el modelo UNICAMENTE las unidades
    elegibles.

    La Tarea #4 dice que las unidades sin incidentes historicos no ingresan al
    modelo. Predecir las G*T combinaciones y pintar despues en gris las no
    elegibles daria el mismo resultado visual, pero haria falsa esa afirmacion:
    el clasificador si habria opinado sobre ellas. Aqui solo se evaluan las
    filas elegibles y el resto de la matriz queda en NaN.

    Devuelve (proba, n_predichas) con proba de forma (G*T, 3).
    """
    indices = np.asarray(indices)
    elegibles = np.asarray(mask_elegible).reshape(-1)
    if indices.size != elegibles.size:
        raise ValueError(f"indices ({indices.size}) y mascara ({elegibles.size}) "
                         "deben cubrir las mismas G*T combinaciones")

    proba = np.full((indices.size, 3), np.nan, dtype=np.float32)
    n = int(elegibles.sum())
    if n:
        proba[elegibles] = bosque.predecirProba(X[indices[elegibles]])
    return proba, n


def resumen_unidades(meta):
    """Catalogo de unidades espaciales para la interfaz."""
    return [{"id": str(u), "nombre": str(n), "canton": str(ct)}
            for u, n, ct in zip(meta["unidades"], meta["nombre_unidad"],
                                meta["canton_unidad"])]
