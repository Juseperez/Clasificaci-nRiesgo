"""
Evaluacion del modelo y baselines obligatorios.

Todas las metricas estan implementadas aqui: el criterio de exito declarado en
la Tarea #4 es superar de forma consistente a los baselines sobre este mismo
dataset, no alcanzar un F1 macro tomado de otro contexto.
"""
import numpy as np

NOMBRES_CLASE = ("bajo", "medio", "alto")


class EvaluadorMetricas:

    @staticmethod
    def matrizConfusion(y, yPred, nClases=3):
        M = np.bincount(y.astype(np.int64) * nClases + yPred.astype(np.int64),
                        minlength=nClases * nClases)
        return M.reshape(nClases, nClases)

    @staticmethod
    def precisionRecallF1(y, yPred, nClases=3):
        M = EvaluadorMetricas.matrizConfusion(y, yPred, nClases)
        vp = np.diag(M).astype(float)
        pred = M.sum(axis=0).astype(float)
        real = M.sum(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(pred > 0, vp / pred, 0.0)
            rec = np.where(real > 0, vp / real, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        return prec, rec, f1

    @staticmethod
    def f1Macro(y, yPred, nClases=3):
        return float(EvaluadorMetricas.precisionRecallF1(y, yPred, nClases)[2].mean())

    @staticmethod
    def exactitud(y, yPred):
        return float((y == yPred).mean())

    @staticmethod
    def aucROC_ovr(y, proba, nClases=3):
        """AUC uno-contra-resto por clase, calculada con rangos (Mann-Whitney)."""
        aucs = []
        for k in range(nClases):
            pos = (y == k)
            n_pos, n_neg = int(pos.sum()), int((~pos).sum())
            if n_pos == 0 or n_neg == 0:
                aucs.append(np.nan)
                continue
            s = proba[:, k]
            orden = np.argsort(s, kind="mergesort")
            rangos = np.empty(len(s), dtype=np.float64)
            rangos[orden] = np.arange(1, len(s) + 1)
            # promedio de rangos en los empates
            s_ord = s[orden]
            i = 0
            while i < len(s_ord):
                j = i
                while j + 1 < len(s_ord) and s_ord[j + 1] == s_ord[i]:
                    j += 1
                if j > i:
                    rangos[orden[i:j + 1]] = (i + j + 2) / 2.0
                i = j + 1
            aucs.append((rangos[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        return np.array(aucs)

    @staticmethod
    def brier(y, proba, nClases=3):
        """Brier multiclase: media del error cuadratico sobre el vector one-hot."""
        oh = np.zeros_like(proba)
        oh[np.arange(len(y)), y] = 1.0
        return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))

    @staticmethod
    def curvaCalibracion(y, proba, nBins=10):
        """
        Confianza declarada vs. acierto observado, sobre max_k f_k(x).
        Es lo que la interfaz muestra al usuario, asi que debe estar calibrado.
        """
        conf = proba.max(axis=1)
        pred = proba.argmax(axis=1)
        acierto = (pred == y).astype(float)
        bordes = np.linspace(0, 1, nBins + 1)
        idx = np.clip(np.digitize(conf, bordes[1:-1]), 0, nBins - 1)
        xs, ys, ns = [], [], []
        for b in range(nBins):
            m = idx == b
            if m.sum() > 0:
                xs.append(conf[m].mean())
                ys.append(acierto[m].mean())
                ns.append(int(m.sum()))
        return np.array(xs), np.array(ys), np.array(ns)

    @classmethod
    def reporte(cls, y, yPred, proba=None, titulo="modelo"):
        prec, rec, f1 = cls.precisionRecallF1(y, yPred)
        d = {
            "modelo": titulo,
            "exactitud": cls.exactitud(y, yPred),
            "f1_macro": float(f1.mean()),
            "recall_alto": float(rec[2]),
            "precision_alto": float(prec[2]),
            "f1_por_clase": f1,
            "precision_por_clase": prec,
            "recall_por_clase": rec,
            "matriz_confusion": cls.matrizConfusion(y, yPred),
        }
        if proba is not None:
            d["auc_ovr"] = cls.aucROC_ovr(y, proba)
            d["brier"] = cls.brier(y, proba)
        return d

    @staticmethod
    def imprimir(d):
        print(f"\n  --- {d['modelo']} ---")
        print(f"  exactitud {d['exactitud']:.4f}   F1 macro {d['f1_macro']:.4f}   "
              f"recall clase alta {d['recall_alto']:.4f}")
        print(f"  {'clase':8s} {'prec':>8s} {'recall':>8s} {'F1':>8s}")
        for k, n in enumerate(NOMBRES_CLASE):
            print(f"  {n:8s} {d['precision_por_clase'][k]:8.4f} "
                  f"{d['recall_por_clase'][k]:8.4f} {d['f1_por_clase'][k]:8.4f}")
        if "brier" in d:
            print(f"  Brier {d['brier']:.4f}   AUC OvR {np.round(d['auc_ovr'], 4)}")
        print("  matriz de confusion (filas = real, columnas = predicho):")
        for k, fila in enumerate(d["matriz_confusion"]):
            print(f"    {NOMBRES_CLASE[k]:6s} {fila}")


# ---------------------------------------------------------------------------
class Baselines:
    """
    Los tres pisos de comparacion declarados en la Tarea #4. Todos operan sobre
    el tensor de etiquetas, no sobre X, porque no aprenden nada de las variables.
    """

    @staticmethod
    def trivial(y_train, n_test):
        """Predecir siempre la clase mayoritaria del entrenamiento."""
        k = int(np.bincount(y_train, minlength=3).argmax())
        return np.full(n_test, k, dtype=np.uint8)

    @staticmethod
    def persistencia(y_tensor, semanas):
        """Repetir la clase observada la semana anterior en la misma unidad-dia."""
        return y_tensor[:, :, np.asarray(semanas) - 1].reshape(-1)

    @staticmethod
    def moda_historica(y_tensor, semanas_train, semanas_obj):
        """Moda de la etiqueta de cada unidad-dia durante el entrenamiento."""
        ytr = y_tensor[:, :, semanas_train]
        G, T, _ = ytr.shape
        conteos = np.stack([(ytr == k).sum(axis=2) for k in range(3)], axis=-1)
        moda = conteos.argmax(axis=-1).astype(np.uint8)          # (G, T)
        return np.repeat(moda[:, :, None], len(semanas_obj), axis=2).reshape(-1)
