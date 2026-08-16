"""
Componente 2 - Bosque aleatorio de arboles de decision, implementado desde cero.

Referencias del algoritmo: CART (Breiman, Friedman, Olshen y Stone, 1984) y
Random Forests (Breiman, 2001). NumPy se usa solo como libreria de algebra
matricial: la induccion del arbol, el criterio de division, el muestreo
bootstrap y la agregacion estan escritos aqui.

Decision de implementacion que hace tratable n ~ 4.3e6:
la busqueda del mejor corte NO ordena el atributo en cada nodo. Como X ya viene
discretizada en <= 32 bins por cuantiles (ver caracteristicas.py), en cada nodo
se acumula un histograma de peso por (bin, clase) con np.bincount y se evaluan
los 31 cortes candidatos de una sola vez con una suma acumulada. El costo por
nodo pasa de O(|Q| log|Q|) a O(|Q|), y la matriz vive en uint8 en lugar de
float64. Esto es lo que se reporta como estrategia de muestreo y de memoria.
"""
import time

import numpy as np


class NodoArbol:
    """Nodo del arbol (Figura 9 de la Tarea #4)."""

    __slots__ = ("indiceAtributo", "umbral", "izquierdo", "derecho",
                 "distribucionClases", "esHoja")

    def __init__(self):
        self.indiceAtributo = -1
        self.umbral = -1
        self.izquierdo = None
        self.derecho = None
        self.distribucionClases = None
        self.esHoja = False


def giniPonderado(pesos_por_clase):
    """
    Gini(Q) = 1 - suma_k pi_k^2 ,  pi_k = W_k / W

    Acepta un vector (3,) o una matriz (m, 3) y devuelve escalar o vector.
    """
    W = pesos_por_clase.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi = pesos_por_clase / W[..., None]
    g = 1.0 - np.sum(pi ** 2, axis=-1)
    return np.where(W > 0, g, 0.0)


class ArbolDecisionPropio:
    """Arbol CART de clasificacion con impureza de Gini ponderada por clase."""

    def __init__(self, profundidadMax=12, minMuestrasHoja=5, mAtributos=None,
                 nBins=32, nClases=3, rng=None):
        self.profundidadMax = profundidadMax
        self.minMuestrasHoja = minMuestrasHoja
        self.mAtributos = mAtributos
        self.nBins = nBins
        self.nClases = nClases
        self.rng = rng if rng is not None else np.random.default_rng()
        self.raiz = None
        self.importancias = None
        # arreglos planos compilados, para predecir de forma vectorizada
        self._atr = self._umb = self._izq = self._der = self._val = None

    # -- criterio de division ----------------------------------------------
    def mejorCorte(self, Xb, y, w, idx, columnas):
        """
        Devuelve (atributo, umbral_bin, reduccion_impureza) o None.

        Para cada atributo candidato acumula el histograma ponderado por
        (bin, clase) y evalua todos los cortes con una suma acumulada.
        """
        yi = y[idx]
        wi = w[yi]
        Wtot = np.bincount(yi, weights=wi, minlength=self.nClases)
        Wsum = Wtot.sum()
        if Wsum <= 0:
            return None
        gini_padre = float(giniPonderado(Wtot))
        if gini_padre <= 0.0:
            return None

        mejor = None
        mejor_crit = gini_padre
        B, K = self.nBins, self.nClases

        for j in columnas:
            b = Xb[idx, j].astype(np.intp)

            hist = np.bincount(b * K + yi, weights=wi,
                               minlength=B * K).reshape(B, K)
            cnt = np.bincount(b, minlength=B)

            W_izq = np.cumsum(hist, axis=0)[:-1]          # (B-1, K)
            W_der = Wtot[None, :] - W_izq
            n_izq = np.cumsum(cnt)[:-1]
            n_der = idx.size - n_izq

            valido = ((n_izq >= self.minMuestrasHoja) &
                      (n_der >= self.minMuestrasHoja))
            if not valido.any():
                continue

            wl = W_izq.sum(axis=1)
            wr = W_der.sum(axis=1)
            crit = (wl * giniPonderado(W_izq) + wr * giniPonderado(W_der)) / Wsum
            crit = np.where(valido, crit, np.inf)

            u = int(np.argmin(crit))
            if crit[u] < mejor_crit - 1e-12:
                mejor_crit = float(crit[u])
                mejor = (int(j), u)

        if mejor is None:
            return None
        return mejor[0], mejor[1], (gini_padre - mejor_crit) * Wsum

    # -- induccion recursiva ------------------------------------------------
    def construir(self, Xb, y, w, idx=None, profundidad=0):
        if idx is None:                       # llamada raiz
            idx = np.arange(Xb.shape[0], dtype=np.int32)
            self.importancias = np.zeros(Xb.shape[1], dtype=np.float64)
            if self.mAtributos is None:
                self.mAtributos = max(1, int(np.sqrt(Xb.shape[1])))
            self.raiz = self.construir(Xb, y, w, idx, 0)
            self._compilar()
            return self.raiz

        nodo = NodoArbol()
        yi = y[idx]
        Wk = np.bincount(yi, weights=w[yi], minlength=self.nClases)
        dist = Wk / Wk.sum() if Wk.sum() > 0 else np.full(self.nClases, 1 / self.nClases)

        detener = (profundidad >= self.profundidadMax
                   or idx.size < 2 * self.minMuestrasHoja
                   or np.count_nonzero(np.bincount(yi, minlength=self.nClases)) == 1)
        if not detener:
            columnas = self.rng.choice(Xb.shape[1], size=self.mAtributos, replace=False)
            corte = self.mejorCorte(Xb, y, w, idx, columnas)
        else:
            corte = None

        if corte is None:
            nodo.esHoja = True
            nodo.distribucionClases = dist.astype(np.float32)
            return nodo

        j, u, ganancia = corte
        self.importancias[j] += ganancia

        va_izq = Xb[idx, j] <= u
        nodo.indiceAtributo, nodo.umbral = j, u
        nodo.distribucionClases = dist.astype(np.float32)
        nodo.izquierdo = self.construir(Xb, y, w, idx[va_izq], profundidad + 1)
        nodo.derecho = self.construir(Xb, y, w, idx[~va_izq], profundidad + 1)
        return nodo

    # -- compilacion a arreglos planos --------------------------------------
    def _compilar(self):
        atr, umb, izq, der, val = [], [], [], [], []

        def visitar(n):
            i = len(atr)
            atr.append(-1); umb.append(0); izq.append(-1); der.append(-1)
            val.append(n.distribucionClases)
            if not n.esHoja:
                atr[i] = n.indiceAtributo
                umb[i] = n.umbral
                izq[i] = visitar(n.izquierdo)
                der[i] = visitar(n.derecho)
            return i

        visitar(self.raiz)
        self._atr = np.array(atr, dtype=np.int32)
        self._umb = np.array(umb, dtype=np.int16)
        self._izq = np.array(izq, dtype=np.int32)
        self._der = np.array(der, dtype=np.int32)
        self._val = np.array(val, dtype=np.float32)

    def predecirProba(self, Xb):
        """Descenso por niveles, vectorizado sobre todas las filas a la vez."""
        n = Xb.shape[0]
        nodo = np.zeros(n, dtype=np.int32)
        for _ in range(self.profundidadMax + 2):
            interno = self._atr[nodo] >= 0
            if not interno.any():
                break
            fil = np.nonzero(interno)[0]
            nd = nodo[fil]
            valores = Xb[fil, self._atr[nd]]
            izquierda = valores <= self._umb[nd]
            nodo[fil] = np.where(izquierda, self._izq[nd], self._der[nd])
        return self._val[nodo]

    @property
    def nNodos(self):
        return 0 if self._atr is None else len(self._atr)

    @property
    def profundidadReal(self):
        def prof(n):
            return 0 if n.esHoja else 1 + max(prof(n.izquierdo), prof(n.derecho))
        return prof(self.raiz) if self.raiz else 0


class BosqueAleatorioPropio:
    """Agregacion bootstrap de arboles con sorteo de atributos por nodo."""

    def __init__(self, nArboles=100, profundidadMax=12, minMuestrasHoja=5,
                 mAtributos=None, maxMuestrasPorArbol=200_000, nBins=32,
                 nClases=3, semilla=42, potenciaPesos=1.0, verboso=True):
        self.nArboles = nArboles
        self.profundidadMax = profundidadMax
        self.minMuestrasHoja = minMuestrasHoja
        self.mAtributos = mAtributos
        self.maxMuestrasPorArbol = maxMuestrasPorArbol
        self.nBins = nBins
        self.nClases = nClases
        self.semilla = semilla
        self.potenciaPesos = potenciaPesos
        self.verboso = verboso
        self.arboles = []
        self.pesosClase = None
        self.tiempo_entrenamiento = None
        self.historial = []

    # -- pesos de clase w_k = (n / (K * n_k))^alpha -------------------------
    @staticmethod
    def pesosDeClase(y, nClases=3, potencia=1.0):
        """
        potencia = 1.0 reproduce w_k = n/(3 n_k) tal como quedo en el diseno.

        Con un desbalance mucho mayor al previsto (la clase alta puede quedar
        por debajo del 1%), w_alto supera 60 y el bosque sobre-predice la clase
        alta: sube el recall pero hunde la precision. La potencia suaviza esa
        compensacion y se elige SOLO con el conjunto de validacion.
        """
        cnt = np.bincount(y, minlength=nClases).astype(np.float64)
        n = cnt.sum()
        w = np.where(cnt > 0, n / (nClases * np.maximum(cnt, 1)), 0.0)
        return w ** potencia

    def muestraBootstrap(self, n, rng):
        tam = min(self.maxMuestrasPorArbol, n)
        return rng.integers(0, n, size=tam, dtype=np.int64)

    def entrenar(self, Xb, y, pesosClase=None):
        n, d = Xb.shape
        self.pesosClase = (self.pesosDeClase(y, self.nClases, self.potenciaPesos)
                           if pesosClase is None else np.asarray(pesosClase, float))
        if self.mAtributos is None:
            self.mAtributos = max(1, int(np.sqrt(d)))

        if self.verboso:
            print(f"  entrenando {self.nArboles} arboles | n={n:,} d={d} "
                  f"m={self.mAtributos} submuestra={min(self.maxMuestrasPorArbol, n):,}")
            print(f"  pesos de clase w_k = {np.round(self.pesosClase, 4)}")

        rng_maestro = np.random.default_rng(self.semilla)
        t0 = time.perf_counter()
        self.arboles = []
        for b in range(self.nArboles):
            rng = np.random.default_rng(rng_maestro.integers(0, 2 ** 62))
            sel = self.muestraBootstrap(n, rng)
            Xb_b, y_b = Xb[sel], y[sel]

            arbol = ArbolDecisionPropio(
                profundidadMax=self.profundidadMax,
                minMuestrasHoja=self.minMuestrasHoja,
                mAtributos=self.mAtributos,
                nBins=self.nBins, nClases=self.nClases, rng=rng)
            arbol.construir(Xb_b, y_b, self.pesosClase)
            self.arboles.append(arbol)

            if self.verboso and ((b + 1) % 10 == 0 or b == 0):
                dt = time.perf_counter() - t0
                print(f"    arbol {b+1:3d}/{self.nArboles}  "
                      f"nodos={arbol.nNodos:5d}  prof={arbol.profundidadReal:2d}  "
                      f"{dt:6.1f}s  (proy. {dt/(b+1)*self.nArboles/60:.1f} min)")
            self.historial.append(time.perf_counter() - t0)

        self.tiempo_entrenamiento = time.perf_counter() - t0
        if self.verboso:
            print(f"  entrenamiento completo en {self.tiempo_entrenamiento/60:.2f} min")
        return self

    def predecirProba(self, Xb, bloque=500_000):
        """f(x) = (1/B) suma_b p^(b)(x). Se procesa por bloques por memoria."""
        n = Xb.shape[0]
        out = np.zeros((n, self.nClases), dtype=np.float32)
        for i in range(0, n, bloque):
            j = min(i + bloque, n)
            acum = np.zeros((j - i, self.nClases), dtype=np.float32)
            for arbol in self.arboles:
                acum += arbol.predecirProba(Xb[i:j])
            out[i:j] = acum / len(self.arboles)
        return out

    def predecir(self, Xb):
        return np.argmax(self.predecirProba(Xb), axis=1).astype(np.uint8)

    def importanciaVariables(self):
        imp = np.sum([a.importancias for a in self.arboles], axis=0)
        return imp / imp.sum() if imp.sum() > 0 else imp
