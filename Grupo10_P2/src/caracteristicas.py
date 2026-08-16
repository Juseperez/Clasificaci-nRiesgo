"""
Particion temporal, etiqueta por percentiles y matriz de caracteristicas X.

Orden critico del pipeline (evita fuga de informacion):
    tensor c  ->  PARTICION TEMPORAL  ->  mascara de elegibilidad (solo train)
              ->  umbrales solo con c_train  ->  aplicar a train/val/test
              ->  construir X  ->  entrenar

X se guarda directamente discretizada en enteros de 8 bits (bins por cuantiles
ajustados solo con entrenamiento). Eso reduce la memoria de la matriz por un
factor de 8 frente a float64.
"""
import numpy as np
import pandas as pd

N_BINS = 32          # umbrales candidatos por variable (Tarea #4, seccion 5)
N_REZAGOS = 4
VENTANAS = (4, 12)
BURN_IN = 12         # semanas iniciales sin instancia (media movil de 12)


class ParticionadorTemporal:
    """Cortes cronologicos sin solapamiento. Sin validacion cruzada aleatoria."""

    def __init__(self, corte_train="2024-12-31", corte_val="2025-06-30"):
        self.corte_train = pd.Timestamp(corte_train)
        self.corte_val = pd.Timestamp(corte_val)

    def dividir(self, meta):
        """Devuelve indices de SEMANA para train / val / test."""
        fechas = pd.to_datetime(pd.Series(meta["fecha_semana"]))
        s = np.arange(meta["S"])
        usable = s >= BURN_IN
        tr = usable & (fechas <= self.corte_train).values
        va = usable & (fechas > self.corte_train).values & (fechas <= self.corte_val).values
        te = usable & (fechas > self.corte_val).values
        idx = (s[tr], s[va], s[te])
        print(f"  particion temporal -> train={len(idx[0])} val={len(idx[1])} "
              f"test={len(idx[2])} semanas")
        if min(len(i) for i in idx) == 0:
            print("  [aviso] alguna particion quedo vacia: revisa los cortes "
                  "contra el rango real de datos")
        return idx


# ---------------------------------------------------------------------------
def mascara_elegibilidad(c, semanas_train, nivel="unidad"):
    """
    Que instancias ingresan al modelo, decidido SOLO con la ventana de
    entrenamiento (Tarea #4: "las unidades que carecen de incidentes no ingresan
    al modelo... y se excluyen de las metricas").

    nivel="unidad"     -> se excluye la parroquia completa si no tuvo ningun
                          incidente durante el entrenamiento. Es la regla
                          literal de la Tarea #4, trasladada a la unidad real.
    nivel="unidad_dia" -> se excluye cada par parroquia-dia sin incidentes.
                          Mas estricto: elimina tambien dias genuinamente
                          tranquilos de parroquias activas, que si son
                          informacion valida de riesgo bajo.

    Devuelve una mascara booleana (G, T): True = elegible.

    Nota: una parroquia que aparece por primera vez en 2025 queda excluida aunque
    tenga eventos, porque durante el entrenamiento no los tenia. Es deliberado:
    el modelo no puede aprender nada de ella y sus miles de filas en cero solo
    inflarian la clase baja.
    """
    G, T, _ = c.shape
    ctr = c[:, :, semanas_train]
    if nivel == "unidad":
        elegible = ctr.sum(axis=(1, 2)) > 0
        mask = np.broadcast_to(elegible[:, None], (G, T)).copy()
    elif nivel == "unidad_dia":
        mask = ctr.sum(axis=2) > 0
    else:
        raise ValueError("nivel debe ser 'unidad' o 'unidad_dia'")
    print(f"  elegibilidad ({nivel}): {mask.sum():,} de {G*T:,} pares unidad-dia "
          f"({mask.mean()*100:.1f}%) ingresan al modelo")
    return mask


def indices_de_semanas(G, T, S_total, semanas):
    """Indices de fila de X/y correspondientes a un conjunto de semanas."""
    mask = np.zeros(S_total, dtype=bool)
    mask[np.asarray(semanas, dtype=int)] = True
    return np.nonzero(np.broadcast_to(mask, (G, T, S_total)).reshape(-1))[0]


def filtrar_elegibles(indices, mask_gt, T, S_total):
    """Se queda solo con las filas cuya unidad-dia es elegible."""
    g = indices // (T * S_total)
    t = (indices // S_total) % T
    return indices[mask_gt[g, t]]


# ---------------------------------------------------------------------------
class ConstructorEtiquetas:
    """
    y = 0 (bajo) si c <= u_bajo ; 1 (medio) si u_bajo < c <= u_alto ; 2 (alto) si c > u_alto

    Los umbrales se ajustan SOLO con la ventana de entrenamiento.

    El conteo por unidad-dia-semana es una variable DISCRETA fuertemente
    cero-inflada. Cuando la mayor parte de la masa cae en un unico valor, los
    percentiles P60 y P90 coinciden y la clase media queda vacia: no existe
    ningun par de umbrales sobre el conteo que produzca las proporciones
    60/30/10, porque las unicas proporciones alcanzables son las que permite la
    propia distribucion. Por eso el ajuste recorre una escalera de alternativas,
    en el orden que autoriza la Tarea #4, y declara en cual se detuvo:

        1. global      percentiles sobre todos los conteos de entrenamiento
        2. positivos   percentiles sobre los conteos > 0
        3. por_dia     percentiles calculados por dia de la semana
        4. discreto    cortes enteros forzados sobre la distribucion acumulada

    El paso 4 NO es un percentil: es un umbral operativo que solo garantiza tres
    clases no vacias. Si el ajuste llega ahi, `corresponde_a_percentiles` queda
    en False y `percentil_real_alto` indica a que percentil equivale de verdad
    el corte superior. Eso es lo que hay que reportar, en lugar de seguir
    llamandolo P90.
    """

    ESCALERA = ("global", "positivos", "por_dia", "discreto")

    def __init__(self, p_bajo=60, p_alto=90, modo="auto", min_frac_clase=0.001,
                 max_dias_colapsados=0):
        self.p_bajo, self.p_alto = p_bajo, p_alto
        self.modo = modo
        self.min_frac_clase = min_frac_clase
        self.max_dias_colapsados = max_dias_colapsados
        self.dias_colapsados = None
        self.base_referencia = None
        self.alto_supera_medio = None
        self.masa_acumulada_bajo = None
        self.masa_acumulada_alto = None
        self.P60 = self.P90 = None          # nombres heredados del reporte
        self.modo_usado = None
        self.percentil_real_bajo = self.percentil_real_alto = None  # obsoletos
        self.por_dia = False
        self.corresponde_a_percentiles = None
        self.percentil_real_bajo = None
        self.percentil_real_alto = None
        self.proporciones = None
        self.diagnostico = []

    # -- helpers ------------------------------------------------------------
    def _proporciones(self, ctr, u_bajo, u_alto, dias=None):
        if dias is None:
            y = (ctr > u_bajo).astype(np.int8) + (ctr > u_alto).astype(np.int8)
        else:
            ub = np.asarray(u_bajo)[dias]
            ua = np.asarray(u_alto)[dias]
            y = (ctr > ub).astype(np.int8) + (ctr > ua).astype(np.int8)
        return np.bincount(y.ravel(), minlength=3) / y.size

    @staticmethod
    def _percentil_de(valores, umbral):
        """A que percentil empirico equivale realmente un umbral."""
        return float((valores <= umbral).mean() * 100)

    def _viable(self, props, extras=None):
        """
        Un modo es viable si produce las tres clases con masa suficiente y, en el
        caso de por_dia, si ningun dia de la semana pierde individualmente la clase media.

        NO se exige que la clase alta sea menos frecuente que la media. En una
        variable discreta eso puede ocurrir con los percentiles correctos: con
        85% de ceros, 5% de unos y 10% de doses, np.percentile da P60=0 y
        P90=1.1, y las clases quedan 85/5/10. "Alto" sigue siendo exactamente el
        extremo superior; no hay ninguna inversion. Rechazarlo seria descartar un
        etiquetado legitimo. La comparacion se conserva solo como diagnostico.
        """
        if not (props >= self.min_frac_clase).all():
            return False
        if extras and "dias_colapsados" in extras:
            return extras["dias_colapsados"] <= self.max_dias_colapsados
        return True

    # -- ajuste -------------------------------------------------------------
    def ajustarUmbrales(self, c, semanas_train, mask_elegible=None):
        G, T, _ = c.shape
        ctr_full = c[:, :, semanas_train]
        dia_gt = np.broadcast_to(np.arange(T)[None, :], (G, T))

        if mask_elegible is not None:
            ctr = ctr_full[mask_elegible]
            dias = np.repeat(dia_gt[mask_elegible][:, None], ctr.shape[1], axis=1)
        else:
            ctr = ctr_full.reshape(-1, ctr_full.shape[2])
            dias = np.repeat(dia_gt.reshape(-1)[:, None], ctr.shape[1], axis=1)
        plano = ctr.ravel()

        print(f"  conteos de entrenamiento (solo elegibles): {plano.size:,} | "
              f"{(plano == 0).mean()*100:.1f}% en cero | media={plano.mean():.4f} | "
              f"max={int(plano.max())}")
        print("  escalera de umbrales:")

        candidatos = self.ESCALERA if self.modo == "auto" else (self.modo,)
        for modo in candidatos:
            res = self._probar(modo, plano, ctr, dias)
            if res is None:
                self.diagnostico.append((modo, "no aplicable", None))
                print(f"    [{modo:10s}] no aplicable")
                continue
            u_b, u_a, props, por_dia, extras = res
            ok = self._viable(props, extras)
            motivos = []
            if not (props >= self.min_frac_clase).all():
                motivos.append("clase vacia")
            detalle = ""
            if "dias_colapsados" in extras:
                if extras["dias_colapsados"] > self.max_dias_colapsados:
                    motivos.append(f"{extras['dias_colapsados']}/"
                                   f"{extras['dias_totales']} dias sin clase media")
                detalle = (f" | {extras['dias_colapsados']}/{extras['dias_totales']} "
                           f"dias con P{self.p_bajo}=P{self.p_alto}")
            if motivos and not ok:
                detalle += "  <- " + "; ".join(motivos)
            self.diagnostico.append((modo, "viable" if ok else "degenera",
                                     np.round(props * 100, 3).tolist(),
                                     extras.get("dias_colapsados")))
            print(f"    [{modo:10s}] {'OK     ' if ok else 'colapsa'} "
                  f"umbrales={_fmt(u_b)}/{_fmt(u_a)} -> {np.round(props*100, 2)} %{detalle}")
            if ok:
                self.dias_colapsados = extras.get("dias_colapsados")
                self._fijar(modo, u_b, u_a, props, plano, por_dia,
                            base=extras.get("base", "todos"))
                return self.P60, self.P90

        raise ValueError(
            "Ningun modo produjo tres clases no vacias. Con estos datos el "
            "problema de tres niveles no es formulable tal como esta definido: "
            "agrega mas meses, usa nivel_espacial='canton', o reduce a "
            "dos clases y declaralo explicitamente en el reporte.")

    def _probar(self, modo, plano, ctr, dias):
        if modo == "global":
            u_b, u_a = np.percentile(plano, [self.p_bajo, self.p_alto])
            return float(u_b), float(u_a), self._proporciones(ctr, u_b, u_a), False, {"base": "todos"}

        if modo == "positivos":
            pos = plano[plano > 0]
            if pos.size == 0:
                return None
            u_b, u_a = np.percentile(pos, [self.p_bajo, self.p_alto])
            return (float(u_b), float(u_a), self._proporciones(ctr, u_b, u_a), False,
                    {"base": "positivos"})

        if modo == "por_dia":
            T = int(dias.max()) + 1
            u_b, u_a = np.zeros(T), np.zeros(T)
            for t in range(T):
                v = ctr[dias == t]
                if v.size:
                    u_b[t], u_a[t] = np.percentile(v, [self.p_bajo, self.p_alto])
            colapsadas = int((u_b == u_a).sum())
            return (u_b, u_a, self._proporciones(ctr, u_b, u_a, dias), True,
                    {"dias_colapsados": colapsadas, "dias_totales": T,
                     "base": "por dia"})

        if modo == "discreto":
            vals, cnts = np.unique(plano, return_counts=True)
            if vals.size < 3:
                return None
            F = np.cumsum(cnts) / cnts.sum()
            i0 = min(int(np.searchsorted(F, self.p_bajo / 100.0, side="left")),
                     vals.size - 2)
            i1 = int(np.searchsorted(F, self.p_alto / 100.0, side="left"))
            i1 = min(max(i1, i0 + 1), vals.size - 1)
            u_b, u_a = float(vals[i0]), float(vals[i1])
            return (u_b, u_a, self._proporciones(ctr, u_b, u_a), False,
                    {"base": "todos (cortes enteros forzados)"})

        raise ValueError(f"modo desconocido: {modo}")

    def _fijar(self, modo, u_b, u_a, props, plano, por_dia, base="todos"):
        self.P60, self.P90 = u_b, u_a
        self.modo_usado = modo
        self.proporciones = props
        self.por_dia = por_dia
        self.base_referencia = base
        self.alto_supera_medio = bool(props[2] > props[1])

        # Los tres primeros peldanos obtienen los umbrales con np.percentile, asi
        # que SON los percentiles pedidos por construccion. Solo "discreto" los
        # abandona. La masa acumulada F(u) es un diagnostico distinto: indica que
        # fraccion de los conteos queda por debajo del umbral, y en una variable
        # discreta puede ser muy superior al nivel percentilar sin que haya
        # contradiccion (con 85% de ceros, el valor 0 es a la vez el percentil
        # 60, el 70 y el 80).
        self.corresponde_a_percentiles = (modo != "discreto")

        if not por_dia:
            self.masa_acumulada_bajo = self._percentil_de(plano, u_b)
            self.masa_acumulada_alto = self._percentil_de(plano, u_a)

        print(f"\n  modo adoptado: {modo}  (base de calculo: {base})")
        print(f"  umbrales: bajo <= {_fmt(u_b)}   alto > {_fmt(u_a)}")
        print(f"  proporciones alcanzadas -> bajo {props[0]*100:.2f}%  "
              f"medio {props[1]*100:.2f}%  alto {props[2]*100:.2f}%")
        if not por_dia:
            print(f"  masa acumulada F(u) en los umbrales: "
                  f"{self.masa_acumulada_bajo:.2f}% y {self.masa_acumulada_alto:.2f}%")

        if self.corresponde_a_percentiles:
            print(f"\n  Los umbrales SON los percentiles {self.p_bajo}/{self.p_alto}, "
                  f"calculados con np.percentile sobre la base '{base}'. La definicion "
                  "de la etiqueta de la Tarea #4 se conserva.")
            if base != "todos":
                print(f"  Precision para el reporte: la base es '{base}', no la "
                      "distribucion completa. Reportar las proporciones de arriba.")
        else:
            print(f"\n  [IMPORTANTE PARA EL REPORTE] los umbrales NO son percentiles: "
                  "son cortes enteros forzados sobre la distribucion acumulada, adoptados "
                  "porque los tres peldanos anteriores degeneraban. La definicion de la "
                  "etiqueta cambia respecto de la Tarea #4 y hay que declararlo, junto "
                  "con la regla usada y las proporciones resultantes.")

        if self.alto_supera_medio:
            print(f"\n  [nota] la clase alta ({props[2]*100:.2f}%) resulta mas frecuente "
                  f"que la media ({props[1]*100:.2f}%). En una variable discreta esto es "
                  "legitimo y no implica inversion de la etiqueta: 'alto' sigue siendo el "
                  "extremo superior. Pero se aparta del 60/30/10 previsto, asi que "
                  "conviene mencionarlo al describir el desbalance.")

        if modo == "discreto":
            print("\n  [DECISION PENDIENTE] este es el peldano de respaldo. Antes de "
                  "aceptarlo como definitivo, revisar si hay mas meses "
                  "disponibles o si conviene agregar por canton: si asi los percentiles "
                  "dejan de colapsar, es metodologicamente mas limpio conservar la "
                  "definicion original de la etiqueta que cambiarla.")

    def construirY(self, c):
        if self.P60 is None:
            raise RuntimeError("Llama ajustarUmbrales antes de construirY.")
        ub, ua = np.asarray(self.P60), np.asarray(self.P90)
        if ub.ndim == 1:
            ub, ua = ub[None, :, None], ua[None, :, None]
        return ((c > ub).astype(np.uint8) + (c > ua).astype(np.uint8))

    @staticmethod
    def distribucion(y):
        cnt = np.bincount(y.ravel(), minlength=3)
        return cnt, cnt / cnt.sum()

    def resumen(self):
        return {
            "modo_usado": self.modo_usado,
            "umbral_bajo": (float(self.P60) if not self.por_dia
                            else np.asarray(self.P60).tolist()),
            "umbral_alto": (float(self.P90) if not self.por_dia
                            else np.asarray(self.P90).tolist()),
            "corresponde_a_percentiles_solicitados": self.corresponde_a_percentiles,
            "base_de_calculo": self.base_referencia,
            "masa_acumulada_umbral_bajo_pct": self.masa_acumulada_bajo,
            "masa_acumulada_umbral_alto_pct": self.masa_acumulada_alto,
            "alto_supera_medio": self.alto_supera_medio,
            "proporciones_pct": (None if self.proporciones is None
                                 else np.round(self.proporciones * 100, 3).tolist()),
            "dias_colapsados": self.dias_colapsados,
            "escalera_recorrida": self.diagnostico,
        }


def _fmt(u):
    a = np.asarray(u)
    return f"{float(a):g}" if a.ndim == 0 else f"vector[{a.size}]"


# ---------------------------------------------------------------------------
class Discretizador:
    """Bordes por cuantiles ajustados solo con entrenamiento."""

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.bordes = []

    def ajustar_columna(self, v_train):
        u = np.unique(v_train)
        if u.size <= self.n_bins:
            bordes = u[:-1] if u.size > 1 else np.array([u[0] - 1.0])
        else:
            q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            bordes = np.unique(np.quantile(v_train, q))
        self.bordes.append(bordes)
        return bordes

    @staticmethod
    def aplicar(v, bordes):
        """
        bin(v) = numero de bordes estrictamente menores que v, es decir, cada
        borde es el LIMITE SUPERIOR de su bin.

        Debe ser side="left". Con side="right" un valor igual a un borde cae en
        el bin siguiente, y como los bordes de una variable de baja cardinalidad
        son sus propios valores (menos el maximo), los dos valores mas altos
        colapsaban en el mismo bin: es_feriado y es_fin_de_semana quedaban
        constantes y no aportaban nada al modelo.
        """
        return np.searchsorted(bordes, v, side="left").astype(np.uint8)


class GeneradorCaracteristicas:
    """
    Construye X con informacion estrictamente anterior a la semana s.

    `semanas_futuras` extiende el tensor con semanas aun no observadas, para
    poder predecir de verdad la semana SIGUIENTE. Sus caracteristicas se
    calculan solo con rezagos y medias moviles de semanas ya observadas: la fila
    de la semana S usa c[S-1], c[S-2], ... como rezagos y nunca c[S], que es
    justamente el valor desconocido. El relleno con ceros no introduce
    informacion inexistente porque ninguna caracteristica lee la semana propia.
    """

    def __init__(self, rezagos=N_REZAGOS, ventanas=VENTANAS, n_bins=N_BINS):
        self.rezagos, self.ventanas, self.n_bins = rezagos, ventanas, n_bins
        self.nombres, self.bordes = [], []
        self.S_total = None
        self.semanas_futuras = 0

    @staticmethod
    def _rezago(c, k):
        out = np.zeros_like(c, dtype=np.float32)
        if k < c.shape[2]:
            out[:, :, k:] = c[:, :, :-k]
        return out

    @staticmethod
    def _media_movil(c, w):
        """Media de las w semanas ANTERIORES (no incluye s)."""
        cum = np.cumsum(c.astype(np.float64), axis=2)
        out = np.zeros(c.shape, dtype=np.float32)
        for s in range(1, c.shape[2]):
            a = max(0, s - w)
            out[:, :, s] = (cum[:, :, s - 1] - (cum[:, :, a - 1] if a > 0 else 0)) / (s - a)
        return out

    @staticmethod
    def _densidad_acumulada(c):
        """Media historica de la unidad espacial usando solo semanas < s."""
        por_unidad = c.sum(axis=1).astype(np.float64)
        cum = np.cumsum(por_unidad, axis=1)
        out = np.zeros((c.shape[0], c.shape[2]), dtype=np.float32)
        for s in range(1, c.shape[2]):
            out[:, s] = cum[:, s - 1] / s
        return out[:, None, :] * np.ones((1, c.shape[1], 1), dtype=np.float32)

    def construirX(self, c, meta, semanas_train, semanas_futuras=1,
                   mask_elegible=None):
        """
        Devuelve (X, nombres). X cubre (S + semanas_futuras) semanas por
        unidad-dia; la primera semana futura ocupa el indice s = S.

        `mask_elegible` (G, T) restringe el ajuste de los bordes de
        discretizacion a las filas que realmente ingresan al modelo. Sin ella,
        los cuantiles de las variables se calcularian tambien con unidades que
        la mascara de elegibilidad declaro fuera, lo que contradice "no ingresan
        al modelo": influirian en la discretizacion aunque luego se eliminen
        de X_train.
        """
        G, T, S = c.shape
        F = int(semanas_futuras)
        self.semanas_futuras = F
        self.S_total = S + F

        if F > 0:
            c = np.concatenate([c, np.zeros((G, T, F), dtype=c.dtype)], axis=2)
        St = c.shape[2]

        fechas = pd.to_datetime(pd.Series(meta["fecha_semana"]))
        if F > 0:
            extra = pd.Series([fechas.iloc[-1] + pd.Timedelta(weeks=k + 1)
                               for k in range(F)])
            fechas = pd.concat([fechas, extra], ignore_index=True)
        self.fechas_totales = fechas

        disc = Discretizador(self.n_bins)
        columnas, nombres = [], []
        mask_tr = np.zeros(St, dtype=bool)
        mask_tr[np.asarray(semanas_train, dtype=int)] = True
        sel = np.broadcast_to(mask_tr, (G, T, St))
        if mask_elegible is not None:
            sel = sel & mask_elegible[:, :, None]
            print(f"  bordes de discretizacion ajustados solo con filas elegibles "
                  f"({int(sel.sum()):,} de {int(np.broadcast_to(mask_tr,(G,T,St)).sum()):,})")
        sel = sel.reshape(-1)

        def agregar(nombre, tensor):
            v = tensor.reshape(-1)
            bordes = disc.ajustar_columna(v[sel])
            columnas.append(Discretizador.aplicar(v, bordes))
            nombres.append(nombre)
            self.bordes.append(bordes)

        # Calendario. El dataset publicado no trae hora, asi que la unidad
        # temporal es el dia de la semana y no existe bloque horario.
        dia_id = np.arange(T, dtype=np.float32)
        agregar("dia_semana", np.broadcast_to(dia_id[None, :, None], (G, T, St)).copy())
        agregar("es_fin_de_semana",
                np.broadcast_to((dia_id >= 5).astype(np.float32)[None, :, None],
                                (G, T, St)).copy())
        # Cada combinacion (t, s) tiene su fecha real: fecha = lunes_s + t dias.
        # Replicar el mes o el feriado del lunes sobre los siete dias es
        # incorrecto: una semana puede cruzar el cambio de mes, y un feriado del
        # miercoles no convierte en feriados al resto de la semana.
        fechas_ts = (fechas.values[None, :].astype("datetime64[D]")
                     + np.arange(T, dtype="timedelta64[D]")[:, None])   # (T, St)
        mes_ts = fechas_ts.astype("datetime64[M]").astype(int) % 12 + 1
        agregar("mes", np.broadcast_to(mes_ts.astype(np.float32)[None, :, :],
                                       (G, T, St)).copy())
        agregar("es_feriado", np.broadcast_to(
            _marca_feriado_por_dia(fechas_ts).astype(np.float32)[None, :, :],
            (G, T, St)).copy())

        rez1 = None
        for k in range(1, self.rezagos + 1):
            r = self._rezago(c, k)
            if k == 1:
                rez1 = r
            agregar(f"rezago_{k}", r)

        for w in self.ventanas:
            agregar(f"media_movil_{w}", self._media_movil(c, w))

        nivel_u = meta.get("nivel_espacial", "parroquia")
        agregar(f"densidad_historica_{nivel_u}", self._densidad_acumulada(c))

        # Sin coordenadas no existe una relacion de adyacencia defendible entre
        # parroquias: pertenecer al mismo canton no las vuelve vecinas (Puna es
        # una isla y Tenguel esta al extremo sur, y ambas son "del mismo canton"
        # que la cabecera). Se prescinde de caracteristicas espaciales de
        # vecindad en lugar de fabricar una proximidad que no se puede sostener.
        # La senal espacial que queda es la propia identidad de la unidad, via
        # su densidad historica.

        X = np.stack(columnas, axis=1)
        self.nombres = nombres
        print(f"  X: {X.shape[0]:,} filas x {X.shape[1]} variables, uint8 "
              f"({X.nbytes / 1e6:.1f} MB) | {S} semanas observadas + {F} futura(s)")
        return X, nombres

    def indices_semana_futura(self, G, T, S_observadas, k=0):
        """Filas de X de la k-esima semana futura, en orden (unidad, dia)."""
        s = S_observadas + k
        g, t = np.meshgrid(np.arange(G), np.arange(T), indexing="ij")
        return (g * T * self.S_total + t * self.S_total + s).reshape(-1)

    def fecha_semana_futura(self, k=0):
        return self.fechas_totales.iloc[-self.semanas_futuras + k]


# Feriados nacionales de Ecuador (Ley Organica de Feriados).
FERIADOS_FIJOS = {
    (1, 1):   "Anio Nuevo",
    (5, 1):   "Dia del Trabajo",
    (5, 24):  "Batalla del Pichincha",
    (8, 10):  "Primer Grito de Independencia",
    (10, 9):  "Independencia de Guayaquil",
    (11, 2):  "Dia de los Difuntos",
    (11, 3):  "Independencia de Cuenca",
    (12, 25): "Navidad",
}

# Moviles, derivados de la Pascua: Carnaval (lunes y martes, 48 y 47 dias antes)
# y Viernes Santo (2 dias antes). Se calculan en lugar de tabularse, para que el
# calendario sea correcto en cualquier anio del periodo.
#
# NO se modelan los traslados que la ley permite mover al lunes o al viernes
# siguiente, porque dependen de decretos anuales. El reporte lo declara asi.
_MOVILES_OFFSET = {-48: "Carnaval (lunes)", -47: "Carnaval (martes)",
                   -2: "Viernes Santo"}


def pascua(anio):
    """Domingo de Pascua, algoritmo gregoriano anonimo (Meeus/Jones/Butcher)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(year=anio, month=mes, day=dia + 1)


def feriados_del_anio(anio):
    """Diccionario {fecha: nombre} de los feriados nacionales de un anio."""
    f = {pd.Timestamp(year=anio, month=m, day=d): n
         for (m, d), n in FERIADOS_FIJOS.items()}
    p = pascua(anio)
    for off, nombre in _MOVILES_OFFSET.items():
        f[p + pd.Timedelta(days=off)] = nombre
    return f


def _marca_feriado_por_dia(fechas_ts):
    """
    Marca feriado en la FECHA EXACTA de cada (t, s), no en toda la semana.

    fechas_ts: arreglo datetime64[D] de forma (T, St).
    Cubre los feriados de fecha fija y los moviles derivados de la Pascua
    (Carnaval y Viernes Santo). No cubre los traslados por decreto anual.
    """
    f = pd.to_datetime(fechas_ts.ravel())
    todos = set()
    for anio in range(int(f.year.min()), int(f.year.max()) + 1):
        todos |= set(feriados_del_anio(anio).keys())
    marca = np.isin(f.normalize().values,
                    np.array(sorted(todos), dtype="datetime64[ns]")).astype(np.int8)
    return marca.reshape(fechas_ts.shape)
