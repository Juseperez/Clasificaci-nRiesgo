"""
Componente 1 - Procesamiento de los datos (UC1).

REPRESENTACION COMPUTACIONAL REVISADA
-------------------------------------
En los avances anteriores se asumio, a partir de la descripcion inicial de la
fuente, que los recursos mensuales del ECU 911 incluian coordenadas y hora del
incidente. Durante la implementacion se verifico directamente la estructura de
los archivos de distintos anios y se determino que la publicacion abierta
contiene:

    Fecha ; provincia ; Canton ; Cod_Parroquia ; Parroquia ; Servicio ; Subtipo

Es decir, ubicacion administrativa hasta parroquia y fecha a nivel de dia, pero
NO coordenadas ni hora. En consecuencia la representacion se ajusto a la maxima
granularidad verificable en la fuente:

    unidad espacial : parroquia (Cod_Parroquia)   [antes: celda de 1 km x 1 km]
    unidad temporal : dia de la semana, |T| = 7   [antes: franja de 3 h, |T| = 56]
    horizonte       : una semana                  [sin cambio]

El tensor de conteos c[g, t, s] conserva su forma, de modo que la construccion
de caracteristicas, la etiqueta por percentiles, la particion temporal y el
bosque aleatorio propio no requieren cambios.
"""
import glob
import os
import unicodedata

import numpy as np
import pandas as pd

T_DIAS = 7
DIAS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]


# ---------------------------------------------------------------------------
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().split())


# Nombres verificados en los archivos reales (2021, 2023, 2025). Se conservan
# sinonimos por si el portal cambia la cabecera en publicaciones futuras.
_ALIAS = {
    "fecha":     ["FECHA", "FECHA INCIDENTE", "FECHA_INCIDENTE"],
    "provincia": ["PROVINCIA", "NOMBRE PROVINCIA", "DPA PROVINCIA"],
    "canton":    ["CANTON", "NOMBRE CANTON", "DPA CANTON", "CIUDAD"],
    "cod_parroquia": ["COD PARROQUIA", "COD_PARROQUIA", "CODIGO PARROQUIA",
                      "DPA PARROQUIA", "COD PARR"],
    "parroquia": ["PARROQUIA", "NOMBRE PARROQUIA"],
    "servicio":  ["SERVICIO", "TIPO", "CATEGORIA", "TIPO INCIDENTE",
                  "TIPO DE INCIDENTE", "TIPO EMERGENCIA"],
    "subtipo":   ["SUBTIPO", "SUB TIPO", "DETALLE", "SUBTIPO INCIDENTE"],
}


def detectar_columnas(df):
    reales = {normalizar(c): c for c in df.columns}
    mapa = {}
    for canonico, alias in _ALIAS.items():
        for a in alias:
            if a in reales:
                mapa[canonico] = reales[a]
                break
        else:
            for norm, real in reales.items():
                if any(a in norm for a in alias):
                    mapa[canonico] = real
                    break
    return mapa


# ---------------------------------------------------------------------------
class Preprocesador:
    """
    Clase de diseno 'Preprocesador'. Produce el tensor c[g, t, s] con

        g = unidad espacial administrativa (parroquia o canton)
        t = dia de la semana (0 = lunes ... 6 = domingo)
        s = semana calendario

    `alcance` define el area de estudio y `nivel_espacial` la unidad:

        ("canton", "GUAYAQUIL")     -> 6 parroquias, una concentra ~99% de los
                                       eventos: la dimension espacial degenera
        ("provincia", "GUAYAS")     -> 53 parroquias, 6 con masa relevante
        (None, None)                -> nacional

    El identificador de unidad NO se usa nunca como variable numerica: solo como
    indice de fila del tensor, porque tratarlo como numero induciria un orden
    artificial entre parroquias.
    """

    def __init__(self, categoria="TRANSITO", alcance=("canton", "GUAYAQUIL"),
                 nivel_espacial="parroquia"):
        self.categoria = normalizar(categoria)
        self.nivel_alcance, self.valor_alcance = (alcance if alcance else (None, None))
        if self.valor_alcance:
            self.valor_alcance = normalizar(self.valor_alcance)
        if nivel_espacial not in ("parroquia", "canton"):
            raise ValueError("nivel_espacial debe ser 'parroquia' o 'canton'")
        self.nivel_espacial = nivel_espacial
        self.reporte = {}
        self.origen_semanas = None

    # -- 1. carga -----------------------------------------------------------
    def consolidar(self, rutas_csv):
        """
        Carga los CSV uno por uno y aplica el filtro de categoria/alcance
        ANTES de concatenarlos.

        Esto evita mantener en RAM simultaneamente todos los registros
        nacionales de los 54 meses.
        """
        if isinstance(rutas_csv, str):
            rutas_csv = sorted(glob.glob(os.path.join(rutas_csv, "*.csv")))

        if not rutas_csv:
            raise FileNotFoundError("No se encontraron CSV para consolidar.")

        partes = []

        total_crudos = 0
        total_categoria = 0
        total_area = 0
        archivos_validos = 0

        for i, ruta in enumerate(rutas_csv, 1):
            df = None

            for sep in [";", ",", "|", "\t"]:
                for enc in ["utf-8-sig", "utf-8", "latin-1"]:
                    try:
                        cand = pd.read_csv(
                            ruta,
                            sep=sep,
                            encoding=enc,
                            low_memory=False,
                            on_bad_lines="skip",
                        )

                        if cand.shape[1] >= 5:
                            df = cand
                            break

                    except Exception:
                        continue

                if df is not None:
                    break

            if df is None:
                print(
                    f"  [aviso] ilegible, se omite: "
                    f"{os.path.basename(ruta)}"
                )
                continue

            total_crudos += len(df)

            mapa = detectar_columnas(df)

            minimas = ["fecha", "canton", "servicio"]

            minimas.append(
                "cod_parroquia"
                if self.nivel_espacial == "parroquia"
                else "canton"
            )

            if self.nivel_alcance == "provincia":
                minimas.append("provincia")

            faltan = [k for k in set(minimas) if k not in mapa]

            if faltan:
                print(
                    f"  [aviso] {os.path.basename(ruta)} "
                    f"sin columnas {faltan} -> se omite"
                )
                del df
                continue

            # Canonicalizar solo las columnas reconocidas.
            sub = pd.DataFrame({
                k: df[v]
                for k, v in mapa.items()
            })

            # --------------------------------------------------------------
            # FILTRO TEMPRANO DE CATEGORIA
            # --------------------------------------------------------------
            servicio_norm = sub["servicio"].map(normalizar)

            raiz = self.categoria[:7]  # TRANSIT

            mask_categoria = servicio_norm.str.contains(
                raiz,
                na=False
            )

            sub = sub.loc[mask_categoria].copy()

            total_categoria += len(sub)

            # --------------------------------------------------------------
            # FILTRO TEMPRANO DE AREA
            # --------------------------------------------------------------
            if self.nivel_alcance:
                col_area = (
                    "provincia"
                    if self.nivel_alcance == "provincia"
                    else "canton"
                )

                area_norm = sub[col_area].map(normalizar)

                sub = sub.loc[
                    area_norm == self.valor_alcance
                ].copy()

            total_area += len(sub)

            # Ya no necesitamos las ~300k filas nacionales de ese mes.
            del df

            if len(sub) == 0:
                print(
                    f"  [{i:02d}/{len(rutas_csv):02d}] "
                    f"{os.path.basename(ruta)} -> 0 registros del alcance"
                )
                continue

            sub["archivo"] = os.path.basename(ruta)

            partes.append(sub)
            archivos_validos += 1

            print(
                f"  [{i:02d}/{len(rutas_csv):02d}] "
                f"{os.path.basename(ruta)} -> "
                f"{len(sub):,} registros relevantes"
            )

        if not partes:
            raise ValueError(
                "Ningun archivo aporto registros para el alcance. "
                "Corre 00_inspeccionar_csv.py y revisa _ALIAS."
            )

        # Aqui ya concatenamos SOLO Transito + Guayaquil,
        # no millones de registros nacionales.
        out = pd.concat(partes, ignore_index=True)

        self.reporte.update({
            "crudos": total_crudos,
            "tras_categoria": total_categoria,
            "tras_area": total_area,
            "archivos_validos": archivos_validos,
        })

        # Indica a filtrarAlcance que categoria y area ya fueron filtradas.
        self._prefiltrado_alcance = True

        print(
            f"\n  consolidados {len(out):,} registros relevantes "
            f"de {archivos_validos} archivos"
        )

        print(
            f"  filas nacionales leidas : {total_crudos:,}"
        )

        print(
            f"  tras categoria           : {total_categoria:,}"
        )

        print(
            f"  tras alcance             : {total_area:,}"
        )

        return out

    # -- 2. filtro de alcance y depuracion ----------------------------------
    def filtrarAlcance(self, df):
        """
        Completa la depuracion despues de consolidar.

        Si consolidar() ya aplico categoria y alcance, no vuelve a cargar
        ni filtrar los registros nacionales; aqui se validan fecha y unidad.
        """
        d = df.copy()

        if getattr(self, "_prefiltrado_alcance", False):
            n0 = int(self.reporte.get("crudos", len(d)))
            n_cat = int(self.reporte.get("tras_categoria", len(d)))
            n_area = int(self.reporte.get("tras_area", len(d)))

        else:
            n0 = len(d)

            d["_servicio"] = d["servicio"].map(normalizar)

            raiz = self.categoria[:7]

            d = d[
                d["_servicio"].str.contains(
                    raiz,
                    na=False
                )
            ]

            n_cat = len(d)

            n_area = n_cat

            if self.nivel_alcance:
                col = (
                    "provincia"
                    if self.nivel_alcance == "provincia"
                    else "canton"
                )

                if col not in d.columns:
                    raise ValueError(
                        f"El dataset no trae la columna '{col}'."
                    )

                d["_area"] = d[col].map(normalizar)

                d = d[
                    d["_area"] == self.valor_alcance
                ]

                n_area = len(d)

        # --------------------------------------------------------------
        # VALIDACION DE FECHA
        # --------------------------------------------------------------
        fecha = pd.to_datetime(
            d["fecha"],
            errors="coerce",
            dayfirst=True
        )

        mask_fecha = fecha.notna()

        d = d.loc[mask_fecha].copy()

        d["ts"] = fecha.loc[mask_fecha]

        n_fecha = len(d)

        # --------------------------------------------------------------
        # VALIDACION DE UNIDAD ESPACIAL
        # --------------------------------------------------------------
        col_u = (
            "cod_parroquia"
            if self.nivel_espacial == "parroquia"
            else "canton"
        )

        d = d[
            d[col_u].notna()
        ].copy()

        n_unidad = len(d)

        self.reporte.update({
            "descartados_categoria": n0 - n_cat,
            "descartados_area": n_cat - n_area,
            "descartados_fecha": n_area - n_fecha,
            "descartados_sin_unidad": n_fecha - n_unidad,
            "validos": n_unidad,
        })

        print(
            f"  filtro alcance: {n0:,} -> "
            f"{n_unidad:,} validos"
        )

        print(
            f"    fuera de categoria '{self.categoria}' : "
            f"{n0 - n_cat:,}"
        )

        if self.nivel_alcance:
            print(
                f"    fuera de {self.nivel_alcance} "
                f"{self.valor_alcance:<12}: "
                f"{n_cat - n_area:,}"
            )

        print(
            f"    fecha invalida                  : "
            f"{n_area - n_fecha:,}"
        )

        print(
            f"    sin unidad espacial             : "
            f"{n_fecha - n_unidad:,}"
        )

        if n_unidad == 0:
            raise ValueError(
                "El filtro dejo cero registros: "
                "revisa categoria y alcance."
            )

        return d

    # -- 3. discretizacion espacial y temporal ------------------------------
    def asignarUnidadDia(self, df):
        """
        Asigna a cada registro su unidad espacial administrativa y su dia de la
        semana, y lo situa en la semana calendario correspondiente. La semana va
        de lunes a domingo y el origen se ancla en el primer lunes del periodo.
        """
        d = df.copy()
        col_u = "cod_parroquia" if self.nivel_espacial == "parroquia" else "canton"
        d["_unidad"] = (d[col_u].astype(str).str.strip() if self.nivel_espacial == "canton"
                        else d[col_u].astype(float).astype("Int64").astype(str))

        d["dia_semana"] = d["ts"].dt.dayofweek.astype(np.int16)     # 0..6

        origen = d["ts"].min().normalize()
        origen = origen - pd.Timedelta(days=int(origen.dayofweek))
        d["semana"] = ((d["ts"].dt.normalize() - origen).dt.days // 7).astype(np.int32)
        self.origen_semanas = origen
        return d

    # -- 4. agregacion en el tensor c[g, t, s] ------------------------------
    def agregarConteos(self, df):
        unidades = np.sort(df["_unidad"].unique())
        idx_u = {u: i for i, u in enumerate(unidades)}
        G, T = len(unidades), T_DIAS
        S = int(df["semana"].max()) + 1

        g = df["_unidad"].map(idx_u).values.astype(np.int64)
        t = df["dia_semana"].values.astype(np.int64)
        s = df["semana"].values.astype(np.int64)
        plano = np.bincount(g * (T * S) + t * S + s, minlength=G * T * S)
        c = plano.reshape(G, T, S).astype(np.int32)

        # Nombre legible y canton de cada unidad, para la interfaz
        if self.nivel_espacial == "parroquia":
            ref = (df.groupby("_unidad")
                     .agg(nombre=("parroquia", "first"), canton=("canton", "first"))
                     .reindex(unidades))
        else:
            ref = pd.DataFrame({"nombre": unidades, "canton": unidades}, index=unidades)
        nombres = ref["nombre"].fillna(ref.index.to_series()).astype(str).values
        cantones = ref["canton"].fillna("").map(normalizar).values

        meta = {
            "unidades": unidades,
            "nombre_unidad": nombres,
            "canton_unidad": cantones,
            "nivel_espacial": self.nivel_espacial,
            "alcance": (self.nivel_alcance, self.valor_alcance),
            "origen_semanas": self.origen_semanas,
            "fecha_semana": [self.origen_semanas + pd.Timedelta(weeks=int(i))
                             for i in range(S)],
            "dias": DIAS,
            "G": G, "T": T, "S": S,
            "semanas_con_datos": np.nonzero(c.sum(axis=(0, 1)) > 0)[0],
        }
        print(f"  tensor c[g,t,s]: G={G} unidades ({self.nivel_espacial}), "
              f"T={T} dias de semana, S={S} semanas ({c.nbytes / 1e6:.2f} MB)")
        print(f"  eventos en el tensor: {int(c.sum()):,}  |  "
              f"media por unidad-dia-semana: {c.mean():.3f}")
        share = c.sum(axis=(1, 2)) / max(c.sum(), 1)
        print(f"  unidad dominante concentra {share.max()*100:.1f}%  |  "
              f"unidades con >=1% de eventos: {int((share >= 0.01).sum())}")
        # Salvaguarda: meses no contiguos producen semanas enteras vacias que
        # envenenan los rezagos, las medias moviles y los percentiles de la
        # etiqueta. Con la descarga completa del portal no deberia ocurrir.
        vacias = int((c.sum(axis=(0, 1)) == 0).sum())
        if vacias:
            print(f"  [AVISO IMPORTANTE] {vacias} de {S} semanas no tienen ningun evento. "
                  "Si descargaste meses sueltos, el tensor tiene huecos: los rezagos y "
                  "los umbrales saldran mal. Descarga los meses contiguos, o filtra "
                  "las semanas con datos antes de entrenar.")
            meta_vacias = np.nonzero(c.sum(axis=(0, 1)) > 0)[0]
            print(f"  semanas con datos: {len(meta_vacias)} "
                  f"({100*len(meta_vacias)/S:.1f}% del rango)")

        if share.max() > 0.9:
            print(f"  [nota] la unidad dominante concentra {share.max()*100:.1f}% de los "
                  "eventos: la resolucion espacial que ofrece el dataset abierto es "
                  "limitada. Hay que declararlo como limitacion en el reporte y "
                  "reportar las metricas por unidad, no solo agregadas.")
        return c, meta

    # -- pipeline completo ---------------------------------------------------
    def ejecutar(self, rutas_csv):
        df = self.consolidar(rutas_csv)
        df = self.filtrarAlcance(df)
        df = self.asignarUnidadDia(df)
        return self.agregarConteos(df)
