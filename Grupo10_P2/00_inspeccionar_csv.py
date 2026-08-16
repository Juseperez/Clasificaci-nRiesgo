"""
Paso 0: inspeccionar los CSV reales del ECU 911 antes de escribir cualquier pipeline.

Uso:
    python 00_inspeccionar_csv.py datos/

Responde las cuatro preguntas que quedaron abiertas en la Tarea #4:
  1. Cuales son los nombres exactos de las columnas y cambian entre anios.
  2. Que calidad tienen las coordenadas (nulos, ceros, fuera de rango).
  3. Como viene escrita la categoria de transito.
  4. Si el campo canton permite filtrar Guayaquil y con que ortografia.

No modifica nada: solo lee y reporta.
"""
import sys
import glob
import os
import unicodedata

import pandas as pd


def normalizar(texto):
    """Mayusculas, sin tildes, sin espacios sobrantes."""
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().split())


def leer_cabecera(ruta, n=5000):
    """Intenta varias combinaciones de separador y codificacion."""
    for sep in [";", ",", "|", "\t"]:
        for enc in ["utf-8", "latin-1", "utf-8-sig"]:
            try:
                df = pd.read_csv(ruta, sep=sep, encoding=enc, nrows=n,
                                 low_memory=False, on_bad_lines="skip")
                if df.shape[1] >= 4:
                    return df, sep, enc
            except Exception:
                continue
    return None, None, None


def inspeccionar(ruta):
    print("=" * 78)
    print(os.path.basename(ruta), f"({os.path.getsize(ruta) / 1e6:.1f} MB)")
    print("=" * 78)

    df, sep, enc = leer_cabecera(ruta)
    if df is None:
        print("  !! No se pudo leer con ningun separador/codificacion probados.")
        return

    print(f"  separador={sep!r}  codificacion={enc}  filas leidas={len(df)}  columnas={df.shape[1]}")
    print("\n  COLUMNAS:")
    for c in df.columns:
        nulos = df[c].isna().mean() * 100
        muestra = df[c].dropna().head(3).tolist()
        print(f"    - {c!r:38s} nulos={nulos:5.1f}%  ej: {muestra}")

    # --- categorias / tipos ---
    for c in df.columns:
        cn = normalizar(c)
        if any(k in cn for k in ["TIPO", "CATEGORIA", "INCIDENTE", "SERVICIO"]):
            vals = df[c].astype(str).map(normalizar).value_counts().head(12)
            print(f"\n  VALORES DE {c!r}:")
            for v, k in vals.items():
                marca = "  <-- TRANSITO?" if "TRANS" in v else ""
                print(f"    {k:>8d}  {v}{marca}")

    # --- canton ---
    for c in df.columns:
        cn = normalizar(c)
        if "CANTON" in cn or "CIUDAD" in cn:
            vals = df[c].astype(str).map(normalizar).value_counts().head(10)
            print(f"\n  VALORES DE {c!r}:")
            for v, k in vals.items():
                marca = "  <-- GUAYAQUIL" if "GUAYAQUIL" in v else ""
                print(f"    {k:>8d}  {v}{marca}")

    # --- coordenadas ---
    for c in df.columns:
        cn = normalizar(c)
        if any(k in cn for k in ["LAT", "LON", "LNG", "X", "Y", "COORD"]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0:
                continue
            print(f"\n  COORDENADA {c!r}: min={s.min():.5f} max={s.max():.5f} "
                  f"nulos={s.isna().mean()*100:.1f}% ceros={(s == 0).mean()*100:.1f}%")

    # --- fecha / hora ---
    for c in df.columns:
        cn = normalizar(c)
        if "FECHA" in cn or "HORA" in cn:
            print(f"\n  TIEMPO {c!r}: ej {df[c].dropna().head(4).tolist()}")


if __name__ == "__main__":
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "datos"
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
    if not rutas:
        print(f"No hay CSV en {carpeta!r}.")
        print("Descargalos de https://datosabiertos.gob.ec/dataset/base-de-emergencias")
        print("Recomendado empezar con 3 de anios distintos: 202108, 202301, 202501.")
        sys.exit(1)
    for r in rutas:
        inspeccionar(r)
        print()
    print(f"Inspeccionados {len(rutas)} archivos.")
