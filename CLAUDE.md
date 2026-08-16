# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ESPOL AI course project (Grupo #10, Paralelo P2): classifies traffic-emergency
risk level (Bajo/Medio/Alto) per `(parroquia, día de la semana)` one week
ahead, using open monthly CSVs from ECU 911 (Servicio Integrado de Seguridad),
filtered to Guayaquil / "Tránsito y Movilidad". All real work lives under
`Grupo10_P2/`; the repo root only has a synced copy of the README (kept
identical to `Grupo10_P2/README.md` — update both together).

The core deliverable is a **Random Forest implemented from scratch in NumPy**
(`src/bosque.py`) — CART induction, weighted Gini, bootstrap, per-node random
feature subsampling, class weighting, multiclass leaf probabilities, forest
probability averaging, impurity-based feature importance. `scikit-learn` is
used only as an experimental comparison baseline, never in the actual
prediction path.

Full methodology, data caveats, and current results are documented in detail
in [Grupo10_P2/README.md](Grupo10_P2/README.md) — read it before changing
labeling, partitioning, or feature logic; it explains *why* choices were made,
not just what they are.

## Commands

All commands run from `Grupo10_P2/` (not repo root), with the project's
venv/dependencies from `requirements.txt` active.

```bash
# install
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
python -m pip install -r requirements.txt

# inspect raw monthly CSVs (columns, categories, row counts)
python 00_inspeccionar_csv.py datos/

# run the full test suite
python -m pytest tests/ -q

# run a single test file / test
python -m pytest tests/test_gini.py -q
python -m pytest tests/test_pipeline.py::test_nombre_especifico -q

# run the full pipeline (load -> preprocess -> features -> train -> eval -> artifacts)
# open proyecto.ipynb and run all cells with the project's Python kernel

# launch the UI (reads salidas/ artifacts only, does not retrain)
streamlit run app.py
```

There is no build/lint step configured; correctness is verified via pytest
and by inspecting the notebook's final metrics against `salidas/metricas.json`.

## Architecture

Pipeline, strictly in this order to avoid temporal leakage (see
`src/caracteristicas.py` module docstring):

```
monthly CSVs (datos/)
  -> Preprocesador.consolidar()        [src/preprocesamiento.py]
     per-file filter (Tránsito y Movilidad, cantón Guayaquil) BEFORE concat,
     to avoid holding national-scale data in memory
  -> tensor c[g, t, s]                  counts per (parroquia, día_semana, semana);
     trailing weeks with partial data get dropped (`descartar_semanas_incompletas`,
     default True) — otherwise missing days count as zero and fake a "Bajo" label
  -> ParticionadorTemporal.dividir()    [src/caracteristicas.py]
     chronological train/val/test split on semana index, no shuffling
  -> mascara_elegibilidad()             computed from TRAIN weeks only —
     a parroquia with zero incidents in training never enters the model
     (shown in the UI as "Sin datos suficientes", not "Bajo")
  -> ConstructorEtiquetas                percentile thresholds (P60/P90) fit
     on TRAIN only, applied to all splits; falls back through an escalation
     ladder (global -> positivos -> por_dia -> discreto) when percentiles
     collapse on a zero-inflated count distribution — see class docstring
  -> feature matrix X                    11 features, discretized into uint8
     quantile bins (fit on train only) to cut memory ~8x vs float64
  -> BosqueAleatorioPropio.entrenar()    [src/bosque.py]
  -> EvaluadorMetricas + Baselines       [src/metricas.py]
  -> MatrizRiesgo                        [src/riesgo.py]  next-week predictions,
     only for elegible unidad-día pairs; ineligible ones stay NaN, never
     silently predicted-then-masked
  -> salidas/ artifacts (metricas.json, modelo.joblib, matriz_riesgo.parquet,
     unidades.json, *.png)
  -> app.py (Streamlit)                  reads salidas/ only, never retrains
```

Key invariants to preserve when touching this code:

- **No leakage**: any threshold, bin edge, or eligibility mask must be fit
  using only training-window data (`semanas_train`), then applied to
  val/test — never refit per split.
- **No fake trailing weeks**: `Preprocesador` drops a partial week at the end
  of the data (`descartar_semanas_incompletas=True` by default) instead of
  letting missing days silently count as zero incidents — turn it off only
  in unit tests using tiny synthetic data, not on real pipelines.
- **Unit IDs are never numeric features**: `Cod_Parroquia` is only ever used
  as a tensor row index, never fed to the model as a number (would imply a
  false ordering between parroquias).
- **Ineligible units don't get predictions**: `predecir_matriz_semana` in
  `src/riesgo.py` only calls the model on eligible rows; it does not predict
  everything and mask afterward, since that would misrepresent what the
  classifier actually evaluated.
- `ArbolDecisionPropio` builds tree nodes recursively, then compiles them
  into flat NumPy arrays (`_compilar`) so `predecirProba` can descend all
  rows vectorized level-by-level instead of node-by-node.

### Module map (`Grupo10_P2/src/`)

- `preprocesamiento.py` — CSV loading/column normalization, category/cantón
  filtering, tensor `c[g,t,s]` construction.
- `caracteristicas.py` — temporal split, eligibility mask, label thresholds,
  feature matrix `X` construction (lags, rolling means, historical density,
  calendar features incl. Ecuadorian holidays/Carnaval/Viernes Santo).
- `bosque.py` — the from-scratch Random Forest (`NodoArbol`,
  `ArbolDecisionPropio`, `BosqueAleatorioPropio`, `giniPonderado`).
- `metricas.py` — `EvaluadorMetricas` (F1 macro/per-class, AUC-ROC OvR,
  Brier, calibration curve) and `Baselines` (majority class, persistence,
  historical mode) — all three baselines are mandatory comparison points,
  not optional.
- `riesgo.py` — `MatrizRiesgo` builds the next-week risk grid consumed by
  the UI, respecting the eligibility mask.

### Tests (`Grupo10_P2/tests/`)

`test_gini.py` (Gini criterion vs hand-solved cases, plus a seed-determinism
check), `test_pipeline.py` (preprocessing/labeling/split pipeline, plus
leakage-injection tests and a `ParticionadorTemporal` test added in the
Aug-16 audit), `test_representacion.py` (parroquia × día representation),
`test_metricas.py` (`EvaluadorMetricas` verified against the published
confusion matrix). Run as a module from the project root when invoked
directly, e.g. `python -m tests.test_gini`.
