# Repository Architecture

The repo is structured as a `src/` Python package with decoupled loaders, models, evaluation, and validation modules. Experiments wire these together in `experiments/` scripts.

```
src/
  config/         antenna positions
  loaders/        CSV → (X, y, df)
  models/         (params, X_train, y_train) → fitted model
  evaluation/     error arrays → scalar metrics
  utils/          trilateration solver
  validation/     generic k-fold CV
experiments/
  run_*.py        orchestration scripts
data/
  feb17/          ds_advanced_17Feb26.csv, ds_basic_17Feb26.csv
  march11/        ds_elemen.csv, ds_Tx.csv, ds_Tx_Rx.csv
```

---

## Data: CSV → (X, y)

All loaders live in `src/loaders/` and expose a single `.load(path: str)` method returning `(X, y, df)`.

| Loader | `X` | `y` | Use when |
|---|---|---|---|
| `AdvancedDsLoader` | RSSI features, NaN→−130 dBm | positions `(n, 3)` | direct position prediction |
| `AdvancedDsWithDistancesLoader` | RSSI features | distances to antennas `(n, n_ant)` | distance-first pipeline |
| `ElemenDsLoader` | receiver RSSI + one-hot(Tx, power) | positions `(n, 3)` | element dataset |

**Shapes:**
- `X`: `(n_samples, n_features)` float32
- `y` positions: `(n_samples, 3)` — (x, y, z) in meters
- `y` distances: `(n_samples, n_antennas)` — meters

**`AdvancedDsWithDistancesLoader`** additionally populates:
- `.antenna_names: list[str]` — antenna IDs in column order
- `.distance_cols: list[str]` — `D_*` column names used as targets

```python
loader = AdvancedDsWithDistancesLoader()
X, y_dist, df = loader.load("data/feb17/ds_advanced_17Feb26.csv")
antenna_positions = np.array([ANTENNA_POSITIONS[a] for a in loader.antenna_names])
```

---

## Models

Models live in `src/models/` as plain functions — no classes. Each takes `(params_dict, X_train, y_train)` and returns a fitted sklearn estimator.

| Function | Returns | `y_train` shape | Use for |
|---|---|---|---|
| `train_extratrees(params, X, y)` | `MultiOutputRegressor` | `(n, 3)` | direct xyz prediction |
| `train_distance_regressor(params, X, y)` | `ExtraTreesRegressor` | `(n, n_antennas)` | distance prediction |

Both return objects with `.predict(X_test)` producing the same shape as `y_train`.

**Why two wrappers:** `MultiOutputRegressor` trains one tree per output (needed for heterogeneous xyz); `ExtraTreesRegressor` natively handles multiple homogeneous outputs (distances).

**Adding a new model:** create `src/models/train_<name>.py` with signature:
```python
def train_<name>(params: dict, X_train: np.ndarray, y_train: np.ndarray) -> <sklearn estimator>:
```

---

## Trilateration (`src/utils/trilaterate_numerical.py`)

Converts predicted distances → position. Entry point:

```python
trilaterate(distances: np.ndarray, antenna_positions: np.ndarray,
            max_iter: int = 20, tol: float = 1e-8) -> np.ndarray  # shape (3,)
```

- **Input:** `distances` `(n_antennas,)`, `antenna_positions` `(n_antennas, 3)`
- **Output:** estimated position `(3,)` in meters
- **Algorithm:** LSQ anchor linearization (initial estimate) → Gauss-Newton refinement
- Handles near-coplanar antenna arrays (ceiling-mounted) by estimating z from distance residuals

Used in experiment scripts by mapping over each test sample:
```python
y_xyz_pred = np.array([trilaterate(row, antenna_positions) for row in y_dist_pred])
```

---

## Evaluation (`src/evaluation/metrics.py`)

All functions take an error array `errs: np.ndarray` (Euclidean or absolute) and return a scalar:

```python
mae_3d(errs)                        # mean of errs
rmse_3d(errs)                       # sqrt(mean(errs²))
threshold_accuracy(errs, threshold) # % of samples ≤ threshold (0–100)
```

Standard metrics dict returned per fold:
```python
{"rmse": float, "acc_1m": float, "acc_2m": float, "acc_3m": float}
# distance pipeline also includes:
{"dist_mae": float, ...}
```

---

## Cross-Validation (`src/validation/cross_validation.py`)

Generic k-fold framework. All model logic is injected as callables:

```python
result = cross_validate(
    X, y,
    train_fn   = lambda X_tr, y_tr: ...,         # returns model
    predict_fn = lambda model, X_te: ...,         # returns y_pred
    eval_fn    = lambda y_te, y_pred, idx: {...}, # returns metrics dict
    aggregate_fn = lambda folds: {...},           # returns {metric: {mean, std}}
    k=5, random_state=42
)
# result["folds"]   → List[Dict[str, float]]  — per-fold metrics
# result["summary"] → Dict[str, Dict]          — {metric: {mean, std}}
```

The `idx` argument in `eval_fn` is the test fold indices — use it to access external arrays (e.g., `y_xyz[idx]` in the distance pipeline).

**Adding a new experiment:** copy an `experiments/run_*.py`, swap the loader, model function, and eval logic. The CV framework is unchanged.

---

## Config (`src/config/antennas.py`)

```python
ANTENNA_NAMES: list[str]               # 8 antenna IDs
ANTENNA_POSITIONS: dict[str, tuple]    # {antenna_id: (x, y, z)}
```

Used to build the `antenna_positions` array needed by `trilaterate()`.

---

## Two Pipeline Patterns

**Pattern A — Direct position:**
```
CSV → AdvancedDsLoader → (X, y_xyz) → train_extratrees → predict → Euclidean error
```

**Pattern B — Distance + trilateration:**
```
CSV → AdvancedDsWithDistancesLoader → (X, y_dist, df)
    → train_distance_regressor → predict distances
    → trilaterate() per sample → y_xyz_pred → Euclidean error
```

Pattern B requires keeping `y_xyz` from `df` in a closure for the `eval_fn`.

---

## Current Baselines (5-fold CV, `ds_advanced_17Feb26.csv`)

| Pipeline | RMSE (m) | Acc@1m (%) | Acc@2m (%) |
|---|---|---|---|
| ExtraTrees direct | 1.26 | 49.3 | 92.9 |
| ExtraTrees + trilateration | 1.31 | 45.7 | 92.0 |
