# Notes — 01 Honest Model

## 1. Mistakes I made
- `uv init --package <name>` created a nested subfolder; using `uv init --package --name <name> .` properly targets the current directory.
- `import pyyaml` failed with `ModuleNotFoundError` because the PyPI package is `pyyaml`, but the Python import name is `yaml`.
- Overriding Pydantic's `model_validate` method manually caused a `NameError` on `AppConfig` and bypassed Pydantic's built-in recursive parsing; Pydantic handles nested sub-models automatically.
- Kaggle API returned `403 Forbidden` on download because competition terms/rules must be explicitly accepted on the Kaggle website in the browser before the API is authorized.

## 2. Learnings
- **`pyproject.toml` vs `uv.lock`**: `pyproject.toml` defines high-level direct dependencies, version constraints, and tool configurations; `uv.lock` freezes the entire transitive dependency graph (exact wheels, versions, and hashes) ensuring deterministic bit-for-bit builds across environments.
- **Fail-fast configuration**: Loading settings through typed Pydantic models catches missing keys, typos, and invalid types immediately at startup, preventing runtime crashes hours into model training.
- **Idempotent ingestion**: Scripts that fetch data should be idempotent—checking if raw files or archives already exist to avoid redundant multi-hundred-megabyte downloads.
- **Pre-commit quality gates**: Automating `ruff` and file hygiene via `.pre-commit-config.yaml` catches formatting errors, syntax issues, and large file leaks before code enters Git history.

## 3. Things to remember
### Sources of Nondeterminism in ML

A single `random_state` on an estimator (model) does **not** fix:

- Unseeded train/test splits
- Unordered column iterations (`set()` / `dict.keys()`)
- Multi-threaded floating-point reduction order — `(a + b) + c != a + (b + c)`

The sections below explain each of these in detail.

---

#### 1. Randomness in Data 🎲

`random_state` is a seed used to make random operations reproducible.

**Without a seed:**

| Run | Result |
|-----|--------|
| Run 1 | Different train/test split |
| Run 2 | Different train/test split |
| Run 3 | Different train/test split |

**With `random_state=42`:** the same random operation produces the same result on every run.

```python
train_test_split(X, y, random_state=42)
```

> **Important:** The model's `random_state` does not control how the data is split.

Two separate concerns:

| Seed | Controls |
|------|----------|
| Data split seed | Which rows are used? |
| Model seed | How does the model learn? |

---

#### 2. Randomness in the Whole Pipeline

An ML pipeline has many steps, not just the final model.

```text
Raw Data
    ↓
Clean data
    ↓
Fill missing values
    ↓
Reduce features
    ↓
Balance classes
    ↓
Search for best parameters
    ↓
Train model
    ↓
Prediction
```

Different steps can have their own randomness:

| Step | Source of randomness |
|------|----------------------|
| `KMeans` | Random centroid initialization |
| `PCA(svd_solver="randomized")` | Randomized computation |
| `TruncatedSVD` | Can use randomness |
| `IterativeImputer` | May involve randomness depending on configuration |
| `SMOTE` | Randomly creates synthetic samples |
| `RandomizedSearchCV` | Randomly selects parameter combinations |
| Early stopping | May randomly create an internal validation set |

Setting:

```python
model = Model(random_state=42)
```

does **not** automatically make the entire pipeline reproducible.

##### Main rule

> Don't ask *"Is my model reproducible?"*
>
> Ask *"Is my entire pipeline reproducible?"*

---

#### 3. The Machine Underneath

Even after controlling randomness, the computer can introduce tiny numerical differences.

`n_jobs=-1` generally means:

> Use all available CPU resources for the parallelizable work.

Multiple threads/workers may perform calculations in parallel. Floating-point numbers are approximations, so:

```text
(a + b) + c
```

can sometimes produce a slightly different result from:

```text
a + (b + c)
```

Different execution orders can therefore create tiny numerical differences.

- Usually these differences are harmless.
- Sometimes a tiny difference can change an algorithm's decision:

```text
Tiny numerical difference
        ↓
Different decision
        ↓
Different result
```


### **Debugging order for variance**: When validation metrics drift across runs on the same code:
  1. Check train/test row split index alignment (`assert (idx1 == idx2).all()`).
  2. Check sha256 checksum of preprocessed feature matrices.
  3. Verify estimator seeds and single-threaded execution (`n_jobs=1`).

### **Data & Environment Isolation**: Never commit raw/processed data, model binaries, or `.venv` to Git. Keep `data/` and `.venv/` strictly in `.gitignore`.

## 4. Project recipe
*(To be populated as feature engineering, validation split strategy, and leak detection are built).*
