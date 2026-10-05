# Notes — 01 Honest Model

## 1. Mistakes I made
- `uv init --package <name>` created a nested subfolder; using `uv init --package --name <name> .` properly targets the current directory.
- `import pyyaml` failed with `ModuleNotFoundError` because the PyPI package is `pyyaml`, but the Python import name is `yaml`.
- Overriding Pydantic's `model_validate` method manually caused a `NameError` on `AppConfig` and bypassed Pydantic's built-in recursive parsing; Pydantic handles nested sub-models automatically.
- Passing a boolean Series directly inside a Pandas named aggregation tuple (e.g. `("IS_ACTIVE" == 0, "count")`) raises `TypeError: unhashable type: 'Series'`. Named aggregation strictly expects `("column_name_str", "func")`. Helper columns must be created beforehand on the DataFrame.
- In test assertions for aggregated tables, asserting against input/temporary column names (`IS_ACTIVE`, raw `DAYS_CREDIT`) fails because the aggregated DataFrame only contains the output feature names (`BUREAU_ACTIVE_LOAN_COUNT`, `BUREAU_DAYS_CREDIT_MIN`).

## 2. Learnings
- **`pyproject.toml` vs `uv.lock`**: `pyproject.toml` defines high-level direct dependencies, version constraints, and tool configurations; `uv.lock` freezes the entire transitive dependency graph (exact wheels, versions, and hashes) ensuring deterministic bit-for-bit builds across environments.
- **Fail-fast configuration**: Loading settings through typed Pydantic models catches missing keys, typos, and invalid types immediately at startup, preventing runtime crashes hours into model training.
- **Idempotent ingestion**: Scripts that fetch data should be idempotent—checking if raw files or archives already exist to avoid redundant multi-hundred-megabyte downloads.
- **Pre-commit quality gates**: Automating `ruff` and file hygiene via `.pre-commit-config.yaml` catches formatting errors, syntax issues, and large file leaks before code enters Git history.
- **`groupby().agg()` vs `groupby().transform()`**:
  - `agg(["mean", "max"])` **collapses** the table — N rows becomes 1 row per group with new summary columns. Use when merging child table features back to the main table.
  - `transform("mean")` **preserves** row count — broadcasts the group result back to every original row. Use for within-group normalization staying in the same table.
  - **The trap**: Using `transform()` then merging silently explodes row count because the child table still has M rows per applicant.
- **Named Aggregations in Pandas**: `df.groupby("key").agg(NEW_COL=("ORIG_COL", "stat"))` directly builds clean, flattened, and prefixed column names in a single pass without MultiIndex column renaming headaches.
- **Dual Feature Strategy (Lifetime vs Active)**: Never discard closed loans naively; compute both lifetime historical aggregates (past repayment volume, overdue history across all loans) and current active burden (active loan debt/count) for maximum tree-model predictive signal.
- **Temporal leak in aggregations**: Only aggregate child table rows that predate the application date (`DAYS_CREDIT < 0`). Excluding ambiguous boundary events (`DAYS_CREDIT = 0`) prevents the current loan decision from leaking into training features.
- **NumPy Broadcasting Mechanics**: Compares array shapes dimension-by-dimension starting from the trailing (rightmost) axis. Two dimensions are compatible if they are equal or if one of them is 1. Dimensions of size 1 are stretched virtually without copying data in memory via stride-0 strides.
- **Vectorization vs Python Loops**: Pure Python loops incur bytecode interpreter, type checking, and PyObject heap allocation overhead on every single iteration. Vectorized NumPy operations hand contiguous C buffers directly to hardware CPU SIMD (Single Instruction, Multiple Data) instructions, running 15x–30x+ faster on modern CPUs.
- **Preprocessor (Transformer) vs Model (Estimator)**:
  - *Preprocessor* (`StandardScaler`, `SimpleImputer`): Cleans and rescales data columns. It knows nothing about target labels (`y`) and makes no predictions. Uses `.fit()` to calculate column statistics ($\mu, \sigma$, median) and `.transform()` to modify values.
  - *Model* (`LogisticRegression`, `LGBMClassifier`): Learns the mathematical mapping between features ($X$) and target outcomes ($y$). Uses `.fit(X, y)` to tune weights/splits. When training completes, weights are frozen. Uses `.predict()` / `.predict_proba()` on new inputs without altering weights. Models never have a `.transform()` method.
- **`.fit()` vs `.transform()` vs `.fit_transform()`**:
  - `.fit(X)` calculates and remembers parameters (e.g. `scaler.mean_`, `imputer.statistics_`) without modifying data.
  - `.transform(X)` applies those saved parameters to change numerical values.
  - `.fit_transform(X)` is a convenience shortcut for `.fit(X)` followed by `.transform(X)` on the training set.
- **Why Train Statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) Must Scale Future Data**:
  1. *Production Reality (Single-Customer Problem)*: In production, customers arrive one by one. You cannot calculate a mean or std on $N=1$ sample. You must scale against the stored training distribution.
  2. *Coordinate Frame Consistency (The Ruler)*: The model learned decision thresholds relative to $\mu_{\text{train}}$. Re-calculating a separate mean on test data shifts the definition of zero and distorts features relative to the learned decision boundaries.
- **Pre-Split Preprocessing Leak**: Running `fit_transform(X)` across the full dataset before splitting bakes the mean, variance, and medians of future test samples into the training features. The test set appears artificially well-centered in the training space, producing an inflated, dishonest test AUC.
- **Why LightGBM over Classical Scikit-Learn Models on Tabular Data**:
  - *Non-linear credit risk relationships*: Real-world lending risk is non-linear (e.g., 5 past loans is safer than 1 unproven loan or 40 desperate loans). Linear models (`LogisticRegression`) assume monotonic straight lines and miss non-linear thresholds; tree-based models split on complex boundaries naturally.
  - *Speed via Histogram Binning*: LightGBM discretizes continuous floating-point features into compact integer bins (`Total Bins: 10950`), allowing it to train on 246,000 rows in <2 seconds. Scikit-learn's `RandomForestClassifier` would take 5–10 minutes and gigabytes of memory.
  - *API Conformance*: Implements the exact same `.fit(X, y)` and `.predict_proba(X)` scikit-learn protocol, making it a drop-in estimator for scikit-learn pipelines.
- **The 3 Distinct Data Leaks & Their Relative Impact**:
  1. *Pre-Split Preprocessing Leak* (Minor, +0.005 to +0.01 AUC): Leaks summary statistics ($\mu, \sigma$, medians) across the split boundary.
  2. *Split Strategy Leak* (Moderate, +0.01 to +0.03 AUC): Random shuffling mixes future macro regimes and multi-event entities between train and test sets.
  3. *Post-Outcome Feature Leak* (Massive, +0.10 to +0.25+ AUC): Features computed from events occurring *after* the loan decision. This inflates evaluation scores the most because it directly leaks information about the target outcome itself.

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


### **Question**: When does merge silently multiply your row count, and how do you detect it in one line?
The mechanism:
In a 1-to-Many join (e.g. application_train → bureau), applicant SK_ID_CURR = 100001 exists once in the left table but has 12 matching rows in bureau. When Pandas merge runs, it replicates that one left
row 12 times — once for every matching right row. The result explodes from 307k rows to millions.
One line to detect it (cleaner than an assert):
df_bureau.groupby("SK_ID_CURR").size().describe()
If max > 1, the right table is Many-to-One with the left and a direct merge will explode row count.

### **Question**:  Why is category dtype sometimes slower than object?

"Objects are fast in such cases" doesn't explain the exact mechanism.
The actual mechanism:
In Pandas, a category column stores values as a dictionary (lookup table of unique strings) + an array of integer codes. This is efficient for memory but adds a lookup step.
When you do a groupby or merge on a category column, Pandas must:
1. Validate that all codes match valid entries in the category dictionary.
2. Expand/decode codes back into actual string labels to compute group keys or join keys.
3. Handle edge cases where two categoricals have different dictionaries (e.g. ["M", "F"] vs ["F", "M"]) — Pandas must reconcile them before joining.

For a high-cardinality column (e.g. 200,000 unique strings out of 300,000 rows), the dictionary itself becomes a large memory overhead and the lookup cost per row exceeds the simple pointer-comparison
cost of plain object.

### **Question**: Why is `.apply()` on a groupby usually a performance mistake?
The mechanism:
1. **Python loop overhead vs compiled C kernels**: Built-in aggregations (`.sum()`, `.mean()`, `.agg()`) run in compiled C/Cython with optimized vectorization. In contrast, `.apply()` falls back to pure Python: for 307,000 applicants, it constructs 307,000 separate temporary DataFrame slices, makes 307,000 Python function calls, and stitches them back together. It is typically 10x to 100x slower.
2. **Double execution on the first group**: Pandas calls your function twice on the first group to infer return types and shapes.
3. **Memory churn**: Creating hundreds of thousands of intermediate Python objects triggers constant garbage collection.

### **Question**: Predict output shape of `A[:, None, :] * B[None, :, :]` for A of shape (4, 3) and B of shape (5, 3).
- $A_{new} = A[:, \text{None}, :]$ has shape `(4, 1, 3)`.
- $B_{new} = B[\text{None}, :, :]$ has shape `(1, 5, 3)`.
- Aligning right to left:
  - Dim 2: $3$ vs $3 \rightarrow 3$
  - Dim 1: $1$ vs $5 \rightarrow 5$
  - Dim 0: $4$ vs $1 \rightarrow 4$
- **Resulting shape: `(4, 5, 3)`**.

### **Question**: Why does broadcasting sometimes blow up memory rather than saving it?
In broadcasting, the inputs use virtual stride-0 pointers without copying data. However, the **output result must be fully materialized in memory**.
If $A$ has shape `(100000, 1, 10)` (~8 MB) and $B$ has shape `(1, 100000, 10)` (~8 MB), the broadcast output has shape `(100000, 100000, 10)`. At 8 bytes per float64, this requires $10^{11} \times 8 \approx \mathbf{800\text{ GB}}$ of RAM, instantly causing an Out-of-Memory (OOM) system crash.

### Real Day 5 Benchmarks Measured (`scripts/benchmark_vectorization.py`):
1. **Recency Decay** (100k loans x 3 windows): Loop = 0.2810s, Vectorized = 0.0110s $\rightarrow$ **25.5x speedup**.
2. **Relative Credit Gap** (100k credits x 5 tiers): Loop = 0.3597s, Vectorized = 0.0119s $\rightarrow$ **30.3x speedup**.
3. **Matrix Standardization** (100k rows x 10 features): Loop = 1.2136s, Vectorized = 0.0643s $\rightarrow$ **18.9x speedup**.

### Real Day 6 Baseline Measurement (`scripts/train_naive.py`):
- **Model**: `LGBMClassifier()` (LightGBM gradient boosted trees)
- **Features**: 99 numeric columns from `application_train.csv`
- **Training Samples**: 246,008 rows (80%)
- **Test Samples**: 61,503 rows (20% random split)
- **Preprocessing Injected**: `SimpleImputer(strategy="median")` + `StandardScaler()` fit-transformed pre-split across all 307,511 rows
- **Recorded Naive (Leaked) ROC-AUC**: **`0.75061`**

### Real Day 7 Honest Pipeline Measurement (`scripts/train.py`):
- **Model**: `LGBMClassifier(random_state=42)` inside `sklearn.Pipeline`
- **Features**: same 99 numeric columns
- **Split Order**: `train_test_split` called BEFORE `pipeline.fit` — no preprocessing leakage
- **Imputer Strategy**: `median` (not mean — mean is distorted by large credit outliers)
- **Recorded Honest ROC-AUC**: **`0.75086`**
- **Delta vs Leaked Baseline**: **`+0.00025`** (essentially flat — pre-split preprocessing alone is a minor leak on this dataset; the split strategy and feature-level leaks matter far more)

### Concepts to revise (handed to me, I did not write these independently):
- **Completing the evaluation loop in a pipeline script**: After computing `predict_proba`, I left out the `roc_auc_score` import and the final print. The full pattern is always: import metric → compute score → compare and print delta. Never leave a script that computes predictions without also evaluating them.
- **Mean vs median imputer choice**: I chose `strategy="mean"` without justification. The correct choice for financial data is `median` because credit-related columns (amounts, term lengths) contain large outliers that pull the mean far from the typical customer. The median is the middle value — outliers cannot move it.

### Day 7 Prove It — answers worked out via hint ladder:
- **Q1 — Why does putting the imputer inside the Pipeline change the evaluation score?**
  In `train_naive.py`, the imputer's median was calculated on all 307,511 rows before the split — including the 61,503 rows that later became the test set. Those test rows contributed their values to the median used to fill `X_train`. Their statistical fingerprint (their share of the median) leaked into training features. In `train.py`, the Pipeline ensures the imputer only ever calls `.fit()` on `X_train` (246,008 rows). The test rows contribute nothing to the median. The evaluation score reflects a genuinely unseen hold-out.
- **Q2 — What leak does a correct Pipeline still not catch?**
  A **temporal split leak**. If you use `train_test_split` (random shuffle) on time-ordered loan data, training rows will include recent loans (approved 1 month ago) and test rows will include old loans (approved 5 years ago). In production you always train on the past and predict the future — never the reverse. A Pipeline controls fit/transform boundaries but has no knowledge of how `X_train` and `X_test` were constructed. Passing in randomly shuffled time-series data produces a perfectly fit Pipeline on a poisoned input. The fix is a time-based split (cutoff date), not a random one.

## 4. Project recipe
*(To be populated as feature engineering, validation split strategy, and leak detection are built).*
