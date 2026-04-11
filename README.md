---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Data Cleaning Env — OpenEnv Hackathon

This environment simulates a real-world data science task: **Data Cleaning and Preprocessing**.  
Agents are provided with a messy, raw pandas DataFrame and must apply structured operations to perfectly match a hidden ground-truth dataset.

Fully compliant with the Meta OpenEnv specification. Passes `openenv validate` with all four deployment modes.

---

## Environment Overview

Real-world data is messy. Agents must act as automated data scientists, navigating null values, duplicate rows, incorrect currency encodings, and unscaled features.

The environment provides a Pydantic-typed `Observation` state summarising the current DataFrame and evaluates the agent using a **dense, partial-credit reward function** that gives feedback at every step — not just at completion.

---

## Observation Space

The agent receives an `Observation` Pydantic model at each step:

| Field | Type | Description |
|---|---|---|
| `sample_rows` | `List[dict]` | First 5 rows of the dataset as a preview |
| `columns` | `List[str]` | All current column names |
| `null_counts` | `dict[str, int]` | Number of nulls per column |
| `dtype_map` | `dict[str, str]` | Current dtype of each column |
| `column_stats` | `dict[str, dict]` | Min, max, and unique count per column |
| `cleaning_history` | `List[str]` | Log of all successful operations so far |
| `total_rows` | `int` | Current row count |
| `total_columns` | `int` | Current column count |
| `step` | `int` | Steps taken in this episode |
| `message` | `str` | Human-readable result of the last action |
| `reward` | `float` | Score after this step (0.0–1.0) |
| `done` | `bool` | Whether the episode has ended |
| `error` | `Optional[str]` | Error message if the last action failed |

---

## Action Space

The agent submits an `Action` Pydantic model:

| Field | Type | Description |
|---|---|---|
| `operation` | `Literal[...]` | One of 11 supported cleaning operations |
| `column` | `Optional[str]` | Target column (if required) |
| `value` | `Optional[str]` | Extra parameter (fill value, dtype, unit symbol, expression, etc.) |

**Supported operations:**

| Operation | Description |
|---|---|
| `fill_na` | Fill NaN values in a column (`value`: `median`, `mean`, `mode`, or a literal) |
| `drop_na` | Drop rows with NaNs (optionally scoped to a column) |
| `drop_duplicates` | Remove exact duplicate rows |
| `drop_column` | Drop a column entirely |
| `rename_column` | Rename a column (`value`: new name) |
| `fix_type` | Cast a column to a dtype (`value`: `float` or `int`) |
| `remove_units` | Strip a unit string from a column (`value`: e.g. `$`) |
| `encode` | Label-encode a categorical column (alphabetical order) |
| `normalize` | Apply min-max normalization to a numeric column |
| `feature_engineering` | Evaluate a pandas expression into a column (`value`: expression) |
| `filter_rows` | Drop rows matching a query string (`value`: pandas query) |

---

## Tasks

Datasets are generated deterministically via `generate_data.py` (seed=42).

### Easy — `tasks/task_easy.py`

**Max steps:** 10  
**Dataset:** 300 rows + 10 injected duplicates, 3 columns (`id`, `name`, `age`)

Required operations (in order):
1. `drop_duplicates` — removes 10 exact duplicate rows **first**
2. `fill_na` on `age` with `median` — fills ~10% missing values

> Order matters: duplicates include rows with NaN ages that shift the median.

**Expected baseline score:** ~0.93–1.0

---

### Medium — `tasks/task_medium.py`

**Max steps:** 15  
**Dataset:** 300 rows + 5 injected duplicates, 3 columns (`product_id`, `category`, `price`)

Required operations (in order):
1. `encode` on `category` — converts `Auto/Home/Tech` → `0/1/2` (alphabetical)
2. `remove_units` on `price` with value `$` — strips currency symbol
3. `fix_type` on `price` with value `float` — casts string to numeric
4. `drop_duplicates` — removes 5 duplicate rows

**Expected baseline score:** ~0.88–0.95

---

### Hard — `tasks/task_hard.py`

**Max steps:** 20  
**Dataset:** 300 rows, 4 columns (`item_id`, `length_cm`, `price_inr`, `weight_norm`)

Required operations:
1. `feature_engineering` on `length_cm` with value `length_cm * 2.54` — converts inches → cm
2. `feature_engineering` on `price_inr` with value `price_inr * 83.0` — converts USD → INR
3. `normalize` on `weight_norm` — applies min-max normalization

> Each transformation must be applied exactly once — the prompt warns the agent not to repeat operations.

**Expected baseline score:** ~0.75–0.88

---

## Reward Function

Calculated at every step with four components:

| Component | Weight | Description |
|---|---|---|
| `shape_score` | 0.20 | Penalises row/column count mismatch vs. ground truth |
| `null_score` | 0.30 | Rewards matching null patterns across all columns |
| `value_score` | 0.50 | Rewards numeric closeness and exact string matches |
| `step_penalty` | −0.05 max | Small penalty scaling with steps used |

Episode ends when score ≥ 0.99 (perfect clean) or `max_steps` is exhausted.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)

### Option 1: Local Python

```bash
git clone https://huggingface.co/spaces/SS9424/data-cleaning-openenv
cd data-cleaning-openenv
pip install -r requirements.txt
```

Create a `.env` file:

```
HF_TOKEN=hf_your_token_here
# Optional overrides:
# API_BASE_URL=https://router.huggingface.co/v1
# MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
```

Generate the datasets, then run the baseline:

```bash
python generate_data.py
python inference.py
```

### Option 2: Docker

```bash
docker build -t data-cleaning-env .
docker run --env-file .env data-cleaning-env python inference.py
```

The Dockerfile automatically runs `generate_data.py` at build time.

### Option 3: Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Or via `uv`:

```bash
uv run --project . server
```

---

## Validation

```bash
pip install openenv-core
openenv validate
```

Expected output:
```
[OK] data-cleaning-openenv: Ready for multi-mode deployment

Supported deployment modes:
  [YES] docker
  [YES] openenv_serve
  [YES] uv_run
  [YES] python_module
```

---

## Project Structure

```
data-cleaning-openenv/
├── server/
│   ├── __init__.py
│   └── app.py                # FastAPI server with main() — OpenEnv entry point
├── env/
│   ├── __init__.py
│   ├── environment.py        # DataCleaningEnv — OpenEnv interface
│   ├── models.py             # Pydantic Observation, Action, Reward models
│   ├── reward.py             # RewardCalculator with partial-credit scoring
│   └── state_manager.py      # DataFrame operations and state summarisation
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
├── data/
│   ├── raw/                  # Messy input CSVs (auto-generated)
│   └── cleaned_ground_truth/ # Perfect target CSVs (auto-generated)
├── generate_data.py          # Deterministic dataset generator (seed=42)
├── inference.py              # Baseline inference/eval script (root — required)
├── openenv.yaml              # OpenEnv metadata
├── pyproject.toml            # Package config with [project.scripts] server entry
├── uv.lock                   # Locked dependency manifest
├── requirements.txt          # pip-compatible dependency list
└── Dockerfile
```

---

## Baseline Scores

Scores produced by `Qwen/Qwen2.5-Coder-32B-Instruct` via HF Inference Router:

| Task | Score | Steps |
|---|---|---|
| Easy | ~0.95 | 2–3 |
| Medium | ~0.90 | 4–5 |
| Hard | ~0.80 | 3–4 |

---

## Tags

`openenv` · `data-science` · `pandas` · `reinforcement-learning` · `llm-agent`