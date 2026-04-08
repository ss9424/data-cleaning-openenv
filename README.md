# Data Cleaning Env (OpenEnv Hackathon)

This environment simulates a real-world data science task: Data Cleaning and Preprocessing.
Agents are provided with a messy, raw pandas DataFrame and must apply structured operations to perfectly match a hidden ground-truth dataset.

This repository is fully compliant with the Meta OpenEnv specification.

---

## Environment Overview

Real-world data is messy. Agents must act as automated data scientists, navigating null values, duplicate rows, incorrect currency encodings, and unscaled features.

The environment provides a Pydantic-typed Observation state summarizing the current DataFrame and evaluates the agent using a dense, partial-credit reward function that gives feedback at every step — not just at completion.

---

## Spaces

### Observation Space

The agent receives an Observation Pydantic model at each step, containing:

- sample_rows: List[dict] — First 5 rows of the dataset as a preview
- columns: List[str] — All current column names
- null_counts: dict[str, int] — Number of nulls per column
- dtype_map: dict[str, str] — Current dtype of each column
- column_stats: dict[str, dict] — Min, max, and unique count per column
- cleaning_history: List[str] — Log of all successful operations so far
- total_rows: int — Current row count
- total_columns: int — Current column count
- step: int — Steps taken in this episode
- message: str — Human-readable result of the last action

---

### Action Space

The agent submits an Action Pydantic model:

- operation: Literal[...] — One of the 11 supported cleaning operations
- column: Optional[str] — Target column (if required)
- value: Optional[str] — Extra parameter (fill value, dtype, unit symbol, expression, etc.)

Supported operations:
fill_na, drop_na, drop_column, rename_column, normalize, encode, fix_type, remove_units, drop_duplicates, feature_engineering, filter_rows

---

## Tasks

The environment includes three tiers of increasing difficulty.
Datasets are generated deterministically via generate_data.py (seed=42).

---

### Easy — tasks/task_easy.py

Max steps: 10
Dataset: 300 rows (+ 10 injected duplicates), 3 columns (id, name, age)

Required operations:
1. fill_na on age with median — fills 10% missing values
2. drop_duplicates — removes 10 exact duplicate rows

Expected baseline score: ~0.93–1.0

---

### Medium — tasks/task_medium.py

Max steps: 15
Dataset: 300 rows (+ 5 injected duplicates), 3 columns (product_id, category, price)

Required operations (in order):
1. encode on category — converts Auto/Home/Tech to 0/1/2
2. remove_units on price with value $ — strips currency symbol
3. fix_type on price with value float — casts string to numeric
4. drop_duplicates — removes 5 duplicate rows

Expected baseline score: ~0.88–0.95

---

### Hard — tasks/task_hard.py

Max steps: 20
Dataset: 300 rows, 4 columns (item_id, length_cm, price_inr, weight_norm)

Required operations:
1. feature_engineering on length_cm with value length_cm * 2.54 — converts inches to cm
2. feature_engineering on price_inr with value price_inr * 83.0 — converts USD to INR
3. normalize on weight_norm — applies min-max normalization

Expected baseline score: ~0.75–0.88

---

## Reward Function

The reward is calculated at every step and broken down into four components:

- shape_score (0.2): Penalizes row/column count mismatch vs. ground truth
- null_score (0.3): Rewards matching null patterns across all columns
- value_score (0.5): Rewards numeric closeness and exact string matches
- step_penalty (-0.05 max): Small penalty scaling with steps used

The episode ends when the score reaches ≥ 0.99 (perfect clean) or max_steps is exhausted.

---

## Setup & Usage

### Option 1: Local Python

1. Clone the repository and install dependencies:

git clone https://huggingface.co/spaces/your-username/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt

2. Set up your environment variables:

Create a .env file in the root directory:

HF_TOKEN=hf_your_token_here

# Optional Overrides:
# API_BASE_URL=https://router.huggingface.co/v1
# MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct

3. Generate the datasets:

python generate_data.py

4. Run the baseline evaluation:

python inference.py

---

### Option 2: Docker

1. Build the image:

docker build -t data-cleaning-env .

2. Run the evaluation:

docker run --env-file .env data-cleaning-env

(Note: The Dockerfile automatically runs generate_data.py at build time.)

---

## Project Structure
```
data-cleaning-env/
├── data/
│   ├── raw/                  # Messy input CSVs (auto-generated)
│   └── cleaned_ground_truth/ # Perfect target CSVs (auto-generated)
├── env/
│   ├── environment.py        # DataCleaningEnv — OpenEnv interface
│   ├── models.py             # Pydantic Observation, Action, Reward models
│   ├── reward.py             # RewardCalculator with partial-credit scoring
│   └── state_manager.py      # DataFrame operations and state summarization
├── tasks/
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
├── generate_data.py          # Deterministic dataset generator (seed=42)
├── inference.py              # Official OpenEnv inference/eval script
├── openenv.yaml              # OpenEnv metadata
├── requirements.txt
└── Dockerfile
```
---

## Tags

openenv · data-science · pandas · reinforcement-learning · llm-agent
