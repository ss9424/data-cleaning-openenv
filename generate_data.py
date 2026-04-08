import os
import pandas as pd
import numpy as np
import random

def create_directories():
    """Creates the necessary data folders if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/cleaned_ground_truth", exist_ok=True)

def generate_easy_task(n=300):
    """Matches task_easy.py: Missing values and duplicates."""
    np.random.seed(42)
    random.seed(42)

    ids = list(range(1, n + 1))
    names = [f"User_{i}" for i in ids]

    ages = np.random.randint(18, 70, size=n).astype(float)

    null_indices = random.sample(range(n), int(n * 0.1))

    mask = np.ones(n, dtype=bool)
    mask[null_indices] = False
    target_median = np.median(ages[mask])

    for idx in null_indices:
        ages[idx] = target_median

    # 1. Ground Truth (Perfect Data)
    df_clean = pd.DataFrame({"id": ids, "name": names, "age": ages})
    df_clean.to_csv("data/cleaned_ground_truth/easy_clean.csv", index=False)

    # 2. Raw Data (Messy)
    df_raw = df_clean.copy()
    for idx in null_indices:
        df_raw.loc[idx, "age"] = np.nan

    duplicates = df_raw.sample(10, random_state=42)
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

    df_raw.to_csv("data/raw/easy.csv", index=False)
    print(f"✅ Easy datasets generated ({len(df_raw)} rows messy -> {len(df_clean)} rows clean).")

def generate_medium_task(n=300):
    np.random.seed(42)
    product_ids = list(range(1001, 1001 + n))

    # BUG FIX 5: Save ground truth category as int8 to match what
    # pandas .astype('category').cat.codes produces, so dtype comparison in
    # reward.py gets full credit instead of falling back to the numeric fallback.
    categories_clean = np.random.choice([0, 1, 2], size=n)
    prices_clean = np.round(np.random.uniform(10.0, 999.99, size=n), 2)

    df_clean = pd.DataFrame({
        "product_id": product_ids,
        "category": categories_clean,
        "price": prices_clean
    })
    df_clean.to_csv("data/cleaned_ground_truth/medium_clean.csv", index=False)

    # Map must be alphabetical so Auto=0, Home=1, Tech=2
    # This aligns with the agent's .astype('category').cat.codes
    category_map = {0: "Auto", 1: "Home", 2: "Tech"}
    df_raw = pd.DataFrame({
        "product_id": product_ids,
        "category": [category_map[c] for c in categories_clean],
        "price": [f"${p:.2f}" for p in prices_clean]
    })

    # Add 5 duplicate rows
    duplicates = df_raw.sample(5, random_state=42)
    df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

    df_raw.to_csv("data/raw/medium.csv", index=False)
    print(f"✅ Updated Medium datasets with actual mess!")

def generate_hard_task(n=300):
    """Matches task_hard.py: Math via feature_engineering and normalization."""
    np.random.seed(42)

    item_ids = list(range(1, n + 1))

    length_inches = np.round(np.random.uniform(5.0, 50.0, size=n), 1)
    price_usd = np.round(np.random.uniform(5.0, 200.0, size=n), 2)
    weight_raw = np.random.randint(10, 500, size=n)

    length_cm = np.round(length_inches * 2.54, 3)
    price_inr = np.round(price_usd * 83.0, 2)

    w_min, w_max = weight_raw.min(), weight_raw.max()
    weight_norm = (weight_raw - w_min) / (w_max - w_min)

    # 1. Ground Truth (Perfect Data)
    df_clean = pd.DataFrame({
        "item_id": item_ids,
        "length_cm": length_cm,
        "price_inr": price_inr,
        "weight_norm": weight_norm
    })
    df_clean.to_csv("data/cleaned_ground_truth/hard_clean.csv", index=False)

    # 2. Raw Data (Messy)
    df_raw = pd.DataFrame({
        "item_id": item_ids,
        "length_cm": length_inches,
        "price_inr": price_usd,
        "weight_norm": weight_raw
    })
    df_raw.to_csv("data/raw/hard.csv", index=False)
    print(f"✅ Hard datasets generated ({len(df_raw)} rows).")

if __name__ == "__main__":
    print("Generating Benchmark Datasets...")
    create_directories()
    generate_easy_task(300)
    generate_medium_task(300)
    generate_hard_task(300)
    print("🎉 All 300-row datasets generated successfully! Check the 'data/' folder.")