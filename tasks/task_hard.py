from env.environment import DataCleaningEnv

def get_task():
    env = DataCleaningEnv(
        csv_path="data/raw/hard.csv",
        ground_truth_path="data/cleaned_ground_truth/hard_clean.csv",
        max_steps=20
    )

    prompt = (
        "You are an automated data scientist. Your goal is to transform the provided raw dataset "
        "to match the required schema and values exactly.\n\n"
        "Requirements:\n"
        "1. The 'length_cm' column currently contains inches. Use the 'feature_engineering' operation "
        "with column='length_cm' and value='length_cm * 2.54' to convert it.\n"
        "2. The 'price_inr' column currently contains USD. Use the 'feature_engineering' operation "
        "with column='price_inr' and value='price_inr * 83.0' to convert it.\n"
        "3. The 'weight_norm' column contains raw integer values. Use the 'normalize' operation on this column.\n\n"
        "Guidelines:\n"
        "- Pay close attention to the 'message' field in your observation. If an operation says it failed or was unrecognized, DO NOT repeat it.\n"
        # BUG FIX: Added this critical guideline. Without it, an agent that applies
        # feature_engineering twice will compound the multiplier (e.g. *2.54*2.54),
        # producing values that can never match the ground truth and wasting all remaining steps.
        "- Do not repeat operations that have already been successfully logged in your 'cleaning_history'. Each transformation should be applied exactly once.\n"
        "- Monitor the 'cleaning_history' and 'column_stats' in your observations to confirm each step before proceeding.\n"
        "- Verify if a column already meets the requirements before applying a transformation.\n"
        "- Sequence your actions logically to reach a perfect 1.0 reward score."
    )

    return env, prompt