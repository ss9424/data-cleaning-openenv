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
        "1. The 'length_cm' column currently contains inches. Use the 'feature_engineering' operation and pass 'length_cm * 2.54' as the 'value'.\n"
        "2. The 'price_inr' column currently contains USD. Use the 'feature_engineering' operation and pass 'price_inr * 83.0' as the 'value'.\n"
        "3. The 'weight_norm' column contains raw values. Use the 'normalize' operation on this column.\n\n"
        "Guidelines:\n"
        "- Pay close attention to the 'message' field in your observation. If an operation says it failed or was unrecognized, DO NOT repeat it.\n"
        "- Monitor the 'cleaning_history' and 'column_stats' in your observations.\n"
        "- Verify if a column already meets the requirements before applying a transformation.\n"
        "- Sequence your actions logically to reach a perfect 1.0 reward score."
    )
    
    return env, prompt