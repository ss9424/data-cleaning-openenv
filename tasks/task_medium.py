from env.environment import DataCleaningEnv

def get_task():
    """Returns the instantiated environment and the system prompt for the medium task."""
    env = DataCleaningEnv(
        csv_path="data/raw/medium.csv",
        ground_truth_path="data/cleaned_ground_truth/medium_clean.csv",
        max_steps=15
    )
    
    prompt = (
        "You are an automated data scientist. You must perform structural transformations "
        "on this dataset to prepare it for a production-ready schema.\n\n"
        "Requirements:\n"
        "1. The 'category' column contains string labels. Use the 'encode' operation to convert them to integers.\n"
        "2. The 'price' column contains currency symbols. Remove them by using the 'remove_units' operation and passing '$' as the 'value'.\n"
        "3. After removing symbols, use the 'fix_type' operation to cast the 'price' column to 'float' (pass 'float' as the 'value').\n"
        "4. The dataset contains exact duplicate rows. Use the 'drop_duplicates' operation to remove them.\n\n"
        "Guidelines:\n"
        "- Pay close attention to the 'message' field in your observation. If an operation says it failed or was unrecognized, DO NOT repeat it.\n"
        "- Check 'dtype_map' and 'column_stats' to verify the current state of the 'price' and 'category' columns.\n"
        "- Sequence your actions carefully: remove units before attempting to cast the data type."
    )
    
    return env, prompt