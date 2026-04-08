from env.environment import DataCleaningEnv

def get_task():
    """Returns the instantiated environment and the system prompt for the easy task."""
    env = DataCleaningEnv(
        csv_path="data/raw/easy.csv",
        ground_truth_path="data/cleaned_ground_truth/easy_clean.csv",
        max_steps=10
    )

    prompt = (
        "You are an automated data scientist. Your goal is to resolve basic data quality "
        "issues in the provided dataset to match the required ground-truth schema.\n\n"
        "Requirements:\n"
        # BUG FIX: Order corrected. drop_duplicates MUST come before fill_na.
        # The raw dataset has 10 duplicate rows appended at the end, which include
        # copies of some rows that have NaN ages. Those duplicates inflate the
        # non-null count from 270 to 280, shifting the median from 44.0 to 43.0.
        # Filling NaNs before deduplication uses the wrong median and will never
        # match the ground truth (which was generated using the 270-value median).
        "1. The dataset contains exact duplicate rows. Use the 'drop_duplicates' operation to remove them FIRST.\n"
        "2. The 'age' column contains missing values (NaNs). Use the 'fill_na' operation and pass 'median' as the 'value'.\n"
        "3. Ensure the 'id' and 'name' columns remain intact.\n\n"
        "Guidelines:\n"
        "- Pay close attention to the 'message' field in your observation. If an operation says it failed or was unrecognized, DO NOT repeat it.\n"
        "- Consult 'cleaning_history' and 'null_counts' in each observation to track your progress.\n"
        "- Do not repeat operations that have already been successfully logged in your history.\n"
        "- Your objective is to reach a perfect 1.0 reward score efficiently."
    )

    return env, prompt