import pandas as pd
import numpy as np
from env.models import Reward

class RewardCalculator:
    """
    Calculates the agent's score by robustly comparing the current dataset 
    against a perfectly clean 'ground truth' dataset.
    """
    def __init__(self, ground_truth_path: str):
        self.ground_truth = pd.read_csv(ground_truth_path)

    def calculate(self, current_df: pd.DataFrame, current_step: int, max_steps: int) -> Reward:
        target_rows, target_cols = self.ground_truth.shape
        current_rows, current_cols = current_df.shape
        
        # 1. Shape Score (Max 0.2)
        row_penalty = min(abs(target_rows - current_rows) / max(target_rows, 1), 1.0)
        col_penalty = min(abs(target_cols - current_cols) / max(target_cols, 1), 1.0)
        shape_score = 0.2 * (1.0 - ((row_penalty + col_penalty) / 2))

        # Find overlapping columns to evaluate
        common_cols = list(set(self.ground_truth.columns).intersection(set(current_df.columns)))
        
        null_score = 0.0
        value_score = 0.0

        if common_cols:
            col_null_scores = []
            col_val_scores = []
            
            # FIX 2: Determine how many rows we can safely compare side-by-side
            # This allows partial credit for value cleaning even if duplicates aren't dropped yet.
            compare_len = min(target_rows, current_rows)

            for col in common_cols:
                # --- 2. Null Score (Max 0.3) ---
                target_nulls = self.ground_truth[col].isna()
                current_nulls = current_df[col].isna()
                
                if target_rows == current_rows:
                    # FIX 3: Use .to_numpy() to avoid pandas index misalignment
                    null_match = (target_nulls.to_numpy() == current_nulls.to_numpy()).mean()
                else:
                    diff = abs(target_nulls.sum() - current_nulls.sum())
                    null_match = max(0.0, 1.0 - (diff / max(target_rows, 1)))
                
                col_null_scores.append(null_match)

                # --- 3. Value/Type Score (Max 0.5) ---
                val_match = 0.0
                
                # FIX 4: Looser dtype checking. Treat all numerics as structurally similar.
                is_tgt_num = pd.api.types.is_numeric_dtype(self.ground_truth[col])
                is_cur_num = pd.api.types.is_numeric_dtype(current_df[col])
                
                if self.ground_truth[col].dtype == current_df[col].dtype:
                    val_match += 0.4
                elif is_tgt_num and is_cur_num:
                    val_match += 0.4 
                    
                # FIX 2 (cont): Compare values up to the matching length 
                if compare_len > 0:
                    try:
                        tgt_vals = self.ground_truth[col].iloc[:compare_len].to_numpy()
                        cur_vals = current_df[col].iloc[:compare_len].to_numpy()

                        if is_tgt_num and is_cur_num:
                            # FIX 1: Removed .fillna(0). Cast to float so np.isclose handles NaNs natively.
                            matches = np.isclose(
                                cur_vals.astype(float), 
                                tgt_vals.astype(float), 
                                equal_nan=True
                            )
                            val_match += 0.6 * matches.mean()
                        else:
                            # Exact string/object matching
                            matches = (cur_vals == tgt_vals)
                            val_match += 0.6 * matches.mean()
                    except Exception:
                        pass # Fails safely if types are wildly incompatible
                        
                col_val_scores.append(val_match)

            null_score = 0.3 * (sum(col_null_scores) / len(common_cols))
            value_score = 0.5 * (sum(col_val_scores) / len(common_cols))

        # 4. Step Penalty
        step_penalty = (current_step / max_steps) * 0.05

        raw_score = shape_score + null_score + value_score - step_penalty
        total_score = max(0.0, min(1.0, round(raw_score, 3)))
        
        is_perfect = (shape_score + null_score + value_score) >= 0.99
        done = is_perfect or current_step >= max_steps

        if is_perfect:
            reason = f"Perfectly cleaned in {current_step} steps!"
        elif done:
            reason = f"Episode ended. Final score: {total_score}."
        else:
            reason = f"Step {current_step}/{max_steps}. Keep iterating to match target schema and values."

        return Reward(
            score=total_score,
            breakdown={
                "shape_score": round(shape_score, 3),
                "null_score": round(null_score, 3),
                "value_score": round(value_score, 3),
                "step_penalty": round(-step_penalty, 3)
            },
            reason=reason,
            done=done
        )