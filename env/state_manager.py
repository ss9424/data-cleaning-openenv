import pandas as pd
import numpy as np

class StateManager:
    """
    Handles the internal pandas DataFrame state, applying cleaning operations,
    and returning summaries of the data for the agent to observe.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.initial_df = self.df.copy()
        self.applied_actions = []

    def reset(self):
        """Restores the dataframe to its original loaded state."""
        self.df = self.initial_df.copy()
        self.applied_actions = []

    def get_state_summary(self) -> dict:
        """Extracts a smart, token-efficient summary of the dataframe."""
        clean_df = self.df.replace({np.nan: None})

        stats = {}
        for col in self.df.columns:
            col_stats = {"unique_count": self.df[col].nunique()}
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_stats["min"] = float(self.df[col].min()) if pd.notnull(self.df[col].min()) else None
                col_stats["max"] = float(self.df[col].max()) if pd.notnull(self.df[col].max()) else None
            stats[col] = col_stats

        return {
            "sample_rows": clean_df.head(5).to_dict(orient="records"),
            "columns": self.df.columns.tolist(),
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "null_counts": self.df.isna().sum().to_dict(),
            "dtype_map": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "column_stats": stats,
            "cleaning_history": self.applied_actions
        }

    def apply_action(self, operation: str, column: str = None, value: str = None) -> str:
        try:
            msg = ""

            # --- Global operations that don't strictly require a target column ---
            if operation == 'drop_duplicates':
                before = len(self.df)
                self.df = self.df.drop_duplicates().reset_index(drop=True)
                msg = f"Dropped {before - len(self.df)} duplicate rows."

            elif operation == 'drop_na':
                before = len(self.df)
                if column and column in self.df.columns:
                    self.df = self.df.dropna(subset=[column]).reset_index(drop=True)
                    msg = f"Dropped {before - len(self.df)} rows with NaNs in '{column}'."
                else:
                    self.df = self.df.dropna().reset_index(drop=True)
                    msg = f"Dropped {before - len(self.df)} rows with NaNs."

            elif operation == 'filter_rows':
                # BUG FIX 1: Guard against None value before calling df.query()
                if not value:
                    return "Error: 'filter_rows' requires a query string in 'value'."
                before = len(self.df)
                self.df = self.df.query(value).reset_index(drop=True)
                msg = f"Filtered rows. Dropped {before - len(self.df)} rows."

            elif operation == 'feature_engineering':
                # BUG FIX 1: Guard against None value/column before calling df.eval()
                if not value or not column:
                    return "Error: 'feature_engineering' requires both 'column' and 'value' (expression)."
                self.df[column] = self.df.eval(value)
                msg = f"Calculated {value} for {column}."

            # --- Column-specific operations ---
            else:
                if column not in self.df.columns:
                    return f"Error: Column '{column}' not found."

                if operation == 'remove_units':
                    self.df[column] = self.df[column].astype(str).str.replace(value, '', regex=False).str.strip()
                    msg = f"Removed unit '{value}' from '{column}'."

                elif operation == 'fix_type':
                    if value == 'float':
                        self.df[column] = pd.to_numeric(
                            self.df[column].astype(str).str.replace('[^0-9.-]', '', regex=True),
                            errors='coerce'
                        )
                        msg = f"Cast '{column}' to float."
                    elif value == 'int':
                        # BUG FIX 2: Use nullable Int64 instead of filling NaNs with 0
                        self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
                        msg = f"Cast '{column}' to int."
                    else:
                        # BUG FIX 3: Return error for unrecognized type instead of silent no-op
                        return f"Error: Unrecognized type '{value}' for fix_type. Use 'int' or 'float'."

                elif operation == 'encode':
                    self.df[column] = self.df[column].astype('category').cat.codes
                    msg = f"Label-encoded '{column}'."

                elif operation == 'normalize':
                    mi, ma = self.df[column].min(), self.df[column].max()
                    if ma > mi:
                        self.df[column] = (self.df[column] - mi) / (ma - mi)
                        msg = f"Normalized {column}."
                    else:
                        return "Error: Min and Max are equal, cannot normalize."

                elif operation == 'fill_na':
                    if value == 'median':
                        fill_val = self.df[column].median()
                    elif value == 'mean':
                        fill_val = self.df[column].mean()
                    elif value == 'mode':
                        fill_val = self.df[column].mode()[0]
                    else:
                        fill_val = value
                    self.df[column] = self.df[column].fillna(fill_val)
                    msg = f"Filled missing values in '{column}' with {value}."

                elif operation == 'drop_column':
                    self.df = self.df.drop(columns=[column])
                    msg = f"Dropped column '{column}'."

                elif operation == 'rename_column':
                    self.df = self.df.rename(columns={column: value})
                    msg = f"Renamed column '{column}' to '{value}'."

            if msg:
                self.applied_actions.append(f"{operation} on {column}" if column else operation)
                return msg

            return "Operation completed with no changes or unrecognized operation."

        except Exception as e:
            return f"Operation failed: {str(e)}"