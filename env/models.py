from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class Observation(BaseModel):
    """What the environment sends to the agent after every step."""
    sample_rows: List[dict] = Field(description="First 5 rows of the dataset as a preview.")
    columns: List[str] = Field(description="List of column names in the dataset")
    step: int = Field(description="How many steps have been taken in this episode so far")
    message: str = Field(description="Human-readable description of what just happened")
    cleaning_history: List[str] = Field(
        default_factory=list, 
        description="A log of all successful cleaning operations performed so far."
    )
    total_rows: int = Field(description="Total number of rows in the current dataset")
    total_columns: int = Field(description="Total number of columns in the current dataset")
    null_counts: dict[str, int] = Field(description="Number of null values per column")
    dtype_map: dict[str, str] = Field(description="Current data type of each column")
    column_stats: dict[str, dict] = Field(
        description="Statistical summary (min, max, unique values) for each column."
    )
    
    # --- REQUIRED OPENENV SERVER FIELDS ---
    reward: float = Field(default=0.0, description="Overall reward for this step (0.0 to 1.0)")
    done: bool = Field(default=False, description="True if the episode is over")
    error: Optional[str] = Field(default=None, description="Error message if the action failed")

# ... Keep your existing Action and Reward classes down here ...

class Action(BaseModel):
    """What the agent sends to the environment to perform a cleaning operation."""
    # Using Literal forces the LLM to pick exactly one of these strings.
    operation: Literal[
        'fill_na', 'drop_na', 'drop_column', 'rename_column', 'normalize', 
        'encode', 'fix_type', 'remove_units', 'drop_duplicates', 
        'feature_engineering', 'filter_rows'
    ] = Field(description="The exact cleaning operation to perform.")
    
    column: Optional[str] = Field(
        default=None,
        description="The column to operate on."
    )
    value: Optional[str] = Field(
        default=None,
        description="Extra parameter (fill value, dtype, unit symbol, expression, or condition)."
    )

class Reward(BaseModel):
    """What the environment returns to tell the agent how well it is doing."""
    score: float = Field(description="Overall reward for this step (0.0 to 1.0)")
    breakdown: dict[str, float] = Field(description="Score broken down by cleaning dimension")
    reason: str = Field(description="Human-readable explanation of the score")
    done: bool = Field(description="True if the episode is over")