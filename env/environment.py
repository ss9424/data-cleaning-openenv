from env.state_manager import StateManager
from env.reward import RewardCalculator
from env.models import Observation, Action

class DataCleaningEnv:
    def __init__(self, csv_path: str, ground_truth_path: str, max_steps: int):
        self.state_manager = StateManager(csv_path)
        self.reward_calculator = RewardCalculator(ground_truth_path)
        self.max_steps = max_steps
        self.current_step = 0
        self._current_state = None

    def reset(self) -> Observation:
        """Resets the environment and returns the initial observation."""
        self.state_manager.reset()
        self.current_step = 0
        state_dict = self.state_manager.get_state_summary()
        
        self._current_state = Observation(
            sample_rows=state_dict["sample_rows"],
            columns=state_dict["columns"],
            step=self.current_step,
            message="Environment reset. Initializing task...",
            cleaning_history=state_dict["cleaning_history"],
            total_rows=state_dict["total_rows"],
            total_columns=state_dict["total_columns"],
            null_counts=state_dict["null_counts"],
            dtype_map=state_dict["dtype_map"],
            column_stats=state_dict["column_stats"]
        )
        return self._current_state

    def state(self) -> Observation:
        """Returns the current state of the environment."""
        if self._current_state is None:
            return self.reset()
        return self._current_state

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Applies an action and returns (observation, reward, done, info)."""
        self.current_step += 1
        
        # Apply the action
        message = self.state_manager.apply_action(action.operation, action.column, action.value)
        
        # Get updated state
        state_dict = self.state_manager.get_state_summary()
        obs = Observation(
            sample_rows=state_dict["sample_rows"],
            columns=state_dict["columns"],
            step=self.current_step,
            message=message,
            cleaning_history=state_dict["cleaning_history"],
            total_rows=state_dict["total_rows"],
            total_columns=state_dict["total_columns"],
            null_counts=state_dict["null_counts"],
            dtype_map=state_dict["dtype_map"],
            column_stats=state_dict["column_stats"]
        )
        self._current_state = obs
        
        # Calculate reward
        reward_obj = self.reward_calculator.calculate(self.state_manager.df, self.current_step, self.max_steps)
        
        # Extract the required OpenEnv tuple elements
        reward = reward_obj.score
        done = reward_obj.done
        info = {
            "reason": reward_obj.reason,
            "breakdown": reward_obj.breakdown
        }
        
        return obs, reward, done, info