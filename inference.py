import os
import sys
import time
from typing import List, Optional
from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import task_easy, task_medium, task_hard
from env.models import Action

# --- MANDATORY LOGGING FUNCTIONS FROM OPENENV ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
# ------------------------------------------------

def run_baseline():
    hf_token = os.getenv("HF_TOKEN")
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    benchmark_name = "data-cleaning-env"

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing. Please set it.")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    task_modules = [
        ("easy", task_easy),
        ("medium", task_medium),
        ("hard", task_hard)
    ]

    for difficulty, task_module in task_modules:
        env, base_prompt = task_module.get_task()
        obs = env.reset()
        
        rewards = []
        steps_taken = 0
        error = None
        
        # 1. Log the Start
        log_start(task=difficulty, env=benchmark_name, model=model_name)

        while True:
            time.sleep(2)  # Respect HF rate limits
            steps_taken += 1
            
            user_msg = (
                f"Instruction: {base_prompt}\n\n"
                f"Current State:\n{obs.model_dump_json(indent=2)}\n\n"
                "Respond strictly with a JSON object representing your next action. "
                "It must contain 'operation', and optionally 'column' and 'value'."
            )

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an automated data scientist. Output valid JSON only."},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                raw_response = response.choices[0].message.content
                action = Action.model_validate_json(raw_response)
                
                # Format action for logging
                action_str = f"{action.operation}({action.column},{action.value})"
                
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                
                # 2. Log the Step
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)

                if done:
                    # 3. Log the End (Success is true if final score is > 0.9)
                    success = reward >= 0.9 
                    log_end(success=success, steps=steps_taken, score=reward, rewards=rewards)
                    break
                    
            except ValidationError as ve:
                error = "ValidationError"
                log_step(step=steps_taken, action="parsing_failed", reward=0.0, done=True, error=error)
                log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
                break
            except Exception as e:
                error = str(e).replace(" ", "_")
                log_step(step=steps_taken, action="exception", reward=0.0, done=True, error=error)
                log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
                break

if __name__ == "__main__":
    run_baseline()