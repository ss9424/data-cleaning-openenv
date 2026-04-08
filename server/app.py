import os
import uvicorn
from openenv.core.env_server import create_fastapi_app
from env.models import Action, Observation
from tasks import task_easy, task_medium, task_hard

# 1. Create a "factory function" that OpenEnv can call to build fresh environments
def env_factory():
    # Read TASK_ID from environment, defaulting to 'easy'
    task_id = os.getenv("TASK_ID", "easy").lower()
    
    if task_id == "medium":
        env, _ = task_medium.get_task()
    elif task_id == "hard":
        env, _ = task_hard.get_task()
    else:
        env, _ = task_easy.get_task()
        
    return env

# 2. Pass the factory function (not the instance) to OpenEnv
app = create_fastapi_app(env_factory, Action, Observation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)