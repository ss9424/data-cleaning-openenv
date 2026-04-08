import uvicorn
from openenv.core.env_server import create_fastapi_app
from env.models import Action, Observation
from tasks import task_easy

# 1. Grab the easy task to act as the live environment
env, _ = task_easy.get_task()

# 2. Wrap it in OpenEnv's official FastAPI server
app = create_fastapi_app(env, Action, Observation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)