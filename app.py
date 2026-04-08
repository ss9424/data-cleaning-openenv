import uvicorn
from openenv.core.env_server import create_fastapi_app
from env.models import Action, Observation
from tasks import task_easy

# 1. Create a "factory function" that OpenEnv can call to build fresh environments
def env_factory():
    env, _ = task_easy.get_task()
    return env

# 2. Pass the factory function (not the instance) to OpenEnv
app = create_fastapi_app(env_factory, Action, Observation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)