"""
FastAPI application for the Data Cleaning Environment.

Exposes the environment over HTTP endpoints compatible with OpenEnv clients.

Endpoints:
    POST /reset  - Reset the environment
    POST /step   - Execute a cleaning action
    GET  /state  - Get current environment state
    GET  /schema - Get action/observation schemas
    GET  /health - Health check
    GET  /metadata - Environment metadata

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    python -m server.app
"""

import os
from openenv.core.env_server import create_fastapi_app

try:
    from env.models import Action, Observation
    from tasks import task_easy, task_medium, task_hard
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.models import Action, Observation
    from tasks import task_easy, task_medium, task_hard

TASK_MAP = {
    "easy": task_easy,
    "medium": task_medium,
    "hard": task_hard,
}

def env_factory(task_id: str = "easy"):
    """Factory that builds a fresh environment for the given task ID."""
    module = TASK_MAP.get(task_id, task_easy)
    env, _ = module.get_task()
    return env

app = create_fastapi_app(env_factory, Action, Observation)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

        uv run --project . server
        python -m server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()