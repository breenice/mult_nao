# Shared config for nao_client and llm_server.
# Loads from config/agent_config.json: agents (name, ip, bigo_personality).
# Base port for ZMQ is 5555.

import json
import os
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
AGENT_CONFIG_PATH = _CONFIG_DIR / "agent_config.json"

NAO_BASE_PORT = 5555
NAO_PORT_MAX_OFFSET = 9


def _load_agent_config():
    # load dict of agents from config/agent_config.json
    if not AGENT_CONFIG_PATH.exists() or AGENT_CONFIG_PATH.stat().st_size == 0:
        return {"agents": []}
    try:
        with open(AGENT_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"agents": []}


_config = _load_agent_config()
AGENTS = _config.get("agents") or []

# Robot name -> IP map
ROBOT_IPS = {str(a["name"]).strip(): str(a["ip"]).strip() for a in AGENTS if isinstance(a, dict) and "name" in a and "ip" in a}
