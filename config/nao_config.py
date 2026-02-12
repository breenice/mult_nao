# Shared config for nao_client and llm_server.
# Loads from config/agent_config.json: agents (name, ip, bigo_personality).
# Base port for ZMQ is 5555.
# Compatible with Python 2.7 (ma_clients) and Python 3 (ma_server).

import json
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "agent_config.json")

NAO_BASE_PORT = 5555
NAO_PORT_MAX_OFFSET = 9


def _load_agent_config():
    # load dict of agents from config/agent_config.json
    if not os.path.isfile(AGENT_CONFIG_PATH) or os.path.getsize(AGENT_CONFIG_PATH) == 0:
        return {"agents": []}
    try:
        with open(AGENT_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {"agents": []}


_config = _load_agent_config()
AGENTS = _config.get("agents") or []

# Robot name -> IP map
ROBOT_IPS = dict(
    (str(a["name"]).strip(), str(a["ip"]).strip())
    for a in AGENTS
    if isinstance(a, dict) and "name" in a and "ip" in a
)
