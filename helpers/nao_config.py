# shared config for nao_client and llm_server 
# physical robot IP mapping 
# base port for ZMQ is 5555

ROBOT_IPS = {
    "FRIENDLY": "192.168.1.128",
    "ANGEL": "192.168.1.35",
    "JOURNEY": "192.168.1.55",
    "SOOMIN": "192.168.1.",
    "SAM": "192.168.1.128",
}

NAO_BASE_PORT = 5555

# increment by for each agent
NAO_PORT_MAX_OFFSET = 9

# default path to agents config (display name + robot name); used by llm_server and nao_client
AGENTS_FILE = "agents.json"
