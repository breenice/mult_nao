#!/usr/bin/env python3
"""
RobotAgent: personality + memory only. NAO AssistantAgent is created via create_nao_agent in ma_server.
- helper for AssistantAgent in autogen_agentchat.agents
"""
import asyncio
from pathlib import Path

from modules.memory.memory_agent import MemoryAgent
from modules.personality.personality_module import PersonalityEngine

_modules_DIR = Path(__file__).resolve().parent
# personality/big5.json lives under modules/ (sibling of llm/)
DEFAULT_BIG5_PATH = str(_modules_DIR.parent / "personality" / "big5.json")


class RobotAgent:
    # data/helper for personality and memory. The actual chat agent is built by create_nao_agent in ma_server.

    def __init__(self, name, client, personality_path: str, memory_path: str | None = None):
        self.name = name
        self.client = client
        self.personality = PersonalityEngine(
            big5_path=DEFAULT_BIG5_PATH,
            personality_path=personality_path
        )
        self.memory = MemoryAgent(db_path=memory_path)


if __name__ == "__main__":
    # Use the same agent factory as the server: one code path for "NAO agent"
    from config.nao_config import NAO_BASE_PORT
    from ma_server import _create_nao_socket, create_nao_agent, extract_text

    root = Path(__file__).resolve().parent.parent
    robot_name = "ANGEL"
    agent_config = {
        "name": robot_name,
        "ip": "127.0.0.1",
        "bigo_personality": None,
    }
    sock = _create_nao_socket(NAO_BASE_PORT)
    agent = create_nao_agent(agent_config, root, 0, sock, session_context={})

    async def chat_loop():
        while True:
            user_prompt = input("User: ")
            if user_prompt.strip().lower() == "exit":
                break
            result = await agent.run(task=user_prompt)
            text = extract_text(result)
            print(f"{agent.name}: {text}")

    asyncio.run(chat_loop())
