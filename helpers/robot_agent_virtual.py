#!/usr/bin/env python3
import json
import time
import argparse
from pathlib import Path
from enum import Enum
from openai import OpenAI

from helpers.memory.memory_agent import MemoryAgent
from helpers.personality.personality_module import PersonalityEngine

_HELPERS_DIR = Path(__file__).resolve().parent
DEFAULT_BIG5_PATH = str(_HELPERS_DIR / "personality" / "big5.json")

tools = [
    {
    "type": "function",
    "function": {
        "name": "text_respond",
        "description": "respond in text",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "This is the text you, the robot, will say"
                },
            },
            "required": ["message"],
            "additionalProperties": False
            }
        }
    },
]

class AgentState(Enum):
    LISTEN = 1
    THINK = 2
    ACT = 3
    EXIT = 4


class RobotAgent:
    def __init__(self, name, client, personality_path: str, memory_path: str | None = None):
        self.name = name
        self.client = client
        self.personality = PersonalityEngine(
            big5_path=DEFAULT_BIG5_PATH,
            personality_path=personality_path
        )
        self.memory = MemoryAgent(db_path=memory_path)

    def planner(self, user_prompt: str):
        memory = self.memory.run_once(user_prompt)

        system_prompt = f"""
            You are a NAO robot with a physical body.
            You MUST always respond with tool calls only.
            You must act in line with your personality.
            Your name is {self.name}.
            You have are able to view your sourroundings through your available tools.

            ---- Memory ----
            {memory}

            ---- Personality ----
            {self.personality.to_prompt_text()}

            Speak in 1 sentence.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools = tools,
            tool_choice="auto"
        )

        return response.choices[0].message.tool_calls or []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["speak", "text"], default="text")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    client = OpenAI()
    agent = RobotAgent(
        name="Dearana",
        client=client,
        personality_path=str(root / "input" / "Dearana" / "personality.json"),
        memory_path=str(root / "input" / "Dearana" / "memory")
    )

    state = AgentState.LISTEN

    while state != AgentState.EXIT:
        if state == AgentState.LISTEN:
            user_prompt = input("User: ")
            if user_prompt.lower() == "exit":
                state = AgentState.EXIT
            else:
                state = AgentState.THINK

        elif state == AgentState.THINK:
            plan = agent.planner(user_prompt)

            answer = json.loads(plan[0].function.arguments)
            print(f"{agent.name}: {answer['message']}")
            state = AgentState.LISTEN


        