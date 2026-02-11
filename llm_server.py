import json
import base64
import asyncio
from pathlib import Path

from openai import OpenAI
from tinydb import TinyDB

from helpers.personality_module import PersonalityEngine
from helpers.memory.memory_agent import MemoryAgent
from helpers.input_names import ensure_input_folder
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent 

from robot_agent_virtual import RobotAgent 
from autogen_ext.models.openai import OpenAIChatCompletionClient



# ====== Model Client ======
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)

# ====== Define NAO Tools ======
async def speak(agent_name: str, text: str):
    print(f"[{agent_name} SPEAKS]: {text}")
    return text

async def move_head(agent_name: str, direction: str):
    print(f"[{agent_name} MOVES HEAD]: {direction}")
    return direction

async def reason_with_vision(agent_name: str, memory_agent: MemoryAgent, prompt: str, see_path: Path):
    memory = memory_agent.run_once(prompt)
    with open(see_path, "rb") as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    response = await model_client.acreate(
        messages=[
            {"role": "system", "content": f"You are {agent_name}, a NAO robot observing through a camera. Answer in 1 sentence."},
            {"role": "user", "content": f"Vision prompt: {prompt}\nMemory: {memory}\nImage: data:image/jpg;base64,{image_b64}"}
        ]
    )
    return response.choices[0].message.content


# ====== NAO Agent Factory ======
def create_nao_agent(name: str, root: Path):
    """Create a NAO agent. Uses input/<name>/ for personality, memory, image, sound."""
    folder = ensure_input_folder(root, name)
    personality_path = str(folder / "personality.json")
    memory_path = str(folder / "memory")
    see_path = folder / "see.jpg"

    client = OpenAI()

    robot = RobotAgent(
        name=name,
        client=client,
        personality_path=personality_path,
        memory_path=memory_path
    )

    personality_text = robot.personality.to_prompt_text()

    async def speak_tool(text: str):
        return await speak(name, text)

    async def move_head_tool(direction: str):
        return await move_head(name, direction)

    async def vision_tool(prompt: str):
        return await reason_with_vision(name, robot.memory, prompt, see_path)

    return AssistantAgent(
        name=name,
        model_client=model_client,
        tools=[speak_tool, move_head_tool, vision_tool],
        system_message=f"""
            You are {name}, a NAO robot with a unique personality.
            Personality:
            {personality_text}

            Use your tools to interact with the world: speak, move_head, reason_with_vision.
            Always respond with tool calls only.
        """,
        reflect_on_tool_use=True,
        model_client_stream=True
    )

def extract_text(result):
    for msg in reversed(result.messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return ""


# ====== Multi-Agent Conversation ======
async def multi_nao_chat():
    root = Path(__file__).resolve().parent
    nao1 = create_nao_agent("NAO_Alpha", root)
    nao2 = create_nao_agent("NAO_Beta", root)
    nao3 = create_nao_agent("NAO_Gamma", root)

    human = UserProxyAgent(name="Human", input_func=input)

    agents = [human, nao1, nao2, nao3]

    current_message = "Hello team!"

    for round_idx in range(5):
        print(f"\n===== ROUND {round_idx + 1} =====")

        # Human turn
        print(f"\nHuman responding to: {current_message}")
        human_result = await human.run(task=current_message)
        current_message = extract_text(human_result)
        print(f"[Human]: {current_message}")

        # NAO turns
        for nao in [nao1, nao2, nao3]:
            print(f"\n{nao.name} responding to: {current_message}")
            result = await nao.run(task=current_message)
            text = extract_text(result)

            # ONLY advance conversation on real speech
            if text.strip():
                current_message = text

            print(f"[{nao.name}]: {text}")



# ====== Run ======
if __name__ == "__main__":
    asyncio.run(multi_nao_chat())
