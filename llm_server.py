import json
import base64
import asyncio
from pathlib import Path

import zmq

from openai import OpenAI
from tinydb import TinyDB

from helpers.personality.personality_module import PersonalityEngine
from helpers.memory.memory_agent import MemoryAgent
from helpers.input_names import ensure_input_folder
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from helpers.robot_agent_virtual import RobotAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


_TOOLS_JSON_PATH = Path(__file__).resolve().parent / "helpers" / "tools.json"
with open(_TOOLS_JSON_PATH, "r") as f:
    NAO_TOOLS = json.load(f)


def _get_tool_schema(tool_name: str) -> dict:
    # return the function schema for a tool name from tools.json, or None
    for t in NAO_TOOLS:
        if t.get("function", {}).get("name") == tool_name:
            return t["function"].get("parameters", {})
    return None


def _filter_args_by_schema(tool_name: str, args: dict) -> dict:
    # keep only args that appear in the tool's parameters in json file
    schema = _get_tool_schema(tool_name)
    if not schema or "properties" not in schema:
        return args
    allowed = set(schema["properties"].keys())
    return {k: v for k, v in args.items() if k in allowed}


# NAO socket (send action commands to nao_client)
NAO_SOCKET_HOST = "localhost"
NAO_SOCKET_PORT = 5555
_nao_socket = None
_nao_context = None


def _init_nao_socket():
    global _nao_socket, _nao_context
    if _nao_socket is not None:
        return
    try:
        _nao_context = zmq.Context()
        _nao_socket = _nao_context.socket(zmq.REQ)
        _nao_socket.setsockopt(zmq.LINGER, 0)
        _nao_socket.setsockopt(zmq.RCVTIMEO, 5000)
        _nao_socket.connect("tcp://%s:%s" % (NAO_SOCKET_HOST, NAO_SOCKET_PORT))
    except Exception as e:
        print("[LLM] NAO socket not available (%s). Actions will be printed only." % e)
        _nao_socket = None


def _send_nao_command_sync(tool: str, args: dict) -> str:
    # blocks until send to nao_client. Returns reply string. Args filtered by tools.json schema.
    if _nao_socket is None:
        return ""
    args = _filter_args_by_schema(tool, args)
    cmd = {"tool": tool, "args": args}
    _nao_socket.send_string(json.dumps(cmd))
    return _nao_socket.recv_string()


async def send_nao_command(agent_name: str, tool: str, args: dict):
    # sends action to nao_client and prints what was sent.
    print("[ACTION] %s -> %s %s" % (agent_name, tool, args))
    loop = asyncio.get_event_loop()
    try:
        reply = await loop.run_in_executor(None, lambda: _send_nao_command_sync(tool, args))
        if reply:
            print("[NAO] %s" % reply)
    except Exception as e:
        print("[NAO] Error: %s" % e)


# ====== Model Client ======
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)


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
    #  makes NAO agent and uses input/<name>/ for personality, memory, image, sound. Tools from tools.json
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

    def _make_nao_tool(tool_name: str):
        async def fn(**kwargs):
            await send_nao_command(name, tool_name, kwargs)
            return str(kwargs)
        fn.__name__ = tool_name
        return fn

    agent_tools = []
    tool_names = []
    for t in NAO_TOOLS:
        fn_name = t.get("function", {}).get("name")
        if not fn_name:
            continue
        tool_names.append(fn_name)
        if fn_name == "reason_with_vision":
            async def vision_tool(prompt: str, **_):
                return await reason_with_vision(name, robot.memory, prompt, see_path)
            vision_tool.__name__ = "reason_with_vision"
            agent_tools.append(vision_tool)
        else:
            agent_tools.append(_make_nao_tool(fn_name))

    tool_list_str = ", ".join(tool_names)

    return AssistantAgent(
        name=name,
        model_client=model_client,
        tools=agent_tools,
        system_message=f"""
            You are {name}, a NAO robot with a unique personality.
            Personality:
            {personality_text}

            Use your tools to interact with the world. Available tools (from tools.json): {tool_list_str}.
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
    _init_nao_socket()
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
