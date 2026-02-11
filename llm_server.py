import argparse
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, Any, List, Tuple

import zmq

from openai import OpenAI
from tinydb import TinyDB

from helpers.personality.personality_module import PersonalityEngine
from helpers.memory.memory_agent import MemoryAgent
from helpers.input_names import ensure_input_folder
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from helpers.robot_agent_virtual import RobotAgent
from helpers.nao_config import ROBOT_IPS, NAO_BASE_PORT
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


def _json_type_to_annotation(prop: dict):
    t = prop.get("type", "string")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    return str


def _make_nao_tool_with_signature(agent_name: str, tool_name: str, params_schema: dict, send_fn):
    """Build an async tool with explicit params and type annotations. send_fn(tool_name, kwargs) is the async send."""
    props = params_schema.get("properties") or {}
    required = set(params_schema.get("required") or [])
    if not props:
        async def _no_arg_tool() -> str:
            await send_fn(tool_name, {})
            return "ok"
        _no_arg_tool.__name__ = tool_name
        return _no_arg_tool

    annotations = {}
    for pname, pdef in props.items():
        annotations[pname] = _json_type_to_annotation(pdef if isinstance(pdef, dict) else {})

    param_parts = []
    for pname in sorted(props.keys()):
        ann = annotations[pname]
        ann_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        if pname in required:
            param_parts.append(f"{pname}: {ann_name}")
        else:
            param_parts.append(f"{pname}: Optional[{ann_name}] = None")

    params_str = ", ".join(param_parts)
    body = """
kwargs = {k: v for k, v in locals().items() if v is not None}
await send_fn(tool_name, kwargs)
return str(kwargs)
"""
    exec_globals = {"send_fn": send_fn, "tool_name": tool_name}
    exec(
        f"async def _fn({params_str}):\n" + "\n".join(" " + line for line in body.strip().split("\n")),
        exec_globals,
        local := {},
    )
    fn = local["_fn"]
    fn.__name__ = tool_name
    fn.__annotations__ = {p: annotations[p] for p in props.keys()}
    return fn


# NAO socket (per-agent; send action commands to nao_client)
NAO_SOCKET_HOST = "localhost"
_nao_context = None


def _create_nao_socket(port: int):
    """Create and connect a ZMQ REQ socket to host:port. Returns socket or None on failure."""
    global _nao_context
    if _nao_context is None:
        _nao_context = zmq.Context()
    try:
        sock = _nao_context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.connect("tcp://%s:%s" % (NAO_SOCKET_HOST, port))
        return sock
    except Exception as e:
        print("[LLM] NAO socket not available for port %s (%s). Actions will be printed only." % (port, e))
        return None


def _send_nao_command_sync(tool: str, args: dict, socket) -> str:
    """Blocking send to nao_client. Args filtered by tools.json schema. Returns reply or '' if socket is None."""
    if socket is None:
        return ""
    args = _filter_args_by_schema(tool, args)
    cmd = {"tool": tool, "args": args}
    socket.send_string(json.dumps(cmd))
    return socket.recv_string()


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
def create_nao_agent(name: str, root: Path, robot_name: str, port: int, socket):
    """Create a NAO agent with a physical robot (robot_name) and its own socket (port)."""
    folder = ensure_input_folder(root, name)
    personality_path = str(folder / "personality.json")
    memory_path = str(folder / "memory")
    see_path = folder / "see.jpg"
    is_text = robot_name not in ROBOT_IPS

    client = OpenAI()

    robot = RobotAgent(
        name=name,
        client=client,
        personality_path=personality_path,
        memory_path=memory_path
    )

    personality_text = robot.personality.to_prompt_text()

    async def send_for_this_agent(tool: str, args: dict):
        print("[ACTION] %s (%s) -> %s %s" % (name, robot_name + (" text" if is_text else ""), tool, args))
        loop = asyncio.get_event_loop()
        try:
            reply = await loop.run_in_executor(None, lambda: _send_nao_command_sync(tool, args, socket))
            if reply:
                print("[NAO] %s" % reply)
        except Exception as e:
            print("[NAO] Error: %s" % e)

    agent_tools = []
    tool_names = []
    for t in NAO_TOOLS:
        fn_name = t.get("function", {}).get("name")
        if not fn_name:
            continue
        tool_names.append(fn_name)
        if fn_name == "reason_with_vision":
            async def vision_tool(prompt: str) -> str:
                return await reason_with_vision(name, robot.memory, prompt, see_path)
            vision_tool.__name__ = "reason_with_vision"
            agent_tools.append(vision_tool)
        else:
            params_schema = t.get("function", {}).get("parameters") or {}
            agent_tools.append(_make_nao_tool_with_signature(name, fn_name, params_schema, send_for_this_agent))

    tool_list_str = ", ".join(tool_names)

    return AssistantAgent(
        name=name,
        model_client=model_client,
        tools=agent_tools,
        system_message=f"""
            You are {name}, a NAO robot (robot: {robot_name}) with a unique personality.
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
def _parse_agent_args() -> List[Tuple[str, str]]:
    parser = argparse.ArgumentParser(description="Run multi-NAO chat. Pass agent/robot pairs or use nao_config.AGENTS.")
    parser.add_argument(
        "--agent",
        nargs=2,
        action="append",
        metavar=("NAME", "ROBOT"),
        help="Agent name and robot name (repeatable). e.g. --agent Beep ANGEL --agent Moop JOURNEY",
    )
    args = parser.parse_args()
    if args.agent:
        return [tuple(pair) for pair in args.agent]
    return []


async def multi_nao_chat(agents_list: List[Tuple[str, str]]):
    if not agents_list:
        raise ValueError("give at least one --agent NAME ROBOT (e.g. --agent <display-name> <physical-robot-name> --agent Casper JOURNEY)")
    root = Path(__file__).resolve().parent
    nao_agents = []
    for i, (agent_name, robot_name) in enumerate(agents_list):
        port = NAO_BASE_PORT + i
        sock = _create_nao_socket(port)
        nao_agents.append(create_nao_agent(agent_name, root, robot_name, port, sock))

    human = UserProxyAgent(name="Human", input_func=input)

    agents = [human] + nao_agents

    current_message = "Hello team!"

    for round_idx in range(5):
        print(f"\n===== ROUND {round_idx + 1} =====")

        # Human turn
        print(f"\nHuman responding to: {current_message}")
        human_result = await human.run(task=current_message)
        current_message = extract_text(human_result)
        print(f"[Human]: {current_message}")

        # NAO turns
        for nao in nao_agents:
            print(f"\n{nao.name} responding to: {current_message}")
            result = await nao.run(task=current_message)
            text = extract_text(result)

            # ONLY advance conversation on real speech
            if text.strip():
                current_message = text

            print(f"[{nao.name}]: {text}")



# ====== Run ======
if __name__ == "__main__":
    agents_list = _parse_agent_args()
    asyncio.run(multi_nao_chat(agents_list))
