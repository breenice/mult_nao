import argparse
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, Any, List, Tuple, Sequence

import zmq

from openai import OpenAI
from tinydb import TinyDB

from helpers.personality.personality_module import PersonalityEngine
from helpers.memory.memory_agent import MemoryAgent
from helpers.input_names import ensure_input_folder
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from helpers.robot_agent_virtual import RobotAgent
from helpers.nao_config import ROBOT_IPS, NAO_BASE_PORT, AGENTS_FILE
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


def _get_speak_tool() -> dict:
    for t in NAO_TOOLS:
        if t.get("function", {}).get("name") == "speak":
            return t
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
    # async tool with explicit params and type annotations
    # to send: send_fn(tool_name, kwargs)
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

    param_order = sorted(props.keys(), key=lambda p: (p not in required, p))
    param_parts = []
    for pname in param_order:
        ann = annotations[pname]
        ann_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        if pname in required:
            param_parts.append(f"{pname}: {ann_name}")
        else:
            param_parts.append(f"{pname}: Optional[{ann_name}] = None")

    params_str = ", ".join(param_parts)
    body_lines = [
        "kwargs = {k: v for k, v in locals().items() if v is not None}",
        "await send_fn(tool_name, kwargs)",
        "return str(kwargs)",
    ]
    body = "\n".join("    " + line for line in body_lines)
    exec_globals = {"send_fn": send_fn, "tool_name": tool_name, "Optional": Optional}
    exec(
        f"async def _fn({params_str}):\n{body}",
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
    # create and connect a ZMQ REQ socket to host:port
    # returns socket or None on failure
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


def _send_nao_command_sync(tool: str, args: dict, socket, agent_index: int = 0) -> str:
    """Blocking send to nao_client. Args filtered by tools.json schema. Returns reply or '' if socket is None."""
    if socket is None:
        return ""
    args = _filter_args_by_schema(tool, args)
    cmd = {"agent": agent_index, "tool": tool, "args": args}
    socket.send_string(json.dumps(cmd))
    return socket.recv_string()


# ====== Model Client ======
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)


def _see_jpg_is_valid(see_path: Path) -> bool:
    try:
        if not see_path.exists() or not see_path.is_file():
            return False
        size = see_path.stat().st_size
        # Placeholder images are tiny ~200 bytes
        return size > 500
    except OSError:
        return False


async def reason_with_vision(
    agent_name: str,
    memory_agent: MemoryAgent,
    prompt: str,
    see_path: Path,
    send_fn: Optional[Any] = None,
):
    # use image + prompt, must send an action (speak as fallback)
    print("[VISION] reason_with_vision called for %s, prompt=%r, see_path=%s" % (agent_name, prompt, see_path))
    memory = memory_agent.run_once(prompt)
    if not _see_jpg_is_valid(see_path):
        print("[VISION] Skipped (see.jpg missing or too small): %s" % see_path)
        return "No camera image available yet. The camera may not have sent a frame, or the image file is missing or too small."
    try:
        with open(see_path, "rb") as img_file:
            data = img_file.read()
        if not data:
            print("[VISION] Skipped (empty file): %s" % see_path)
            return "No camera image available yet."
        image_b64 = base64.b64encode(data).decode("utf-8")
    except OSError as e:
        print("[VISION] Skipped (read error): %s - %s" % (see_path, e))
        return "Could not read the camera image."

    speak_tool = _get_speak_tool()
    if send_fn and speak_tool:
        print("[VISION] Calling vision API (speak-only) for %s (%d bytes)" % (agent_name, len(data)))
        client = OpenAI()
        system = (
            f"""You are {agent_name}, a NAO robot observing through a camera.
            "Answer ONLY with the speak tool. Limit to 1 sentence.
            ---- Memory ----
            {memory}
            """
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            tools=[speak_tool],
            tool_choice="required",
        )
        msg = response.choices[0].message
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = getattr(call.function, "name", None) if hasattr(call, "function") else None
                args_str = getattr(call.function, "arguments", None) if hasattr(call, "function") else None
                if name == "speak" and args_str:
                    try:
                        args = json.loads(args_str)
                        await send_fn("speak", args)
                        return "Spoke to user."
                    except (json.JSONDecodeError, TypeError):
                        pass
        # Fallback if no valid speak call
        return "Could not form a response."
    # No send_fn or no speak tool: return text description for agent to use
    print("[VISION] Calling vision API for %s (%d bytes)" % (agent_name, len(data)))
    response = await model_client.acreate(
        messages=[
            {"role": "system", "content": (
                f"You are {agent_name}, a NAO robot with a camera. Describe what you see in one sentence."
            )},
            {"role": "user", "content": f"Vision prompt: {prompt}\nMemory: {memory}\nImage: data:image/jpg;base64,{image_b64}"}
        ]
    )
    return response.choices[0].message.content


# ====== NAO Agent Factory ======
def create_nao_agent(name: str, root: Path, robot_name: str, agent_index: int, socket):
    # create a NAO agent with a physical robot (robot_name)
    # uses shared socket with agent_index in each message
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
            reply = await loop.run_in_executor(
                None, lambda: _send_nao_command_sync(tool, args, socket, agent_index)
            )
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
                return await reason_with_vision(name, robot.memory, prompt, see_path, send_fn=send_for_this_agent)
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

            When you use reason_with_vision: the result describes what the camera sees. You must then use that result to make follow-up tool calls in the same turn—e.g. call speak() to report what you saw to the user, or wave()/nod() if you see a person. Do not finish your turn with only the vision result; chain into at least one action (speak, wave, etc.) so the user gets a response or the robot acts on what it saw.
        """,
        reflect_on_tool_use=True,
        model_client_stream=True
    )

def extract_text(result):
    for msg in reversed(result.messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return ""


def _action_tool_names() -> frozenset:
    # all tools executable by nao client
    names = set()
    for t in NAO_TOOLS:
        name = t.get("function", {}).get("name")
        if name and name != "reason_with_vision":
            names.add(name)
    return frozenset(names)


def _get_vision_result_and_followup(messages: Sequence[Any]) -> Tuple[Optional[str], bool]:
    """
    Walk result.messages in order. Return (vision_result, had_follow_up_action).
    If reason_with_vision was called, vision_result is its last return value (string).
    had_follow_up_action is True if any other tool from tools.json was called after the last reason_with_vision.
    """
    action_tools = _action_tool_names()
    vision_result = None
    had_follow_up_action = False
    pending_request_names = []

    for msg in messages:
        try:
            type_name = getattr(msg, "type", None) or type(msg).__name__
        except Exception:
            type_name = ""
        try:
            content = getattr(msg, "content", None)
        except Exception:
            content = None

        if "ToolCallRequest" in type_name or type_name == "tool_call_request":
            if isinstance(content, list):
                names = []
                for call in content:
                    name = None
                    if isinstance(call, dict):
                        name = call.get("name") or (call.get("function") or {}).get("name")
                    elif hasattr(call, "name"):
                        name = getattr(call, "name", None)
                    elif hasattr(call, "function"):
                        name = getattr(getattr(call, "function", None), "name", None)
                    names.append(name or "")
                if names:
                    vision_index = -1
                    for i, n in enumerate(names):
                        if n == "reason_with_vision":
                            vision_index = i
                            had_follow_up_action = False
                        elif vision_index >= 0 and n in action_tools:
                            had_follow_up_action = True
                    pending_request_names = names

        elif ("ToolCallExecution" in type_name or type_name == "tool_call_execution") and pending_request_names:
            if isinstance(content, list):
                results = []
                for res in content:
                    result_text = None
                    if isinstance(res, dict):
                        result_text = res.get("result") or res.get("content") or res.get("output")
                    elif hasattr(res, "result"):
                        result_text = getattr(res, "result", None)
                    elif hasattr(res, "content"):
                        result_text = getattr(res, "content", None)
                    results.append(result_text if isinstance(result_text, str) else "")
                for i, name in enumerate(pending_request_names):
                    if i < len(results) and name == "reason_with_vision" and results[i]:
                        vision_result = results[i]
            pending_request_names = []

    return (vision_result, had_follow_up_action)


# ====== Multi-Agent Conversation ======
def _load_agents_from_file(path: Path) -> List[Tuple[str, str]]:
    # load list of (display_name, robot_name) from JSON file
    # returns [] if file missing or invalid
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, dict) and "display" in item and "robot" in item:
            result.append((str(item["display"]).strip(), str(item["robot"]).strip()))
    return result


def _parse_agent_args() -> List[Tuple[str, str]]:
    root = Path(__file__).resolve().parent
    default_agents_path = root / AGENTS_FILE
    parser = argparse.ArgumentParser(
        description="Run multi-NAO chat. Agents from %s by default, or pass --agent NAME ROBOT." % default_agents_path
    )
    parser.add_argument(
        "--agent",
        nargs=2,
        action="append",
        metavar=("NAME", "ROBOT"),
        help="Agent name and robot name (repeatable). Overrides agents file when provided.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_agents_path,
        help="Path to agents JSON file (default: %s)" % AGENTS_FILE,
    )
    args = parser.parse_args()
    if args.agent:
        return [tuple(pair) for pair in args.agent]
    agents_list = _load_agents_from_file(args.config)
    return agents_list


async def multi_nao_chat(agents_list: List[Tuple[str, str]]):
    if not agents_list:
        raise ValueError("give at least one --agent NAME ROBOT (e.g. --agent <display-name> <physical-robot-name> --agent Casper JOURNEY)")
    root = Path(__file__).resolve().parent
    sock = _create_nao_socket(NAO_BASE_PORT)
    nao_agents = []
    for i, (agent_name, robot_name) in enumerate(agents_list):
        nao_agents.append(create_nao_agent(agent_name, root, robot_name, i, sock))

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
