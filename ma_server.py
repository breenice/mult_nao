import argparse
import json
import base64
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Optional, Any, List, Tuple, Sequence, Dict
import speech_recognition as sr

import zmq

from openai import OpenAI
from tinydb import TinyDB

from modules.memory.memory_agent import MemoryAgent
from modules.llm.input_names import ensure_session_folder, get_session_dir
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core.tools import FunctionTool

from modules.llm.robot_agent_virtual import RobotAgent
from config.nao_config import ROBOT_IPS, NAO_BASE_PORT, AGENT_CONFIG_PATH
from autogen_ext.models.openai import OpenAIChatCompletionClient

_TOOLS_JSON_PATH = Path(__file__).resolve().parent / "modules" / "actions" / "tools.json"
with open(_TOOLS_JSON_PATH, "r") as f:
    NAO_TOOLS = json.load(f)


class SessionLogger:
    # populates JSONL log per robot under session/<robot_name>/conversation<time-stamp>.jsonl
    def __init__(self, root: Path, robot_name: str):
        self.root = root
        self.robot_name = robot_name
        self._log_path = get_session_dir(root, robot_name) / f"conversation_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict, round_id: Optional[int] = None) -> None:
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "round": round_id,
            "type": event_type,
            "payload": payload,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
        except OSError:
            pass


def _get_tool_schema(tool_name: str) -> dict:
    # return the function schema for a tool name from actions/tools.json, or None
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
    # keep only args that appear in the tool's parameters in actions/tools.json
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

# ====== VLM Turn Manager (configurable) ======
HUMAN_NAME = "Human"
MAX_CONSECUTIVE_ROBOT_TURNS = 2   # after this many robot turns in a row, hand back to Human
VLM_THRESHOLD = 0.5   # theta: agents with r_i >= this are in S_t
VLM_TOP_K: Optional[int] = 2   # optional: max number of speakers per turn (None = no limit)
VLM_SCORE_MODEL = "gpt-4o"   # vision-capable model for scoring


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
            f"You are {agent_name}, a NAO robot observing through a camera. "
            "Answer ONLY with the speak tool. Limit to 1 sentence. Speak in first person (e.g. I see...).\n\n"
            f"---- Memory ----\n{memory}\n"
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


# ====== NAO Agent Factory (autogen agents from config: robot name, ip, personality) ======
def create_nao_agent(agent_config: dict, root: Path, agent_index: int, socket, session_context: Optional[dict] = None):
    # agent_config: name (physical robot name), ip, bigo_personality. All paths and identity use this name.
    # Persistent data: session/<robot_name>/ (personality.json, memory db, see.jpg, sound.wav, conversation.jsonl)
    robot_name = str(agent_config.get("name", "Agent")).strip()
    bigo = agent_config.get("bigo_personality")
    folder = ensure_session_folder(root, robot_name, bigo_personality=bigo)
    personality_path = str(folder / "personality.json")
    memory_path = str(folder / "memory")
    see_path = folder / "see.jpg"
    is_text = robot_name not in ROBOT_IPS

    session_logger = SessionLogger(root, robot_name)

    client = OpenAI()

    robot = RobotAgent(
        name=robot_name,
        client=client,
        personality_path=personality_path,
        memory_path=memory_path
    )

    personality_text = robot.personality.to_prompt_text()

    async def send_for_this_agent(tool: str, args: dict):
        print("[ACTION] %s (%s) -> %s %s" % (robot_name, "text" if is_text else "robot", tool, args))
        round_id = session_context.get("round") if session_context else None
        loop = asyncio.get_event_loop()
        try:
            reply = await loop.run_in_executor(
                None, lambda: _send_nao_command_sync(tool, args, socket, agent_index)
            )
            if reply:
                print("[NAO] %s" % reply)
            session_logger.log(
                "action",
                {"tool": tool, "args": args, "status": "ok", "result": reply},
                round_id=round_id,
            )
        except Exception as e:
            print("[NAO] Error: %s" % e)
            session_logger.log(
                "action",
                {"tool": tool, "args": args, "status": "error", "error": str(e)},
                round_id=round_id,
            )

    agent_tools = []
    tool_names = []
    for t in NAO_TOOLS:
        fn_def = t.get("function") or {}
        fn_name = fn_def.get("name")
        if not fn_name:
            continue
        tool_names.append(fn_name)
        description = fn_def.get("description", fn_name) or fn_name
        if fn_name == "reason_with_vision":
            async def vision_tool(prompt: str) -> str:
                return await reason_with_vision(robot_name, robot.memory, prompt, see_path, send_fn=send_for_this_agent)
            vision_tool.__name__ = "reason_with_vision"
            agent_tools.append(FunctionTool(vision_tool, description=description, name=fn_name))
        else:
            params_schema = fn_def.get("parameters") or {}
            callable_fn = _make_nao_tool_with_signature(robot_name, fn_name, params_schema, send_for_this_agent)
            agent_tools.append(FunctionTool(callable_fn, description=description, name=fn_name))

    tool_list_str = ", ".join(tool_names)

    agent = AssistantAgent(
        name=robot_name,
        description=personality_text or "NAO robot.",
        model_client=model_client,
        tools=agent_tools,
        system_message=f"""
            You are {robot_name}, a NAO robot with a unique personality.
            Personality:
            {personality_text}

            Use your tools to interact with the world. Available tools (from tools.json): {tool_list_str}.
            Always respond with tool calls only.

            When you use reason_with_vision: the result describes what the camera sees. You must then use that result to make follow-up tool calls in the same turn—e.g. call speak() to report what you saw to the user, or wave()/nod() if you see a person. Do not finish your turn with only the vision result; chain into at least one action (speak, wave, etc.) so the user gets a response or the robot acts on what it saw.
        """,
        reflect_on_tool_use=True,
        model_client_stream=True
    )
    agent.session_logger = session_logger
    return agent


def extract_text(result):
    for msg in reversed(result.messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return ""


# ====== Multi-Agent Conversation ======
def _load_agents_from_config(path: Path) -> List[dict]:
    # Load agents from config/agent_config.json format: {"agents": [{name, ip, bigo_personality}, ...]}
    # Legacy [{"display", "robot"}] uses robot as the single name (name only, no display).
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict) and "agents" in data:
        return list(data["agents"]) if isinstance(data["agents"], list) else []
    if isinstance(data, list):
        # Legacy: [{"display", "robot"}] -> agent dicts with name = robot only
        result = []
        for item in data:
            if isinstance(item, dict) and "robot" in item:
                robot = str(item["robot"]).strip()
                result.append({
                    "name": robot,
                    "ip": ROBOT_IPS.get(robot, "127.0.0.1"),
                    "bigo_personality": None,
                })
        return result
    return []


def _parse_agent_args() -> Tuple[List[dict], bool]:
    root = Path(__file__).resolve().parent
    default_config_path = root / "config" / "agent_config.json"
    parser = argparse.ArgumentParser(
        description="Run multi-NAO chat. Agents from config/agent_config.json by default, or pass --agent ROBOT (repeatable)."
    )
    parser.add_argument(
        "--agent",
        action="append",
        metavar="ROBOT",
        help="Robot name (repeatable, e.g. --agent ANGEL --agent SAM). Overrides config when provided.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="Path to agent config JSON (default: config/agent_config.json)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Use microphone for human turn: record until silence, transcribe with Whisper.",
    )
    args = parser.parse_args()
    if args.agent:
        robot_names = [str(r).strip() for r in args.agent]
        agents_list = [
            {"name": r, "ip": ROBOT_IPS.get(r, "127.0.0.1"), "bigo_personality": None}
            for r in robot_names
        ]
    else:
        agents_list = _load_agents_from_config(args.config)
    return agents_list, args.listen


_LISTEN_WAV_PATH = Path(__file__).resolve().parent / "input" / "listen.wav"
_LISTEN_TIMEOUT = 5
_LISTEN_PHRASE_TIME_LIMIT = 10
_LISTEN_MAX_EMPTY_RETRIES = 5


def listen_for_human_input(prompt: str) -> str:
    # record from microphone until user stops speaking, when --listen is set
    # transcribe with Whisper

    recognizer = sr.Recognizer()
    client = OpenAI()
    _LISTEN_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(_LISTEN_MAX_EMPTY_RETRIES):
        print(prompt)
        print("Listening... (speak now; recording stops when you pause)")
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(
                    source,
                    timeout=_LISTEN_TIMEOUT,
                    phrase_time_limit=_LISTEN_PHRASE_TIME_LIMIT,
                )
        except sr.WaitTimeoutError:
            print("No speech detected. Try again.")
            continue
        except OSError as e:
            print("[Listen] Microphone error: %s. Try again." % e)
            continue

        wav_data = audio.get_wav_data()
        _LISTEN_WAV_PATH.write_bytes(wav_data)

        try:
            with open(_LISTEN_WAV_PATH, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
        except Exception as e:
            print("[Listen] Transcription error: %s. Try again." % e)
            continue

        if transcript and isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
        print("No speech detected. Try again.")

    print("[Listen] Max retries reached. Falling back to keyboard input.")
    return input(prompt)


def _message_source_name(msg) -> Optional[str]:
    """Return the source/agent name for a message if present."""
    try:
        return getattr(msg, "source", None) or getattr(msg, "name", None)
    except Exception:
        return None


def _message_text(msg) -> str:
    # extract displayable text from a message
    try:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and "text" in c:
                        parts.append(c["text"])
                elif hasattr(c, "text"):
                    parts.append(getattr(c, "text", ""))
            return " ".join(parts) if parts else ""
    except Exception:
        pass
    return ""


def _build_context_from_thread(
    thread: Sequence[BaseAgentEvent | BaseChatMessage],
    max_messages: int = 10,
) -> str:
    # build X_t: short dialogue string from the message thread for the VLM
    lines = []
    for msg in thread[-max_messages:]:
        source = _message_source_name(msg)
        text = _message_text(msg)
        if source and (text or getattr(msg, "content", None)):
            lines.append(f"{source}: {text.strip() or '(no text)'}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def _last_speaker_and_robot_count(
    thread: Sequence[BaseAgentEvent | BaseChatMessage],
    nao_names: List[str],
) -> Tuple[Optional[str], int]:
    # return (last_source, consecutive_robot_turn_count)
    # last_source: source of the most recent message with a source, or None if empty
    # consecutive_robot_turn_count: number of distinct robot *turns* at the end of the thread
    # (one agent can add multiple messages per turn e.g. tool calls + chat_message, so we count
    # runs of same source, not raw message count)

    if not thread:
        return None, 0
    last_source: Optional[str] = None
    for msg in reversed(thread):
        src = _message_source_name(msg)
        if src is not None:
            last_source = src
            break
    # Count consecutive robot *turns* (runs of same source) at the end
    prev_src: Optional[str] = None
    turn_count = 0
    for msg in reversed(thread):
        src = _message_source_name(msg)
        if src is not None and src in nao_names:
            if src != prev_src:
                turn_count += 1
                prev_src = src
        else:
            break
    return last_source, turn_count


def _first_valid_see_path(root: Path, nao_names: List[str]) -> Optional[Path]:
    # return the path to the first valid see.jpg among the given NAO session folders, or None
    for name in nao_names:
        path = get_session_dir(root, name) / "see.jpg"
        if _see_jpg_is_valid(path):
            return path
    return None


def _parse_score_from_vlm_reply(reply: str) -> float:
    # parse a number in [0, 1] from the VLM reply; return 0.5 on failure
    if not reply or not isinstance(reply, str):
        return 0.5
    # Look for a number optionally followed by .digits, possibly in "0.5" or "1.0" form
    match = re.search(r"0?\.\d+|1\.0?|1", reply.strip())
    if match:
        try:
            v = float(match.group(0))
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return 0.5


async def _vlm_score_response(
    image_path: Optional[Path],
    context_str: str,
    agent_name: str,
    d_personality: str,
) -> float:
    # call VLM with (image, context, personality) and return r_i in [0, 1]
    # if no valid image, return neutral 0.5
    if not image_path or not _see_jpg_is_valid(image_path):
        return 0.5
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        if not data:
            return 0.5
        image_b64 = base64.b64encode(data).decode("utf-8")
    except OSError:
        return 0.5

    prompt = (
        "Current conversation:\n"
        f"{context_str}\n\n"
        f"This robot's personality (d(p_i)): {d_personality or '(neutral)'}\n\n"
        "How appropriate is it for this robot to respond now? "
        "Reply with a single number between 0 and 1."
    )
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=VLM_SCORE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        reply = (response.choices[0].message.content or "").strip()
        return _parse_score_from_vlm_reply(reply)
    except Exception as e:
        print("[VLM turn manager] Score call failed for %s: %s" % (agent_name, e))
        return 0.5


async def multi_nao_chat(agents_list: List[dict], use_listen: bool = False):
    if not agents_list:
        raise ValueError("give at least one agent (from config/agent_config.json or --agent ROBOT)")
    root = Path(__file__).resolve().parent
    sock = _create_nao_socket(NAO_BASE_PORT)
    session_context = {"round": 0}
    nao_agents = []
    for i, agent_config in enumerate(agents_list):
        nao_agents.append(create_nao_agent(agent_config, root, i, sock, session_context=session_context))

    nao_names = [a.name for a in nao_agents]
    name_to_d_personality: Dict[str, str] = {a.name: (a.description or "") for a in nao_agents}

    def get_first_see_path() -> Optional[Path]:
        return _first_valid_see_path(root, nao_names)

    def _is_human_source(source: Optional[str]) -> bool:
        return source in (HUMAN_NAME, "user")

    async def vlm_selector(thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
        if not nao_names:
            return None
        last_source, consecutive_robot_count = _last_speaker_and_robot_count(thread, nao_names)
        # Hand back to Human after enough consecutive robot turns
        if last_source is not None and last_source in nao_names and consecutive_robot_count >= MAX_CONSECUTIVE_ROBOT_TURNS:
            return HUMAN_NAME
        # When last speaker is human (or empty/initial), or we're still under the limit: pick a robot
        if len(nao_names) == 1:
            return nao_names[0]
        X_t = _build_context_from_thread(thread)
        image_path = get_first_see_path()
        scores: List[Tuple[str, float]] = []
        for name in nao_names:
            d_pi = name_to_d_personality.get(name, "")
            r_i = await _vlm_score_response(image_path, X_t, name, d_pi)
            scores.append((name, r_i))
        # S_t = { i | r_i >= theta }; sort by score descending for top-K
        scores.sort(key=lambda x: -x[1])
        selected = [name for name, r in scores if r >= VLM_THRESHOLD]
        if not selected:
            selected = [scores[0][0]] if scores else [nao_names[0]]
        if VLM_TOP_K is not None and len(selected) > VLM_TOP_K:
            selected = selected[:VLM_TOP_K]
        # SelectorGroupChat expects a single speaker name (str), not a list; it validates "speaker not in participant_names" then returns [speaker].
        return selected[0] if selected else nao_names[0]

    def candidate_func(thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
        last_source, _ = _last_speaker_and_robot_count(thread, nao_names)
        if _is_human_source(last_source) or last_source is None:
            return list(nao_names)
        return list(nao_names) + [HUMAN_NAME]

    input_func = listen_for_human_input if use_listen else input
    human = UserProxyAgent(name="Human", input_func=input_func)

    participants = [human] + nao_agents
    num_rounds = 5
    max_messages = num_rounds * (1 + len(nao_agents))
    termination = MaxMessageTermination(max_messages)
    team = SelectorGroupChat(
        participants,
        model_client=model_client,
        selector_func=vlm_selector,
        candidate_func=candidate_func,
        termination_condition=termination,
    )

    initial_message = "Hello team!"
    result = await team.run(task=initial_message)

    # Session logging: from result.messages derive user_prompt / agent_response per NAO turn
    nao_names = {a.name for a in nao_agents}
    nao_by_name = {a.name: a for a in nao_agents}
    messages = getattr(result, "messages", []) or []
    last_text = ""
    for idx, msg in enumerate(messages):
        source = _message_source_name(msg)
        text = _message_text(msg)
        round_id = idx // (1 + len(nao_agents))
        if source and source in nao_names:
            nao = nao_by_name[source]
            nao.session_logger.log("user_prompt", {"message": last_text}, round_id=round_id)
            nao.session_logger.log("agent_response", {"text": text}, round_id=round_id)
        if text.strip():
            last_text = text
        if source:
            print(f"[{source}]: {text[:200]}{'...' if len(text) > 200 else ''}")
        # Allow time for robot to finish speaking/acting before next agent
        if source and source in nao_names:
            await asyncio.sleep(10)



# ====== Run ======
if __name__ == "__main__":
    agents_list, use_listen = _parse_agent_args()
    asyncio.run(multi_nao_chat(agents_list, use_listen))
