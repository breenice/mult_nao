import argparse
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Sequence, AsyncGenerator
import speech_recognition as sr

import zmq

from openai import OpenAI

from modules.memory.memory_agent import MemoryAgent
from modules.llm.input_names import ensure_session_folder, get_session_dir
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent, TextMessage
from autogen_agentchat.base import Response
from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

from modules.personality.personality_module import PersonalityEngine
from modules.llm.turn_manager import LLMTurnManager, _message_source_name, _message_text
from config.nao_config import ROBOT_IPS, NAO_BASE_PORT
from autogen_ext.models.openai import OpenAIChatCompletionClient
from modules.llm.SessionLogger import SessionLogger
from modules.llm.helpers import _send_nao_command_sync, _make_nao_tool_with_signature, _create_nao_socket
from modules.llm.helpers import NAO_TOOLS

_MODULES_DIR = Path(__file__).resolve().parent / "modules"
DEFAULT_BIG5_PATH = str(_MODULES_DIR / "personality" / "big5.json")

_TOOLS_JSON_PATH = Path(__file__).resolve().parent / "modules" / "actions" / "tools.json"
with open(_TOOLS_JSON_PATH, "r") as f:
    NAO_TOOLS = json.load(f)

_PROMPTS_JSON_PATH = Path(__file__).resolve().parent / "config" / "prompts.json"
with open(_PROMPTS_JSON_PATH, "r") as f:
    PROMPTS = json.load(f)


class NaoAgent(AssistantAgent):
    # AssistantAgent subclass for NAO robots

    def __init__(self, personality: PersonalityEngine, memory: MemoryAgent,
                 session_logger: "SessionLogger", **kwargs):
        super().__init__(**kwargs)
        self.personality = personality
        self.memory = memory
        self.session_logger = session_logger
        self.perception: str = ""
        self.memory_context: str = ""

    # override on_messages_stream to inject updated perception and memory context
    async def on_messages_stream(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        context_parts = []
        if self.perception:
            context_parts.append(f"[What you see right now]: {self.perception}")
        if self.memory_context:
            context_parts.append(f"[Memory context]: {self.memory_context}")
        if context_parts:
            injection = TextMessage(content="\n".join(context_parts), source="system")
            messages = [injection] + list(messages)
        async for item in super().on_messages_stream(messages, cancellation_token):
            yield item


# ====== Model Client ======
llm_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
)

openai_client = OpenAI()


# ====== NAO Agent Factory (autogen agents from config: robot name, ip, personality) ======
def create_nao_agent(agent_config: dict, root: Path, agent_index: int, socket, session_context: Optional[dict] = None, all_agent_names: Optional[List[str]] = None):
    # agent_config: name (physical robot name), ip, bigo_personality. All paths and identity use this name.
    # Persistent data: session/<robot_name>/ (personality.json, memory db, see.jpg, sound.wav, conversation.jsonl)
    robot_name = str(agent_config.get("name", "Agent")).strip()
    bigo = agent_config.get("bigo_personality")
    folder = ensure_session_folder(root, robot_name, bigo_personality=bigo)
    personality_path = str(folder / "personality.json")
    memory_path = str(folder / "memory")
    is_text = robot_name not in ROBOT_IPS

    session_logger = SessionLogger(root, robot_name)

    personality = PersonalityEngine(big5_path=DEFAULT_BIG5_PATH, personality_path=personality_path)
    memory = MemoryAgent(agent_name=robot_name, db_path=memory_path)
    personality_text = personality.to_prompt_text()

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
        description = fn_def.get("description", fn_name) or fn_name
        params_schema = fn_def.get("parameters") or {}
        callable_fn = _make_nao_tool_with_signature(robot_name, fn_name, params_schema, send_for_this_agent)
        agent_tools.append(FunctionTool(callable_fn, description=description, name=fn_name))

    tool_list_str = ", ".join(tool_names)
    teammates = ', '.join(n for n in all_agent_names if n != robot_name) if all_agent_names else 'teammates'

    _sys_msg = PROMPTS["nao_agent_system"].format(
        robot_name=robot_name,
        teammates=teammates,
        personality_text=personality_text,
        tool_list_str=tool_list_str,
    )

    agent = NaoAgent(
        personality=personality,
        memory=memory,
        session_logger=session_logger,
        name=robot_name,
        description=personality_text or "NAO robot.",
        model_client=llm_client,
        tools=agent_tools,
        system_message=_sys_msg,
        reflect_on_tool_use=True,
        model_client_stream=True,
    )
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
#
_recognizer = sr.Recognizer()
_recognizer.dynamic_energy_threshold = False
_recognizer.energy_threshold = 300  # fixed threshold; tune for your environment
_recognizer.pause_threshold = 0.5   # faster end-of-speech detection (default 0.8)
_listen_calibrated = False


def debug_logg(msg, data=None):
    parts = ["[DEBUG]", msg]
    if data:
        parts.append(str(data))
    print(" ".join(parts))


def listen_for_human_input(prompt: str) -> str:
    # record from microphone until user stops speaking, when --listen is set
    # transcribe with Google Speech Recognition
    global _listen_calibrated


    debug_logg("listen_entry", {"calibrated": _listen_calibrated, "energy_threshold": _recognizer.energy_threshold, "pause_threshold": _recognizer.pause_threshold, "dynamic": _recognizer.dynamic_energy_threshold})

    # one-time noise calibration on first call
    if not _listen_calibrated:
        print("[Listen] Calibrating for ambient noise (1s)...")
        try:
            with sr.Microphone() as source:
                _recognizer.adjust_for_ambient_noise(source, duration=1)
            debug_logg("calibration_done", {"energy_threshold_after": _recognizer.energy_threshold})
            print("[Listen] Calibrated. Energy threshold = %d" % _recognizer.energy_threshold)
        except OSError as e:
            debug_logg("calibration_failed", {"error": str(e)})
            print("[Listen] Calibration failed: %s (using default threshold)" % e)
        _listen_calibrated = True

    try:
        _mic_info = {"default_index": sr.Microphone.list_microphone_names()[:5]}
    except Exception as _me:
        _mic_info = {"error": str(_me)}
    debug_logg("mic_devices", _mic_info)

    for attempt in range(_LISTEN_MAX_EMPTY_RETRIES):
        print(prompt)
        print("Listening... (speak now; recording stops when you pause)")
        debug_logg("listen_attempt", {"attempt": attempt, "energy_threshold": _recognizer.energy_threshold, "pause_threshold": _recognizer.pause_threshold})
        try:
            with sr.Microphone() as source:
                audio = _recognizer.listen(
                    source,
                    timeout=_LISTEN_TIMEOUT,
                    phrase_time_limit=_LISTEN_PHRASE_TIME_LIMIT,
                )
            _audio_data = audio.get_raw_data()
            debug_logg("audio_captured", {"bytes": len(_audio_data), "sample_rate": audio.sample_rate, "sample_width": audio.sample_width, "duration_s": round(len(_audio_data) / (audio.sample_rate * audio.sample_width), 2)})
        except sr.WaitTimeoutError:
            debug_logg("wait_timeout", {"attempt": attempt})
            print("No speech detected. Try again.")
            continue
        except OSError as e:
            debug_logg("mic_os_error", {"attempt": attempt, "error": str(e)})
            print("[Listen] Microphone error: %s. Try again." % e)
            continue

        try:
            transcript = _recognizer.recognize_google(audio)
            if transcript and isinstance(transcript, str) and transcript.strip():
                debug_logg("transcribed_ok", {"transcript": transcript.strip()})
                print("[Listen] Transcribed: %s" % transcript.strip())
                return transcript.strip()
            debug_logg("empty_transcript", {"transcript_repr": repr(transcript)})
            print("No speech detected. Try again.")
        except sr.UnknownValueError:
            debug_logg("unknown_value", {"attempt": attempt, "audio_bytes": len(audio.get_raw_data())})
            print("Sorry, I did not understand that. Try again.")
            continue
        except sr.RequestError as e:
            debug_logg("request_error", {"error": str(e)})
            print("Speech Recognition service is not available. Try again.")
            continue

    print("[Listen] Max retries reached. Falling back to keyboard input.")
    return input(prompt)


async def multi_nao_chat(agents_list: List[dict], use_listen: bool = False):
    if not agents_list:
        raise ValueError("give at least one agent (from config/agent_config.json or --agent ROBOT)")
    root = Path(__file__).resolve().parent
    sock = _create_nao_socket(NAO_BASE_PORT)
    session_context = {"round": 0}
    all_agent_names = [str(ac.get("name", "Agent")).strip() for ac in agents_list]
    nao_agents = []
    for i, agent_config in enumerate(agents_list):
        nao_agents.append(create_nao_agent(agent_config, root, i, sock, session_context=session_context, all_agent_names=all_agent_names))

    nao_names = [a.name for a in nao_agents]
    name_to_d_personality: Dict[str, str] = {a.name: (a.description or "") for a in nao_agents}

    tm = LLMTurnManager(
        nao_agents=nao_agents,
        nao_names=nao_names,
        name_to_d_personality=name_to_d_personality,
        root=root,
        openai_client=openai_client,
        prompts=PROMPTS,
    )

    input_func = listen_for_human_input if use_listen else input
    human = UserProxyAgent(name="Human", input_func=input_func)

    participants = [human] + nao_agents
    termination = TextMentionTermination("goodbye") | TextMentionTermination("bye") | TextMentionTermination("exit")
    team = SelectorGroupChat(
        participants,
        model_client=llm_client,
        selector_func=tm.vlm_selector,
        candidate_func=tm.candidate_func,
        termination_condition=termination,
    )

    initial_message = "Hello team!"
    result = await team.run(task=initial_message)

    # Session logging: from result.messages derive user_prompt / agent_response per NAO turn
    nao_names_set = {a.name for a in nao_agents}
    nao_by_name = {a.name: a for a in nao_agents}
    messages = getattr(result, "messages", []) or []
    last_text = ""
    for idx, msg in enumerate(messages):
        source = _message_source_name(msg)
        text = _message_text(msg)
        round_id = idx // (1 + len(nao_agents))
        if source and source in nao_names_set:
            nao = nao_by_name[source]
            nao.session_logger.log("user_prompt", {"message": last_text}, round_id=round_id)
            nao.session_logger.log("agent_response", {"text": text}, round_id=round_id)
        if text.strip():
            last_text = text



if __name__ == "__main__":
    agents_list, use_listen = _parse_agent_args()
    asyncio.run(multi_nao_chat(agents_list, use_listen))
