"""VLM-based turn manager for multi-agentic conversations

Scores each agent's appropriateness to speak (0-1)using a vision-language model,then selects qualified agents in descending score order
"""
import asyncio
import base64
import re
from pathlib import Path
from typing import Optional, List, Tuple, Sequence, Dict

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from modules.llm.input_names import get_session_dir
from modules.llm.perception_vlm import perceive_image

# ====== Constants ======
HUMAN_NAME = "Human"
HUMAN_SOURCE_NAMES = ("Human", "user")
VLM_THRESHOLD = 0.5   # theta: agents with r_i >= this are in S_t
VLM_TOP_K: Optional[int] = 2   # optional: max number of speakers per turn (None = no limit)
VLM_SCORE_MODEL = "gpt-4o"   # vision-capable model for scoring


# message formatters
def _message_source_name(msg) -> Optional[str]:
    # return the source/agent name for a message if present
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


# thread helpers

def _build_context_from_thread(
    thread: Sequence[BaseAgentEvent | BaseChatMessage],
    max_messages: int = 10,
) -> str:
    # build X_t: short dialogue string from the message thread for the VLM scoring
    lines = []
    for msg in thread[-max_messages:]:
        source = _message_source_name(msg)
        text = _message_text(msg)
        if source and (text or getattr(msg, "content", None)):
            lines.append(f"{source}: {text.strip() or '(no text)'}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def _last_speaker_and_robot_count(thread: Sequence[BaseAgentEvent | BaseChatMessage], nao_names: List[str]) -> Tuple[Optional[str], int]:
    # return (last_source, consecutive_robot_turn_count)
    if not thread:
        return None, 0
    last_source: Optional[str] = None
    for msg in reversed(thread):
        src = _message_source_name(msg)
        if src is not None:
            last_source = src
            break
    # Count consecutive robot messages at the end (any robot)
    turn_count = 0
    for msg in reversed(thread):
        src = _message_source_name(msg)
        if src is not None and src in nao_names:
            turn_count += 1
        else:
            break
    return last_source, turn_count


# VLM scoring helpers

def _see_jpg_is_valid(see_path: Path) -> bool:
    try:
        if not see_path.exists() or not see_path.is_file():
            return False
        size = see_path.stat().st_size
        # Placeholder images are tiny ~200 bytes
        return size > 500
    except OSError:
        return False


def _parse_score_from_vlm_reply(reply: str) -> float:
    # parse a number in [0, 1] from the VLM reply; return 0.5 on failure
    if not reply or not isinstance(reply, str):
        return 0.5
    match = re.search(r"0?\.\d+|1\.0?|1", reply.strip())
    if match:
        try:
            v = float(match.group(0))
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return 0.5


def _vlm_score_response_sync(image_path: Optional[Path], context_str: str, agent_name: str, d_personality: str, openai_client, prompts: dict) -> float:
    # call VLM with (I_t, X_t, d(p_i)) and return r_i,t in [0, 1]
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

    prompt = prompts["vlm_score"].format(
        context_str=context_str,
        agent_name=agent_name,
        d_personality=d_personality or "(neutral)",
    )
    try:
        response = openai_client.chat.completions.create(
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


# ===== LLM TurnManager =====

class LLMTurnManager:
    # manages VLM-based turn selection for multi-agent conversations
    # for each agent i, computes r_i,t = f_VLM(I_t, X_t, d(p_i))
    # S_t = {i | r_i,t >= theta}. Optional top-K constraint applied.
    # Qualified agents speak sequentially in descending score order.

    def __init__(self, nao_agents: List["NaoAgent"], nao_names: List[str], name_to_d_personality: Dict[str, str], root: Path, openai_client, prompts: dict):
        self.nao_agents = nao_agents
        self.nao_names = nao_names
        self.name_to_d_personality = name_to_d_personality
        self.root = root
        self.openai_client = openai_client
        self.prompts = prompts
        self.current_round_qualified: List[str] = []
        self.round_state: Dict[str, int] = {"index": 0}

    def _is_human_source(self, source: Optional[str]) -> bool:
        return source in HUMAN_SOURCE_NAMES

    async def vlm_selector(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
        """Select the next speaker using VLM scoring."""
        if not self.nao_names:
            return None
        last_source, _ = _last_speaker_and_robot_count(thread, self.nao_names)

        if self._is_human_source(last_source) or last_source is None:
            context_str = _build_context_from_thread(thread) # new human turn: score all agents with VLM

            # parallelized perception for all agents
            perception_results = await asyncio.gather(*[
                asyncio.to_thread(perceive_image, self.root, name, context_str)
                for name in self.nao_names
            ])
            perception_context: Dict[str, str] = dict(zip(self.nao_names, perception_results))

            # Memory recall + VLM scoring in parallel
            n = len(self.nao_agents)
            all_results = await asyncio.gather(
                # memory recall for each agent (N tasks) with user query and per-agent perception context
                *[asyncio.to_thread(
                    agent.memory.run_once,
                    f"User query: {context_str}\nPerception context: {perception_context[agent.name]}")
                  for agent in self.nao_agents],
                # VLM scoring for each agent (N tasks)
                *[asyncio.to_thread(
                    _vlm_score_response_sync,
                    get_session_dir(self.root, name) / "see.jpg",
                    context_str, name,
                    self.name_to_d_personality[name],
                    self.openai_client,
                    self.prompts,
                  )
                  for name in self.nao_names],
            )
            memory_results = list(all_results[:n])
            scores = list(all_results[n:])  # second half = VLM scores

            # Store latest perception and memory context on each agent
            for agent, perception, mem_result in zip(self.nao_agents, perception_results, memory_results):
                agent.perception = perception or ""
                agent.memory_context = mem_result or ""

            name_score_pairs = list(zip(self.nao_names, scores))

            # Log scores
            scores_str = ", ".join("%s=%.2f" % (n, s) for n, s in name_score_pairs)
            print("[VLM TURN MANAGER] Scores: %s | theta=%.2f" % (scores_str, VLM_THRESHOLD))

            # Apply threshold and top-K
            qualified = [(n, s) for n, s in name_score_pairs if s >= VLM_THRESHOLD]
            qualified.sort(key=lambda x: x[1], reverse=True)
            if VLM_TOP_K is not None:
                qualified = qualified[:VLM_TOP_K]

            self.current_round_qualified.clear()
            self.current_round_qualified.extend([n for n, _ in qualified])
            self.round_state["index"] = 0

            qualified_str = ", ".join("%s(%.2f)" % (n, s) for n, s in qualified)
            print("[VLM TURN MANAGER] S_t=[%s]" % qualified_str)

            if not self.current_round_qualified:
                print("[VLM TURN MANAGER] No agents qualified (all below theta)")
                return HUMAN_NAME

            self.round_state["index"] = 1
            return self.current_round_qualified[0]

        # robot agent just spoke, inject what it said into its memory 
        last_robot_msg = _message_text(thread[-1]) if thread else ""
        last_robot_name = _message_source_name(thread[-1]) if thread else None
        if last_robot_name and last_robot_msg:
            for agent in self.nao_agents:
                if agent.name == last_robot_name:
                    mem_result = await asyncio.to_thread(
                        agent.memory.run_once,
                        f"{agent.name} said: {last_robot_msg}")
                    agent.memory_context = mem_result or ""
                    break

        if self.round_state["index"] < len(self.current_round_qualified):
            out = self.current_round_qualified[self.round_state["index"]]
            self.round_state["index"] += 1
            print("[VLM TURN MANAGER] Next speaker: %s (index %d/%d)" % (out, self.round_state["index"], len(self.current_round_qualified)))
            return out
        return HUMAN_NAME

    def candidate_func(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
        """Return candidate speakers for the current turn."""
        last_source, _ = _last_speaker_and_robot_count(thread, self.nao_names)
        if self._is_human_source(last_source) or last_source is None:
            return self.nao_names + [HUMAN_NAME]
        if self.round_state["index"] < len(self.current_round_qualified):
            return [self.current_round_qualified[self.round_state["index"]]]
        return [HUMAN_NAME]
