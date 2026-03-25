# M2HRI — Multi-NAO HRI

Multi-robot human-robot interaction: an LLM server drives multiple NAO agents that speak, gesture, and act, while a NAO client receives commands and runs them on physical (or simulated) robots. Each agent has its own personality, optional long-term memory, and optional vision.

---

- **ma_server.py** — AutoGen `SelectorGroupChat` with a `UserProxyAgent` (human) and one `AssistantAgent` per robot. A turn manager decides which agents respond each round. Each agent has tools for NAO actions (speak, wave, nod, walk, etc.), optional long-term memory (ChromaDB + LangGraph memory agent; disable with `--no-memory`), and vision (reason about camera images).
- **ma_clients.py** — ZMQ REP server that receives JSON `{tool, args}` messages and dispatches them to physical NAOs via NaoQi, or prints them in text mode.

---

## Prerequisites

- **Two environments:**
  - **NAO**: Python 2.7, NaoQi paths set (for `ma_clients` / robot control).
  - **LLM**: Python 3.10+, OpenAI API key (for `ma_server`).
- **Two terminals** (or split terminal): one for the NAO client, one for the LLM server.


## Setup

1. **Conda environments**
  - NAO: Python 2.7 with NaoQi SDK.
  - LLM: Python 3.10+.

  - conda create -n nao python=2.7 anaconda
  - conda activate nao

  - conda create -n llm python=3.10 anaconda
  - conda activate llm

2. **Install dependencies**
  ```bash
  cd initialize
  pip install -r nao_requirements.txt   # NAO terminal
  pip install -r llm_requirements.txt   # LLM terminal
  ```

3. **Environment variables**
  - **NAO terminal:** export NaoQi paths as required for your setup.
  - **LLM terminal:** `export OPENAI_API_KEY=your_key`

---

## Configuration

### `config/agent_config.json`

Defines the agent roster — names, robot IPs, and Big Five personality traits (1-5 scale). Both `ma_server.py` and `ma_clients.py` read this file so port assignment stays in sync.

```json
{
  "agents": [
    {
      "name": "ANGEL",
      "ip": "192.168.0.101",
      "bigo_personality": {
        "openness": 4,
        "conscientiousness": 5,
        "extraversion": 2,
        "agreeableness": 5,
        "neuroticism": 1
      }
    },
    {
      "name": "SAM",
      "ip": "192.168.0.102",
      "bigo_personality": {
        "openness": 5,
        "conscientiousness": 5,
        "extraversion": 5,
        "agreeableness": 5,
        "neuroticism": 1
      }
    },
    {
      "name": "JOURNEY",
      "ip": "192.168.0.102",
      "bigo_personality": {
        "openness": 1,
        "conscientiousness": 1,
        "extraversion": 1,
        "agreeableness": 1,
        "neuroticism": 5
      }
    }
  ]
}
```

Required nao configuration fields:
-------------------------------------------------------------------------------
| `name` | Robot name — used as identity, folder name, and tool dispatch key. |
| `ip` | Robot IP address (NaoQi). |
| `bigo_personality` | Big Five traits (1-5). Seeds `session/<name>/personality.json` on first run. Personality shapes the agent's speaking style and behavior. |

### `config/turn_manager.json`

Option to toggle face attention criteria for turn-taking behavior:

```json
{
  "require_facing_when_directed": false
}
```
If `true`, a robot must have `facing.txt` set to `true` (from the camera thread) to respond to directed messages. Set to `false` for text-only mode. |

### `config/nao_config.py`

Shared config loaded by both server and client: `NAO_BASE_PORT` (default 5555), `ROBOT_IPS` map, and agent list.

---

## Per-Robot Session Data

All persistent data is stored under `session/<robot-name>/`:

| `personality.json` | Big Five traits (auto-generated from `agent_config.json`). |
| `memory/` | ChromaDB long-term memory (semantic, episodic, procedural). Created and used only when memory is enabled (default). Not used with `--no-memory`. |
| `see.jpg` | Latest camera frame (written by client camera thread). |
| `facing.txt` | Whether a human face is detected (`true`/`false`). |
| `conversation_*.jsonl` | Per-session conversation log. |

---

## Agent Capabilities

Each agent has the following tools available every turn:

**Memory (default; skipped with `--no-memory`):** The turn manager runs the memory agent each round to inject `[Memory context]` into the chat (recall before the agent responds, and update after the robot speaks). This is not separate `recall_memory` / `save_memory` tools on the NAO tool list—the pipeline is internal to the server when memory is enabled.

- With long-term memory **on** (default): prior context is retrieved from ChromaDB via the LangGraph memory agent.
- With **`--no-memory`**: no ChromaDB, no memory agent graph, and no `[Memory context]` injections—agents rely on personality, perception, and the current thread only.

**Action tools:**
- `speak(message)` — Text-to-speech.
- `wave(hand)` — Wave gesture (left/right).
- `nod()` — Head nod.
- `shake_hand()` — Handshake animation.
- `walk(x, y, theta)` — Move the robot.
- `stand()` / `sit()` / `rest()` — Posture changes.
- `point_at(x, y, z)` — Point in a direction.
- `move_head(x, z)` — Pan/tilt head.
- `track_face(enable)` — Enable/disable face tracking.
- `adjust_volume(level)` — Change speech volume.

**Vision tool:**
- `reason_with_vision(prompt)` — Analyze the camera image and describe what the robot sees. Chains into `speak()` automatically.

---

## Turn Management

Each round follows this pattern:

1. **Human speaks** (keyboard input or microphone with `--listen`).
2. **Turn manager** determines which agents respond:
   - **General message** (e.g. "hello everyone") — all agents respond in config order.
   - **Directed message** (e.g. "ANGEL, what do you think?") — only the named agent responds.
3. **Each qualified agent** responds: with memory enabled (default), the server has already run memory recall so `[Memory context]` is present; the agent then calls `speak`/other actions. With `--no-memory`, there is no memory step—only perception and personality inform the reply.
4. **Control returns to human** for the next round.

The conversation runs for 5 rounds by default (configurable via `num_rounds` in `ma_server.py`).

---

## Running

### 1. Start the NAO client (terminal 1)

**Multi-agent mode (one process, all agents):**
```bash
python ma_clients.py --multi --connection text
```
Loads agents from `config/agent_config.json`, binds a single ZMQ port, dispatches by agent index. Use `--connection text` for text-only (no physical robot).

**Single robot (one process per robot):**
```bash
python ma_clients.py --config config/agent_config.json --slot 0   # first agent
python ma_clients.py --config config/agent_config.json --slot 1   # second agent
```

### 2. Start the LLM server (terminal 2)

```bash
python ma_server.py
```

**No memory mode** (lighter setup, no ChromaDB / LangGraph memory on disk):

```bash
python ma_server.py --no-memory
```

**Options:**

| Flag | Description |
| --- | --- |
| `--agent ROBOT` | Override config with specific robots (repeatable). |
| `--config PATH` | Custom agent config JSON path. |
| `--listen` | Use microphone for human input (records until silence, transcribes with Google Speech Recognition). |
| `--no-memory` | Disable long-term memory: no ChromaDB, no LangGraph memory agent, no `[Memory context]` injection. |

### Debugging physical NAO robot connection (no LLM)

Use [`scripts/debug_nao.py`](scripts/debug_nao.py) to isolate NaoQi and the ZMQ path without LangChain or AutoGen. It runs: (1) NaoQi ping via Python 2, (2) direct `ALTextToSpeech.say` (“hello world”), (3) ZMQ `speak` JSON like `ma_server`, and optionally (4) an interactive speak loop. Requires `pip install pyzmq` in the same Python 3 environment you use to run the script.

Start **`ma_clients.py`** on the matching port before steps 3–4 (e.g. `--connection speech --mode execute` for real motion, or `--connection text` to verify only the reply string). Example:

```bash
python3 scripts/debug_nao.py --robot-ip <NAO_IP> --zmq-port 5555 --interactive
```

Use `--skip-naoqi` if you only have Python 3 (ZMQ tests only). Use `--no-include-agent` if your client is single-robot mode (JSON without `agent`).

---

## Summary/Quickstart
| Driver file | default | optional flags |
| --- | --- | --- |
| **ma_server** | `config/agent_config.json` | `--config`, `--agent ROBOT`, `--listen`, `--no-memory` |
| **ma_clients** | `config/agent_config.json` | `--config`, `--agent`, `--multi`, `--slot N`, `--connection text` |
