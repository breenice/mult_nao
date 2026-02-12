# M2HRI — Multi-NAO HRI

Multi-robot human-robot interaction: an LLM server drives multiple NAO agents that speak, gesture, and act, while a NAO client receives commands and runs them on physical (or simulated) robots. Each agent has its own personality, long-term memory, and optional vision.

---

- **ma_server.py** — AutoGen `SelectorGroupChat` with a `UserProxyAgent` (human) and one `AssistantAgent` per robot. A turn manager decides which agents respond each round. Each agent has tools for NAO actions (speak, wave, nod, walk, etc.), long-term memory (recall/save via ChromaDB), and vision (reason about camera images).
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
| `memory/` | ChromaDB long-term memory (semantic, episodic, procedural). Persists across conversations. |
| `see.jpg` | Latest camera frame (written by client camera thread). |
| `facing.txt` | Whether a human face is detected (`true`/`false`). |
| `conversation_*.jsonl` | Per-session conversation log. |

---

## Agent Capabilities

Each agent has the following tools available every turn:

**Memory tools:**
- `recall_memory(query)` — Search long-term memory before responding. Called first every turn.
- `save_memory(memory, memory_type)` — Store new facts after responding. Types: `semantic` (facts), `episodic` (preferences), `procedural` (instructions).

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
3. **Each qualified agent** runs: `recall_memory` -> `speak`/action -> `save_memory`.
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

**Options:**
| `--agent ROBOT` | Override config with specific robots (repeatable). |
| `--config PATH` | Custom agent config JSON path. |
| `--listen` | Use microphone for human input (records until silence, transcribes with Google Speech Recognition). |

---

## Summary/Quickstart
| Driver file  | default  | optional flags  |
| **ma_server** | `config/agent_config.json` | `--config`, `--agent ROBOT`, `--listen` |
| **ma_clients** | `config/agent_config.json` | `--config`, `--agent`, `--multi`, `--slot N`, `--connection text` |
