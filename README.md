# M2HRI — Multi-NAO HRI

Multi-robot human–robot interaction: an LLM server drives NAO agents that speak and act, and a NAO client receives commands and runs them on physical (or simulated) robots.

---

## Prerequisites

- **Two environments:**
  - **NAO**: Python 2.7, NaoQi paths set (for `ma_clients` / robotx control).
  - **LLM**: Python 3, OpenAI API (for `ma_server`).
- **Two terminals** (or split terminal): one for the NAO client, one for the LLM server.

## Setup

1. **Conda environments**
   - NAO: Python 2.7.
   - LLM: Python 3.

2. **Install dependencies**
   ```bash
   cd initialize
   pip install -r nao_requirements.txt   # NAO terminal
   pip install -r llm_requirements.txt  # LLM terminal
   ```

3. **Environment variables**
   - **NAO terminal:** export NaoQi paths as required for your setup.
   - **LLM terminal:** `export OPENAI_API_KEY=your_key`

---

## Configuration: `config/agent_config.json`

Agent list, robot IPs, and Big Five personality are read from **`config/agent_config.json`** by default. Both `ma_server.py` and `ma_clients.py` use it so port assignment stays in sync: **port = `NAO_BASE_PORT` + index** (see `config/nao_config.py` for `NAO_BASE_PORT`).

**Example `config/agent_config.json`:**

```json
{
  "agents": [
    {
      "name": "ANGEL",
      "ip": "192.168.0.101",
      "bigo_personality": {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5
      }
    },
    {
      "name": "SAM",
      "ip": "192.168.0.102",
      "bigo_personality": {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5
      }
    }
  ]
}
```

- **`name`**: Physical robot name (used for personality, memory, logging, IP lookup, and all paths).
- **`ip`**: Robot IP address.
- **`bigo_personality`**: Big Five traits (0–1). Used to create or seed `session/<robot-name>/personality.json` when the session folder is first created.

**Paths:** All per-robot persistent data (memory db, `personality.json`, `see.jpg`, `sound.wav`) is stored under **`session/<robot-name>/`** (no separate display name or input folder).

---

## Running

### 1. Start the NAO client (terminal 1)

**Multi-port (one process, all agents):**
```bash
python ma_clients.py --multi
```
Loads agents from `config/agent_config.json`, binds one port per agent (`NAO_BASE_PORT + index`). Use `--connection text` for text-only (no real robot IP connection).

**Single robot (one process per robot):**
```bash
python ma_clients.py --config config/agent_config.json --slot 0   # first agent
python ma_clients.py --config config/agent_config.json --slot 1   # second agent
```

Override agents from file with `--agent ROBOT` (repeatable, e.g. `--agent ANGEL --agent SAM`). Custom config file: `--config /path/to/agent_config.json`.

### 2. Start the LLM server (terminal 2)

```bash
python ma_server.py
```

- Reads agents from **`config/agent_config.json`** by default.
- **Override agents:** `--agent ROBOT` (e.g. `--agent ANGEL --agent SAM`)
- **Custom config:** `--config /path/to/agent_config.json`
- **Listen mode (microphone for human turn):** `--listen` — records until you stop speaking, transcribes with Whisper, retries until non-empty.

After each NAO agent’s turn, the server waits 10 seconds so the robot can finish speaking/acting before the next turn.

---

## Summary

| Component    | Default config | Override / options |
| ----------- | -------------- | ------------------ |
| **ma_server**  | `config/agent_config.json` | `--config`, `--agent ROBOT`, `--listen` |
| **ma_clients**  | `config/agent_config.json` | `--config`, `--agent`, `--multi`, `--slot N`, `--connection text` |

One config file keeps ports, IPs, and agent order in sync between server and client.
