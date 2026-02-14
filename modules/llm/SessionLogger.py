import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from modules.llm.input_names import get_session_dir

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