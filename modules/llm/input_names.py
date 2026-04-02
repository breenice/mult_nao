# - session/<robot_name>/: persistent per-robot data (personality.json, memory db, see.jpg, sound.wav)
# - robot_name as unique identifier

from pathlib import Path
from modules.common.session_files import (
    safe_folder_name,
    ensure_session_artifacts,
    DEFAULT_PERSONALITY,
)

# Map bigo_personality keys (1-5) to trait keys (o, c, e, a, n)
BIGO_TO_TRAITS = {
    "openness": "o",
    "conscientiousness": "c",
    "extraversion": "e",
    "agreeableness": "a",
    "neuroticism": "n",
}


def personality_from_bigo(bigo_personality: dict) -> dict:
    # convert bigo_personality to personality.json 'self.traits' (1-5)
    # accepts either 0-1 scale (legacy) or 1-5 scale (current agent_config.json)
    # scale detection: if ANY value > 1, assume the whole dict is 1-5 scale
    traits = {"o": 3, "c": 3, "e": 3, "a": 3, "n": 3}
    if not bigo_personality:
        return {**DEFAULT_PERSONALITY, "self": {"scale": {"min": 1, "max": 5}, "traits": traits}}
    # Detect scale: if any trait value > 1, treat all as 1-5
    is_1_5_scale = any(
        float(bigo_personality.get(k, 0)) > 1
        for k in BIGO_TO_TRAITS
        if bigo_personality.get(k) is not None
    )
    for key, trait_key in BIGO_TO_TRAITS.items():
        val = bigo_personality.get(key)
        if val is not None:
            try:
                v = float(val)
                if is_1_5_scale:
                    # 1-5 scale: use directly (clamp to 1-5)
                    traits[trait_key] = max(1, min(5, round(v)))
                else:
                    # 0-1 scale: convert to 1-5
                    traits[trait_key] = max(1, min(5, 1 + round(v * 4)))
            except (TypeError, ValueError):
                pass
    return {
        "self": {
            "scale": {"min": 1, "max": 5},
            "traits": traits,
        }
    }


def _safe_folder_name(name: str) -> str:
    # sanitize folder name
    return safe_folder_name(name)


def get_input_dir(root: Path, name: str) -> Path:
    # return the input folder path for this name (input/<safe_name>/)
    safe = _safe_folder_name(name)
    return root / "input" / safe


def get_session_dir(root: Path, robot_name: str) -> Path:
    # return the session folder path for this robot (session/<safe_robot_name>/)
    safe = _safe_folder_name(robot_name)
    return root / "session" / safe


def ensure_session_folder(
    root: Path, robot_name: str, bigo_personality: dict | None = None
) -> Path:
    # create session/<robot_name>/ with personality.json, see.jpg, sound.wav
    # memory db will use session/<robot_name>/memory. Used for persistent per-robot data.
    # if bigo_personality is provided (from agent_config), always regenerate so config changes take effect.
    
    folder = get_session_dir(root, robot_name)
    personality = personality_from_bigo(bigo_personality) if bigo_personality else None
    ensure_session_artifacts(
        str(folder),
        personality=personality,
        overwrite_personality=bool(bigo_personality),
    )
    return folder


def ensure_input_folder(root: Path, name: str, bigo_personality: dict | None = None) -> Path:
    # create input/<name>/ with personality.json, see.jpg, sound.wav. prefer ensure_session_folder(root, robot_name, ...) for per-robot data.
    folder = get_input_dir(root, name)
    personality = personality_from_bigo(bigo_personality) if bigo_personality else DEFAULT_PERSONALITY
    ensure_session_artifacts(str(folder), personality=personality, overwrite_personality=False)
    return folder
