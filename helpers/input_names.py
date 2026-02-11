"""
Ensure input/<name>/ exists with personality.json, see.jpg, and sound.wav.
When a name is used for the first time, create the folder and default files.
"""
import json
import struct
from pathlib import Path

# Default personality (same structure as personality_module default)
DEFAULT_PERSONALITY = {
    "self": {
        "scale": {"min": 1, "max": 5},
        "traits": {
            "o": 3,
            "c": 3,
            "e": 3,
            "a": 3,
            "n": 3,
        },
    }
}


def _safe_folder_name(name: str) -> str:
    """Sanitize name for use as a directory name."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._") or "unnamed"


def _write_placeholder_image(path: Path) -> None:
    """Write a minimal 1x1 JPEG so the file exists (will be overwritten by camera)."""
    # Minimal JPEG (1x1 red pixel)
    minimal_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c"
        b" $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.7\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
        b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00"
        b"\xf5\xc3\xff\xd9"
    )
    path.write_bytes(minimal_jpeg)


def _write_placeholder_wav(path: Path) -> None:
    """Write a minimal valid silent WAV (1 second, 8 kHz, 16-bit mono)."""
    sample_rate = 8000
    duration_sec = 1
    n_samples = sample_rate * duration_sec
    n_bytes = n_samples * 2  # 16-bit
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_bytes))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", n_bytes))
        f.write(b"\x00" * n_bytes)


def get_input_dir(root: Path, name: str) -> Path:
    """Return the input folder path for this name (input/<safe_name>/)."""
    safe = _safe_folder_name(name)
    return root / "input" / safe


def ensure_input_folder(root: Path, name: str) -> Path:
    """
    If this name has never been used, create input/<name>/ with
    personality.json, see.jpg, and sound.wav. Return the folder path.
    """
    folder = get_input_dir(root, name)
    folder.mkdir(parents=True, exist_ok=True)

    personality_path = folder / "personality.json"
    if not personality_path.exists():
        with open(personality_path, "w") as f:
            json.dump(DEFAULT_PERSONALITY, f, indent=2)

    see_path = folder / "see.jpg"
    if not see_path.exists():
        _write_placeholder_image(see_path)

    sound_path = folder / "sound.wav"
    if not sound_path.exists():
        _write_placeholder_wav(sound_path)

    return folder
