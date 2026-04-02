import os
import struct
import json


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


def safe_folder_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._") or "unnamed"


def write_placeholder_image(path):
    # Minimal valid JPEG; overwritten by camera pipeline later.
    minimal_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c"
        b" $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.7\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
        b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00"
        b"\xf5\xc3\xff\xd9"
    )
    with open(path, "wb") as f:
        f.write(minimal_jpeg)


def write_placeholder_wav(path):
    sample_rate = 8000
    duration_sec = 1
    n_samples = sample_rate * duration_sec
    n_bytes = n_samples * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_bytes))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n_bytes))
        f.write(b"\x00" * n_bytes)


def ensure_session_artifacts(folder, personality=None, overwrite_personality=False):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    personality_path = os.path.join(folder, "personality.json")
    if overwrite_personality or not os.path.isfile(personality_path):
        payload = personality if personality is not None else DEFAULT_PERSONALITY
        with open(personality_path, "w") as f:
            json.dump(payload, f, indent=2)

    see_path = os.path.join(folder, "see.jpg")
    if not os.path.isfile(see_path):
        write_placeholder_image(see_path)

    sound_path = os.path.join(folder, "sound.wav")
    if not os.path.isfile(sound_path):
        write_placeholder_wav(sound_path)
