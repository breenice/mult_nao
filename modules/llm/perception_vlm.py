import base64
import json
import openai
from pathlib import Path
from modules.llm.input_names import get_session_dir

VLM_MODEL = "gpt-4o-mini"
PROMPTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.json"
with open(PROMPTS_PATH, "r") as f:
    PROMPTS = json.load(f)

_vlm_client = openai.OpenAI()

def update_perception(agent_name, response_text, root):
    session_dir = get_session_dir(root, agent_name)
    session_dir.mkdir(parents=True, exist_ok=True)
    perception_file = session_dir / "perception.txt"

    # overwrite the file
    with open(perception_file, "w") as f:
        f.write(response_text + "\n")


def perceive_image(root: Path, agent_name: str, user_query: str) -> str:
    # reads image from session folder and sends base64-encoded to VLM
    image_path = get_session_dir(root, agent_name) / "see.jpg"
    if not image_path.exists() or image_path.stat().st_size < 500:
        return ""
    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except OSError:
        return ""
    prompt = PROMPTS["perception_vlm_system"].format(user_query=user_query)
    response = _vlm_client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]},
        ],
    )
    content = response.choices[0].message.content or ""
    update_perception(agent_name, content, root)
    return content
