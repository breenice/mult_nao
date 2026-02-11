# Run from project root: source initialize/set_llm_path.sh && python llm_server.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/.."
export OPENAI_API_KEY="your_openai_api_key_here"