import json
import zmq
from typing import Optional

from pathlib import Path

TOOLS_PATH = Path(__file__).resolve().parent.parent / "actions" / "tools.json"
with open(TOOLS_PATH, "r") as f:
    NAO_TOOLS = json.load(f)

def _get_tool_schema(tool_name: str) -> dict:
    # return the function schema for a tool name from actions/tools.json, or None
    for t in NAO_TOOLS:
        if t.get("function", {}).get("name") == tool_name:
            return t["function"].get("parameters", {})
    return None



def _filter_args_by_schema(tool_name: str, args: dict) -> dict:
    # keep only args that appear in the tool's parameters in actions/tools.json
    schema = _get_tool_schema(tool_name)
    if not schema or "properties" not in schema:
        return args
    allowed = set(schema["properties"].keys())
    return {k: v for k, v in args.items() if k in allowed}


def _json_type_to_annotation(prop: dict):
    t = prop.get("type", "string")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    return str


def _make_nao_tool_with_signature(agent_name: str, tool_name: str, params_schema: dict, send_fn):
    # async tool with explicit params and type annotations
    # to send: send_fn(tool_name, kwargs)
    props = params_schema.get("properties") or {}
    required = set(params_schema.get("required") or [])
    if not props:
        async def _no_arg_tool() -> str:
            await send_fn(tool_name, {})
            return "ok"
        _no_arg_tool.__name__ = tool_name
        return _no_arg_tool

    annotations = {}
    for pname, pdef in props.items():
        annotations[pname] = _json_type_to_annotation(pdef if isinstance(pdef, dict) else {})

    param_order = sorted(props.keys(), key=lambda p: (p not in required, p))
    param_parts = []
    for pname in param_order:
        ann = annotations[pname]
        ann_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        if pname in required:
            param_parts.append(f"{pname}: {ann_name}")
        else:
            param_parts.append(f"{pname}: Optional[{ann_name}] = None")

    params_str = ", ".join(param_parts)
    body_lines = [
        "kwargs = {k: v for k, v in locals().items() if v is not None}",
        "await send_fn(tool_name, kwargs)",
        "return str(kwargs)",
    ]
    body = "\n".join("    " + line for line in body_lines)
    exec_globals = {"send_fn": send_fn, "tool_name": tool_name, "Optional": Optional}
    exec(
        f"async def _fn({params_str}):\n{body}",
        exec_globals,
        local := {},
    )
    fn = local["_fn"]
    fn.__name__ = tool_name
    fn.__annotations__ = {p: annotations[p] for p in props.keys()}
    return fn


# NAO socket (per-agent; send action commands to nao_client)
NAO_SOCKET_HOST = "localhost"
_nao_context = None


def _create_nao_socket(port: int):
    # create and connect a ZMQ REQ socket to host:port
    # returns socket or None on failure
    global _nao_context
    if _nao_context is None:
        _nao_context = zmq.Context()
    try:
        sock = _nao_context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 30000) # 30 sec speaking timeout
        sock.connect("tcp://%s:%s" % (NAO_SOCKET_HOST, port))
        return sock
    except Exception as e:
        print("[LLM] NAO socket not available for port %s (%s). Actions will be printed only." % (port, e))
        return None


def _send_nao_command_sync(tool: str, args: dict, socket, agent_index: int = 0) -> str:
    # blocking send to nao_client. Args filtered by tools.json schema
    # returns reply or '' if socket is None
    if socket is None:
        return ""
    args = _filter_args_by_schema(tool, args)
    cmd = {"agent": agent_index, "tool": tool, "args": args}
    socket.send_string(json.dumps(cmd))
    return socket.recv_string()