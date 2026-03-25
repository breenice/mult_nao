#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barebones NAO pipeline debugger (no LangChain / Autogen)

Still need to start ma_clients.py on the matching ZMQ port to verify ZMQ only).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

try:
    import zmq
except ImportError as e:
    print("Install pyzmq: pip install pyzmq", file=sys.stderr)
    raise SystemExit(1) from e

LOG = logging.getLogger("debug_nao")

# NaoQi default broker port (matches ma_clients.py ZMQ_PORT)
DEFAULT_NAOQI_PORT = 9559
DEFAULT_ZMQ_PORT = 5555


def _find_python2(explicit: Optional[str]) -> Optional[str]:
    import os

    if explicit:
        if os.path.isfile(explicit):
            return explicit
        w = shutil.which(explicit)
        return w
    for name in ("python2.7", "python2"):
        p = shutil.which(name)
        if p:
            return p
    return None


def run_naoqi_snippet(python2: str, robot_ip: str, naoqi_port: int, code: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    wrapper = (
        "import sys\n"
        "if len(sys.argv) < 3:\n"
        "    raise SystemExit('need ip port')\n"
        "ip, port = sys.argv[1], int(sys.argv[2])\n"
    ) + code
    return subprocess.run(
        [python2, "-c", wrapper, robot_ip, str(naoqi_port)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def step1_naoqi_ping(python2: str, robot_ip: str, naoqi_port: int) -> bool:
    LOG.info("========== STEP 1: NaoQi connectivity (ALTextToSpeech.getVolume) ==========")
    code = (
        "from naoqi import ALProxy\n"
        "tts = ALProxy('ALTextToSpeech', ip, port)\n"
        "v = tts.getVolume()\n"
        "print('NAOQI_PING_OK volume=' + str(v))\n"
    )
    try:
        cp = run_naoqi_snippet(python2, robot_ip, naoqi_port, code, timeout=30.0)
    except subprocess.TimeoutExpired as e:
        LOG.error("Step 1 timed out: %s", e)
        return False
    except FileNotFoundError as e:
        LOG.error("Python 2 not found: %s", e)
        return False
    LOG.debug("stdout: %s", cp.stdout.strip())
    LOG.debug("stderr: %s", cp.stderr.strip())
    if cp.returncode != 0:
        LOG.error("Step 1 failed rc=%s stderr=%s", cp.returncode, cp.stderr)
        return False
    LOG.info("Step 1 OK: %s", cp.stdout.strip())
    return True


def step2_naoqi_speak(python2: str, robot_ip: str, naoqi_port: int) -> bool:
    LOG.info("========== STEP 2: Direct NaoQi speak (hello world) ==========")
    code = (
        "from naoqi import ALProxy\n"
        "tts = ALProxy('ALTextToSpeech', ip, port)\n"
        'tts.say("hello world")\n'
        "print('NAOQI_SPEAK_OK')\n"
    )
    try:
        cp = run_naoqi_snippet(python2, robot_ip, naoqi_port, code, timeout=60.0)
    except subprocess.TimeoutExpired as e:
        LOG.error("Step 2 timed out: %s", e)
        return False
    LOG.debug("stdout: %s", cp.stdout.strip())
    LOG.debug("stderr: %s", cp.stderr.strip())
    if cp.returncode != 0:
        LOG.error("Step 2 failed rc=%s stderr=%s", cp.returncode, cp.stderr)
        return False
    LOG.info("Step 2 OK: %s", cp.stdout.strip())
    return True


def build_speak_payload(message: str, include_agent: bool, agent_index: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {"message": message}
    if include_agent:
        return {"agent": agent_index, "tool": "speak", "args": args} 
    return {"tool": "speak", "args": args}


def zmq_send_speak(host: str, port: int, recv_timeout_ms: int, message: str, include_agent: bool, agent_index: int) -> str:
    payload = build_speak_payload(message, include_agent, agent_index)
    body = json.dumps(payload, ensure_ascii=False)
    url = "tcp://%s:%s" % (host, port)
    LOG.debug("ZMQ connect %s send %s", url, body)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
    try:
        sock.connect(url)
        sock.send_string(body)
        reply = sock.recv_string()
        return reply
    finally:
        sock.close()
        ctx.term()


def step3_zmq_speak(host: str, port: int, recv_timeout_ms: int, include_agent: bool, agent_index: int) -> bool:
    LOG.info("========== STEP 3: ZMQ speak (hello world) ==========")
    try:
        reply = zmq_send_speak(
            host,
            port,
            recv_timeout_ms,
            "hello world",
            include_agent,
            agent_index,
        )
    except zmq.Again:
        LOG.exception("Step 3: recv timeout (is ma_clients listening on %s:%s?)", host, port)
        return False
    except Exception:
        LOG.exception("Step 3 failed")
        return False
    LOG.info("Step 3 reply: %s", reply)
    return True


def step4_interactive(host: str, port: int, recv_timeout_ms: int, include_agent: bool, agent_index: int) -> None:
    LOG.info("========== STEP 4: Interactive ZMQ speak (quit or EOF to exit) ==========")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
    url = "tcp://%s:%s" % (host, port)
    sock.connect(url)
    try:
        while True:
            try:
                line = input("speak> ").strip()
            except EOFError:
                LOG.info("EOF, exiting interactive loop")
                break
            if not line or line.lower() in ("quit", "exit", "q"):
                LOG.info("Done interactive loop")
                break
            payload = build_speak_payload(line, include_agent, agent_index)
            body = json.dumps(payload, ensure_ascii=False)
            LOG.info("Sending: %s", body)
            try:
                sock.send_string(body)
                reply = sock.recv_string()
                LOG.info("Reply: %s", reply)
            except zmq.Again:
                LOG.error("Recv timeout — check ma_clients on %s", url)
            except Exception:
                LOG.exception("Send/recv failed")
    finally:
        sock.close()
        ctx.term()


def main() -> None:
    p = argparse.ArgumentParser(description="Barebones NAO / ZMQ debug (see module docstring).")
    p.add_argument("--robot-ip", default="127.0.0.1", help="NAO robot IP for NaoQi (steps 1–2)")
    p.add_argument("--naoqi-port", type=int, default=DEFAULT_NAOQI_PORT, help="NaoQi broker port (default 9559)")
    p.add_argument("--zmq-host", default="127.0.0.1", help="ZMQ REP bind host (client listens here)")
    p.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT, help="ZMQ port (default 5555)")
    p.add_argument("--agent-index", type=int, default=0, help="agent field for multi-port client")
    p.add_argument(
        "--no-include-agent",
        dest="include_agent",
        action="store_false",
        default=True,
        help="Omit agent key (single-robot ma_clients). Default: include agent (ma_server style)",
    )
    p.add_argument("--recv-timeout-ms", type=int, default=30000, help="ZMQ recv timeout")
    p.add_argument("--python2", default=None, help="Path to python2 (else search PATH)")
    p.add_argument("--skip-naoqi", action="store_true", help="Skip steps 1–2 (NaoQi)")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="After step 3, run interactive speak loop (step 4)",
    )
    p.add_argument("--debug", action="store_true", help="DEBUG logging")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    py2 = None if args.skip_naoqi else _find_python2(args.python2)
    if not args.skip_naoqi and not py2:
        LOG.warning(
            "No python2 on PATH; skipping steps 1–2. Install NaoQi Python 2.7 or pass --python2 / --skip-naoqi"
        )

    ok = True
    if not args.skip_naoqi and py2:
        ok = step1_naoqi_ping(py2, args.robot_ip, args.naoqi_port) and ok
        ok = step2_naoqi_speak(py2, args.robot_ip, args.naoqi_port) and ok
    elif args.skip_naoqi:
        LOG.info("Skipping steps 1–2 (--skip-naoqi)")

    ok = step3_zmq_speak(
        args.zmq_host,
        args.zmq_port,
        args.recv_timeout_ms,
        args.include_agent,
        args.agent_index,
    ) and ok

    if args.interactive:
        step4_interactive(
            args.zmq_host,
            args.zmq_port,
            args.recv_timeout_ms,
            args.include_agent,
            args.agent_index,
        )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
