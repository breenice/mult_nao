#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import json
import struct
import zmq
import cv2
import os
import sys
import threading
import numpy as np
import signal
from collections import defaultdict
from naoqi import ALProxy
import argparse
import time

# Project root on path so modules package is found
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from modules.actions.actions import actions
try:
    from config import nao_config
    ROBOT_IPS = nao_config.ROBOT_IPS
    _BASE_PORT = nao_config.NAO_BASE_PORT
    _PORT_MAX_OFFSET = getattr(nao_config, "NAO_PORT_MAX_OFFSET", 9)
except ImportError:
    print("Error importing nao_config")
    sys.exit(1) 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
ZMQ_PORT = 9559

timestep_buffer = defaultdict(list)

parser = argparse.ArgumentParser()
parser.add_argument("--connection", choices=["text", "speech"], default="speech",
                    help="text: dummy IP, receive/print commands only; speech: connect to physical NAO")
parser.add_argument("--mode", choices=["print", "execute"], default="execute")
parser.add_argument("--port", type=int, default=None, help="ZMQ port (default: try from %s)" % _BASE_PORT)
parser.add_argument("--robot", type=str, default="ANGEL")
parser.add_argument("--multi", action="store_true", help="run single-port server (one port, dispatch by agent index from config)")
parser.add_argument("--config", type=str, default=None,
                    help="Path to agent config JSON (for --multi or --slot). Default: project_root/config/agent_config.json")
parser.add_argument("--slot", type=int, default=None, metavar="N",
                    help="Single-port mode for slot N (0-based) from agents config; uses port BASE+N and that entry's robot")
parser.add_argument("--agent", action="append", metavar="ROBOT",
                    help="For --multi: robot name (repeatable, e.g. --agent ANGEL --agent SAM). Overrides --config when provided.")

# ----- Load agents from config: list of robot names (strings) only -----
def _load_agents_from_file(path):
    # load list of robot names from JSON. Config format: {"agents": [{name, ...}, ...]}. Legacy: [{"robot": ...}] uses robot only. Returns [] if missing/invalid.
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (ValueError, IOError):
        return []
    if isinstance(data, dict) and "agents" in data and isinstance(data["agents"], list):
        return [str(a["name"]).strip() for a in data["agents"] if isinstance(a, dict) and "name" in a]
    if isinstance(data, list):
        return [str(item["robot"]).strip() for item in data if isinstance(item, dict) and "robot" in item]
    return []


# ----- Session folder per robot: session/<robot_name>/ (personality, see.jpg, sound.wav, memory) -----
def _safe_folder_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._") or "unnamed"

def _ensure_session_folder(root, robot_name):
    # create session/<robot_name>/ with personality.json, see.jpg, sound.wav for persistent per-robot data.
    safe = _safe_folder_name(robot_name)
    folder = os.path.join(root, "session", safe)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    personality_path = os.path.join(folder, "personality.json")
    if not os.path.isfile(personality_path):
        default = {"self": {"scale": {"min": 1, "max": 5}, "traits": {"o": 3, "c": 3, "e": 3, "a": 3, "n": 3}}}
        with open(personality_path, "w") as f:
            json.dump(default, f, indent=2)
    see_path = os.path.join(folder, "see.jpg")
    if not os.path.isfile(see_path):
        minimal_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c"
            b" $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.7\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
            b"\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00"
            b"\xf5\xc3\xff\xd9"
        )
        with open(see_path, "wb") as f:
            f.write(minimal_jpeg)
    sound_path = os.path.join(folder, "sound.wav")
    if not os.path.isfile(sound_path):
        sample_rate = 8000
        n_samples = sample_rate * 1
        n_bytes = n_samples * 2
        with open(sound_path, "wb") as f:
            f.write("RIFF")
            f.write(struct.pack("<I", 36 + n_bytes))
            f.write("WAVE")
            f.write("fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
            f.write("data")
            f.write(struct.pack("<I", n_bytes))
            f.write(b"\x00" * n_bytes)
    return folder


def run_multi_port(args, agent_list=None):
    # single port (NAO_BASE_PORT); receive messages with agent index, dispatch by agent id.
    if agent_list is None:
        agent_list = list(args.agent) if getattr(args, "agent", None) else []
    if not agent_list:
        raise SystemExit("Multi mode requires agents from --config or at least one --agent ROBOT (e.g. --multi --agent ANGEL --agent SAM)")
    base_port = getattr(nao_config, "NAO_BASE_PORT", 5555)
    port_max_offset = getattr(nao_config, "NAO_PORT_MAX_OFFSET", 9)
    for robot_name in agent_list:
        _ensure_session_folder(PROJECT_ROOT, robot_name)
    mode = getattr(args, "mode", "print")
    connection = getattr(args, "connection", "text")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    port = base_port
    for offset in range(0, port_max_offset + 1):
        try:
            port = base_port + offset
            sock.bind("tcp://*:" + str(port))
            break
        except zmq.ZMQError:
            if offset >= port_max_offset:
                raise
    print("[nao_client] Single-port mode: %s (agents: %s)" % (port, ", ".join(agent_list)))
    state = {"running": True}
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    def shutdown_multi(sig, frame):
        state["running"] = False

    signal.signal(signal.SIGINT, shutdown_multi)
    try:
        while state["running"]:
            socks = dict(poller.poll(500))
            if sock not in socks or socks[sock] != zmq.POLLIN:
                continue
            message = sock.recv_string().strip()
            try:
                msg = json.loads(message)
            except (ValueError, TypeError):
                sock.send_string("Error: invalid JSON")
                continue
            agent_index = msg.get("agent", 0)
            try:
                agent_index = int(agent_index)
            except (TypeError, ValueError):
                agent_index = 0
            if agent_index < 0 or agent_index >= len(agent_list):
                sock.send_string("Error: agent index %s out of range (0..%s)" % (agent_index, len(agent_list) - 1))
                continue
            robot_name = agent_list[agent_index]
            tool = str(msg.get("tool", ""))
            args = msg.get("args", {})
            if not tool:
                sock.send_string("Error: missing tool")
                continue
            print("[%s] [TEXT] tool=%s args=%s" % (robot_name, tool, args))
            if connection == "text" or robot_name not in ROBOT_IPS:
                sock.send_string("Message received by NAO")
                continue
            if mode == "execute":
                try:
                    robot_ip = ROBOT_IPS[robot_name]
                    actions.init(robot_ip, ZMQ_PORT)
                    arg_dict = dict(args)
                    for k, v in arg_dict.items():
                        if isinstance(v, unicode):
                            arg_dict[k] = v.encode("utf-8")
                    funct_call = getattr(actions, tool)
                    funct_call(**arg_dict)
                    sock.send_string("Action executed")
                except Exception as e:
                    sock.send_string("Error executing: " + str(e))
            else:
                sock.send_string("Message received by NAO")
    except KeyboardInterrupt:
        pass
    try:
        sock.close()
    except Exception:
        pass
    ctx.term()


args = parser.parse_args()

# default agents config path
if getattr(args, "config", None) is None:
    args.config = os.path.join(PROJECT_ROOT, "config", "agent_config.json")

if args.multi:
    agent_list = _load_agents_from_file(args.config)
    if getattr(args, "agent", None):
        agent_list = [str(r).strip() for r in args.agent]
    if not agent_list:
        sys.exit("Multi-port mode requires agents from --config file or at least one --agent ROBOT")
    run_multi_port(args, agent_list)
    sys.exit(0)

# --slot: single-port mode for one entry from agents config
if getattr(args, "slot", None) is not None:
    agent_list = _load_agents_from_file(args.config)
    if not agent_list or args.slot < 0 or args.slot >= len(agent_list):
        sys.exit("--slot %s requires a valid agents config with an entry at that index (0..%s)" % (args.slot, len(agent_list) - 1 if agent_list else 0))
    args.robot = agent_list[args.slot]
    base_port = getattr(nao_config, "NAO_BASE_PORT", 5555)
    args.port = base_port + args.slot

# single-port mode ========
connection = args.connection
mode = args.mode
ROBOT_NAME = args.robot

# assume text connection (no physical robot)
if connection == "text":
    ROBOT_IP = "127.0.0.1"
    _use_text_behavior = True
else:
    if ROBOT_NAME in ROBOT_IPS:
        ROBOT_IP = ROBOT_IPS[ROBOT_NAME]
        _use_text_behavior = False
    else:
        ROBOT_IP = "127.0.0.1"
        _use_text_behavior = True
        connection = "text"
        print("[nao_client] Robot '%s' not in ROBOT_IPS; assuming text connection." % ROBOT_NAME)

# Session folder keyed by physical robot name (session/<robot_name>/)
SESSION_FOLDER = _ensure_session_folder(PROJECT_ROOT, ROBOT_NAME)
print("[%s] Connection: %s" % (ROBOT_NAME, connection))
print("[%s] ROBOT_IP: %s" % (ROBOT_NAME, ROBOT_IP))
print("[%s] Session folder: %s" % (ROBOT_NAME, SESSION_FOLDER))

context = zmq.Context()
socket = context.socket(zmq.REP)
start_port = args.port if args.port is not None else _BASE_PORT
PORT = start_port
for offset in range(0, _PORT_MAX_OFFSET + 1):
    try:
        PORT = start_port + offset
        socket.bind("tcp://*:" + str(PORT))
        break
    except zmq.ZMQError:
        if offset >= _PORT_MAX_OFFSET:
            raise
print("[%s:%s] Bound to port: %s" % (ROBOT_NAME, PORT, str(PORT)))

running = True
vision_started = False

def shutdown(sig, frame):
    global running
    print("Shutting down...")
    running = False
    if vision_started:
        try:
            vision_service.unsubscribe(vision_client)
        except Exception:
            pass
    socket.close()
    context.term()
    sys.exit(0)



human_looking = False
    

def capture_image():
    global human_looking
    while running:
        nao_image = vision_service.getImageRemote(vision_client)
        width = nao_image[0] 
        height = nao_image[1] 
        raw_image = nao_image[6] 
            
        byte_img = np.frombuffer(raw_image, dtype=np.uint8) # into numpy array
        resized_img = byte_img.reshape((height, width, 3)) # flat array
        img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR) # rgb to bgr

        # Release the image AFTER using it
        vision_service.releaseImage(vision_client)

        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        human_looking = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                human_looking = True
                break

        tmp_file = os.path.join(SESSION_FOLDER, "see_tmp.jpg")
        final_file = os.path.join(SESSION_FOLDER, "see.jpg")
        cv2.imwrite(tmp_file, img)
        os.rename(tmp_file, final_file)

        time.sleep(0.05)  # reduce CPU load



def execute_timestep(timestep, calls):
    print("\nExecuting timestep "+str(timestep)+" ---")
    threads = []

    for call in calls:
        try:
            tool = str(call["tool"])
            args = call["args"]

            if mode == "print":
                print("[THREAD] actions.%s(%s)" % (
                    tool,
                    ", ".join("%s=%s" % (k, v) for k, v in args.iteritems())
                ))
                continue

            arg_dict = dict(
                (k.encode('utf-8') if isinstance(k, unicode) else k,
                 v.encode('utf-8') if isinstance(v, unicode) else v)
                for k, v in args.iteritems()
            )
            arg_dict.pop("timestep", None)

            funct_call = getattr(actions, tool)
            t = threading.Thread(target=funct_call, kwargs=arg_dict)
            t.daemon = True
            t.start()
            threads.append(t)

        except Exception as e:
            print("Error executing "+call['tool']+": "+str(e))

    for t in threads:
        t.join()

def socket_send():
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while running:
        socks = dict(poller.poll(500))
        if socket not in socks:
            continue

        message = socket.recv_string().strip()
        msg = json.loads(message)

        tool = str(msg["tool"])
        args = msg["args"]

        if connection == "text":
            print("[%s:%s] [TEXT] tool=%s args=%s" % (ROBOT_NAME, PORT, tool, args))
            socket.send_string("Message received by NAO")
        elif mode == "execute":
            try:
                for k, v in args.items():
                    if isinstance(v, unicode):
                        args[k] = v.encode("utf-8")
                funct_call = getattr(actions, tool)
                funct_call(**args)
                socket.send_string("Action executed")
            except Exception as e:
                socket.send_string("Error executing: "+str(e))
        else:
            timestep = msg.get("timestep", 0)
            timestep_buffer[timestep].append({"tool": tool, "args": args})
            socket.send_string("Message received by NAO")


if connection == "speech":
    actions.init(ROBOT_IP, ZMQ_PORT)
    if mode == "execute":
        vision_service = ALProxy("ALVideoDevice", ROBOT_IP, ZMQ_PORT)
        vision_client = vision_service.subscribe("python_camera_" + ROBOT_NAME, 2, 11, 30)
        camera_thread = threading.Thread(target=capture_image)
        camera_thread.daemon = True
        camera_thread.start()
        vision_started = True

socket_thread = threading.Thread(target=socket_send)
socket_thread.daemon = True
socket_thread.start()

# Keep main thread alive so Ctrl-C works
while running:
    try:
        while running:
            time.sleep(0.5)  # Keep main thread alive
    except KeyboardInterrupt:
        shutdown(None, None)

