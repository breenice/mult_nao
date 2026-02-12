import naoqi
from naoqi import ALProxy
import sys
import time
import math
import motion
import almath

def init(robot_ip, port):
    global motion_service, tts, posture_service, speech_volume, tracker

    print("ACTIONS connecting to:", robot_ip, port)

    motion_service = ALProxy("ALMotion", robot_ip, port)
    tts = ALProxy("ALTextToSpeech", robot_ip, port)
    posture_service = ALProxy("ALRobotPosture", robot_ip, port)
        
    speech_volume = 90.0
    tracker = ALProxy("ALTracker", robot_ip, port)

def toggle_breath(enable):
    motion_service.setBreathEnabled('Body', enable)
    time.sleep(10)

def speak(message):
    message_str = message.encode('utf-8')
    message_str = "\\style=didactic\\ \\vol="+str(speech_volume)+"\\ \\wait=5\\" + message_str
    tts.post.say(message_str)

def walk(x,y,theta):
    walk_config = [["MaxStepX", 0.04],       # Smaller steps
               ["MaxStepTheta", 0.2],    # Smaller turns
               ["MaxStepFrequency", 0.3],# Lower step frequency
               ["StepHeight", 0.01],     # Lower lift
               ["TorsoWx", 0.1],         # Optional: pitch balance
               ["TorsoWy", 0.1]]
    
    motion_service.wakeUp()
    motion_service.setStiffnesses("Body", 0.8)
    posture_service.goToPosture("StandInit", 0.9)

    motion_service.post.moveTo(float(x),float(y),float(theta), walk_config)
    
def stand():
    posture_service.post.goToPosture("Stand", 0.5)

def sit():
    posture_service.post.goToPosture("Sit", 0.5)

def rest():
    motion_service.rest()

def nod():
    motion_service.setStiffnesses("Head", 1.0)
    names      = ["HeadYaw", "HeadPitch"]
    angleLists = [-20.0*almath.TO_RAD, 30.0*almath.TO_RAD]
    timeLists  = 1.0
    motion_service.angleInterpolation(names[1], angleLists[0], timeLists, True)
    time.sleep(0.2)
    motion_service.angleInterpolation(names[1], angleLists[1], timeLists, True)
    time.sleep(0.2)
    motion_service.angleInterpolation(names[1], angleLists[0], timeLists, True)

def shake_hand():
    motion_service.setStiffnesses("Body", 1.0)

    # Extend right arm forward
    motion_service.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"],
                     [0.5, -0.3, 1.5, 0.5], 0.2)
    time.sleep(2)

    # Simulate handshake (small up-down motion)
    for _ in range(3):
        motion_service.setAngles("RShoulderPitch", 0.3, 0.3)
        time.sleep(0.3)
        motion_service.setAngles("RShoulderPitch", 0.5, 0.3)
        time.sleep(0.3)
    
    rest_arm()


def wave(hand='left'):
    motion_service.setStiffnesses("Body", 1.0)

    if hand == 'left':
        pitch_joint = "LShoulderPitch"
        roll_joint = "LShoulderRoll"
    else:
        pitch_joint = "RShoulderPitch"
        roll_joint = "RShoulderRoll"

    initial_pitch = motion_service.getAngles(pitch_joint, True)[0]
    initial_roll = motion_service.getAngles(roll_joint, True)[0]

    motion_service.setAngles([pitch_joint, roll_joint], [-1.0, 0.8], 0.3)
    time.sleep(0.5)

    for _ in range(2):
        motion_service.setAngles(roll_joint, -0.5, 0.5)
        time.sleep(0.4)
        motion_service.setAngles(roll_joint, 0.8, 0.3)
        time.sleep(0.4)

    motion_service.setAngles([pitch_joint, roll_joint], [initial_pitch, initial_roll], 0.3)

def adjust_volume(level):
    global speech_volume
    speech_volume = level

def track_face(enable=True):
    if enable:
        tracker.registerTarget("Face", 0.1)
        tracker.track("Face")
    else:
        tracker.stopTracker()
        tracker.unregisterAllTargets()

def point_at(x,y,z,open, side):
    if side == 'left':
        pitch_joint = "LShoulderPitch"
        roll_joint = "LShoulderRoll"
        yaw_joint = "LElbowYaw"
        hand = "LHand"
    else:
        pitch_joint = "RShoulderPitch"
        roll_joint = "RShoulderRoll"
        yaw_joint = "LElbowYaw"
        hand = "RHand"

    motion_service.setStiffnesses("Body", 1.0)
    motion_service.setStiffnesses(hand, 1.0)

    if open:
        motion_service.openHand(hand) # can't point with given hand operations
    else:
        motion_service.closeHand(hand)

    # Extend left arm in specifed direction
    motion_service.setAngles([roll_joint, pitch_joint, yaw_joint],
                            [x, y, z], 0.2)
    time.sleep(1)

def rest_arm():
    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]

    angles = [1.5, 0.3, -1.2, -0.5,
            1.5, -0.3, 1.2, 0.5]

    motion_service.angleInterpolation(names, angles, 1.0, True)

def rest_body():
    motion_service.rest()

def move_head(x,z, prompt=""):
    motion_service.setStiffnesses("Head", 1.0)

    names  = ["HeadYaw", "HeadPitch"] 
    angles = [-x, z]                  
    fractionMaxSpeed  = 0.2              # Speed of the movement (0.0 to 1.0)

    motion_service.setAngles(names, angles, fractionMaxSpeed)

def wait(seconds):
    time.sleep(seconds)

if __name__ == "__main__":
    # toggle_breath(True)
    walk(0.4,0,math.pi/2) #WARNING fix