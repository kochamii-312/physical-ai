import pybullet as p
import time

# ===== Pandaアームの初期位置設定 =====
HOME_JOINT_ANGLES = [0.0, -0.3, 0.0, -2.1, 0.0, 2.0, 0.8]
import time

def move_to_joint_position(panda_id, joint_angles, steps=100):
    for i in range(7):
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_angles[i])
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)