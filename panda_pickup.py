import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from PIL import Image

# pybullet GUI起動
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 地面とPandaアーム読み込み
p.loadURDF("plane.urdf")
panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# ごみ（カラフルな立方体）を複数配置
colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
positions = [[0.5, -0.1, 0.02], [0.5, 0.0, 0.02], [0.5, 0.1, 0.02]]
for color, pos in zip(colors, positions):
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=color)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
    p.createMultiBody(baseMass=0.1,
                      baseCollisionShapeIndex=collision_shape,
                      baseVisualShapeIndex=visual_shape,
                      basePosition=pos)

# ターゲット位置にアームを移動
target_pos = positions[2]
target_orient = p.getQuaternionFromEuler([math.pi, 0, 0])
joint_angles = p.calculateInverseKinematics(panda_id, 11, target_pos, target_orient)

for i in range(len(joint_angles)):
    p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_angles[i], force=500)

# グリッパー（指）を閉じる（掴む）
for finger_id in [9, 10]:
    p.setJointMotorControl2(panda_id, finger_id, p.POSITION_CONTROL, targetPosition=0.0, force=10)

# 動作の安定化
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240.0)

# カメラ画像を取得して保存
_, _, _, _, _, _, rgb_img, _, _ = p.getCameraImage(
    width=640,
    height=480,
    viewMatrix=p.computeViewMatrix([0.8, 0, 0.5], [0.4, 0, 0], [0, 0, 1]),
    projectionMatrix=p.computeProjectionMatrixFOV(60, 1.0, 0.01, 2),
    renderer=p.ER_BULLET_HARDWARE_OPENGL
)

rgb = np.reshape(rgb_img, (480, 640, 4))[:, :, :3]
cv2.imwrite("panda_camera_image.png", rgb)
cv2.imshow("Panda Camera", rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1, 1, 1],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 0, 1]
)
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=1.0,
    nearVal=0.1,
    farVal=100.0
)
width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    width=512,
    height=512,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix
)

img = Image.fromarray(rgbImg)
img.save("screenshot.png")

p.disconnect()

