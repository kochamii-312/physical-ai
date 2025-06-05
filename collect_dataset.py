import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np
import os
import random

# 保存フォルダ
save_dir = "dataset/images/train"
os.makedirs(save_dir, exist_ok=True)

# PyBullet起動
p.connect(p.DIRECT)  # GUI使う場合は p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

colors = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1)
}

# 画像枚数
n_images = 200

for i in range(n_images):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # ランダム配置の色付きキューブ
    color_name, color_rgba = random.choice(list(colors.items()))
    pos = [random.uniform(0.4, 0.6), random.uniform(-0.15, 0.15), 0.02]
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=color_rgba)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
    p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    # 数フレーム安定化
    for _ in range(20):
        p.stepSimulation()

    # カメラ画像取得
    width, height, rgb, depth, seg = p.getCameraImage(
        width=640, 
        height=480,
        viewMatrix=p.computeViewMatrix(
        cameraEyePosition=[0.8, 0, 0.5],
        cameraTargetPosition=[0.5, 0, 0],
        cameraUpVector=[0, 0, 1]
        ),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=2
        )
    )

    # 画像保存
    rgb_np = np.reshape(rgb, (height, width, 4))[:, :, :3]
    filename = f"{save_dir}/{color_name}_{i}.jpg"
    cv2.imwrite(filename, rgb_np)

    print(f"Saved: {filename}")

p.disconnect()
