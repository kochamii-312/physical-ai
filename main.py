import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from detect import class_id_to_name
from grasp import pixel_to_world, try_grasp_with_retries
from sort import sort_object_by_color

# ===== 初期設定（pybullet GUI起動） =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 地面とPandaアーム読み込み
p.loadURDF("plane.urdf")
panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# 立方体オブジェクトを複数配置
object_ids = []
colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
positions = [[0.5, -0.1, 0.02], [0.5, 0.0, 0.02], [0.5, 0.1, 0.02]]
for color, pos in zip(colors, positions):
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=color)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
    cube_id = p.createMultiBody(baseMass=0.1,
                      baseCollisionShapeIndex=collision_shape,
                      baseVisualShapeIndex=visual_shape,
                      basePosition=pos)
    object_ids.append(cube_id)



# ===== カメラ画像取得関数 =====
def get_camera_image():
    width, height, rgb, _, _ = p.getCameraImage(
        640, 480,
        viewMatrix=p.computeViewMatrix([0.8, 0, 0.5], [0.4, 0, 0], [0, 0, 1]),
        projectionMatrix=p.computeProjectionMatrixFOV(60, 1.0, 0.01, 2)
    )
    return np.reshape(rgb, (480, 640, 4))[:, :, :3]



# ===== 物体検出と分別処理 =====
model = YOLO("runs/detect/train/weights/best.pt")
processed_positions = []

for loop_id in range(10):  # 最大10ループまで実行
    # 1. カメラ画像取得・保存
    img = get_camera_image()
    img_path = f"panda_view_loop{loop_id}.png"
    cv2.imwrite(img_path, img)

    # 2. YOLO推論
    results = model(img_path)[0]
    if len(results.boxes) == 0:
        print("検出なし、終了")
        break

    # 3. 検出された複数物体の中心座標を取得
    candidates = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        color_name = class_id_to_name[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        world_pos = pixel_to_world(u, v)
        
        # すでに処理済みの近い位置ならスキップ
        if any(np.linalg.norm(np.array(world_pos) - np.array(prev)) < 0.04 for prev in processed_positions):
            continue
        
        candidates.append((u, v, color_name, cls, world_pos))

    if not candidates:
        print("未処理の物体は残っていません")
        break

    # 4. Pandaに最も近い物体を選ぶ
    ee_pos = p.getLinkState(panda_id, 11)[0]
    nearest = min(candidates, key=lambda c: np.linalg.norm(np.array(c[4]) - np.array(ee_pos)))
    u, v, color_name, class_id, world_pos = nearest

    # 5. grasp（成功するまでリトライ）
    success, picked_id = try_grasp_with_retries(panda_id, u, v, object_ids)
    if not success:
        print("把持失敗、次のループへ")
        continue

    # 6. 分別動作
    sort_object_by_color(panda_id, picked_id, color_name)
    
    # 7. 処理済み座標を記録
    processed_positions.append(world_pos)


p.disconnect()

