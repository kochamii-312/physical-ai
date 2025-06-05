import pybullet as p
import pybullet_data
import numpy as np

def pixel_to_world(u, v, depth=0.02, camera_config=None):
    fx, fy = 600, 600  # 仮の焦点距離
    cx, cy = 320, 240  # 画像中心点（640x480の場合）
    
    # カメラ基準の座標を計算（仮想）
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    # カメラは[0.8, 0, 0.5] から [0.4, 0, 0] を見ている
    # world座標に変換（視点から前方方向に合わせて調整）
    x_world = 0.4 - x_cam
    y_world = y_cam
    z_world = z_cam
    return [x_world, y_world, z_world]

def is_grasp_successful(panda_id, object_ids, threshold=0.1):
    # Pandaのエンドエフェクタ（リンク11）の位置
    ee_pos = p.getLinkState(panda_id, 11)[0]
    for obj_id in object_ids: 
        # オブジェクトの位置
        obj_pos = p.getBasePositionAndOrientation(obj_id)[0]
        # 距離で判定
        dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        if dist < threshold:
            return True, obj_id
    return False, None

def try_grasp_with_retries(panda_id, u, v, object_ids, max_attempts=10):
    for attempt in range(max_attempts):
        # YOLO中心座標 → 世界座標に変換
        pos = pixel_to_world(u, v, depth=0.02 + 0.005 * attempt)

        # 逆運動学でエンドエフェクタを移動
        joint_angles = p.calculateInverseKinematics(panda_id, 11, pos)
        for i in range(len(joint_angles)):
            p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_angles[i], force=500)
        for _ in range(100):  # 動作反映まで待つ
            p.stepSimulation()

        # グリッパーを閉じる
        for j in [9, 10]:
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=10)
        for _ in range(50):
            p.stepSimulation()

        # 判定
        success, obj_id = is_grasp_successful(panda_id, object_ids)
        if success:
            print(f"✔ Grasp succeeded on attempt {attempt + 1}")
            return True, obj_id
        else:
            print(f"✘ Grasp failed on attempt {attempt + 1}")

    print("❌ Grasp failed after all retries")
    return False, None
