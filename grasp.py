import pybullet as p
import pybullet_data
import numpy as np
import time
from settings import HOME_JOINT_ANGLES, move_to_joint_position

def pixel_to_world(u, v, depth, view_matrix, projection_matrix):

    # 画面座標 → NDC（Normalized Device Coordinates）
    ndc_x = (u / 640) * 4 - 1
    ndc_y = 1 - (v / 480) * 4  # Y軸は上下反転

    # 逆射影のための4次元同次座標
    clip_coords = np.array([ndc_x, ndc_y, 2 * depth - 1, 1.0])

    # 逆プロジェクションマトリクス
    proj_inv = np.linalg.inv(np.array(projection_matrix).reshape(4, 4))
    view_inv = np.linalg.inv(np.array(view_matrix).reshape(4, 4))

    eye_coords = proj_inv @ clip_coords
    eye_coords = eye_coords / eye_coords[3]

    world_coords = view_inv @ eye_coords
    world_coords = world_coords / world_coords[3]

    return world_coords[:3].tolist()

def bbox_to_world(x1, y1, x2, y2, depth, camera_position, camera_target):
    """
    バウンディングボックスの左上(x1, y1)と右下(x2, y2)から
    画像中心を求め、世界座標に変換する

    - depth: 仮定されるZ（物体高さ） [m]
    - cam_pos: カメラ位置 [X, Y, Z]
    - cam_target: カメラが見ている位置

    Returns: [x, y, z] in world coordinates
    """
    # 画像中心の u, v を計算
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # 焦点距離 fx, fy（仮定）
    fx, fy = 600, 600

    # カメラ座標系での位置（OpenGL準拠）
    x_cam = (u - 320) * depth / fx
    y_cam = (v - 240) * depth / fy
    z_cam = depth

    # カメラ→ワールド変換のためのview matrix
    view_matrix = p.computeViewMatrix(camera_position, camera_target, [0, 0, 1])
    view_matrix = np.array(view_matrix).reshape((4, 4), order='F')
    inv_view = np.linalg.inv(view_matrix)

    # カメラ座標 (x, y, z, 1) を world座標に変換
    cam_point = np.array([x_cam, y_cam, z_cam, 1.0])
    world_point = inv_view @ cam_point

    return [world_point[0]-0.3, world_point[1], 0.0]

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

def try_grasp_with_retries(panda_id, detected_world_pos, object_ids, max_attempts=10):
    for attempt in range(max_attempts):
        # グリッパーを開く
        for j in [9, 10]:
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=10)
        # 初期位置に戻る
        move_to_joint_position(panda_id, HOME_JOINT_ANGLES)
        # 逆運動学でエンドエフェクタを移動
        joint_angles = p.calculateInverseKinematics(panda_id, endEffectorLinkIndex=11, targetPosition=detected_world_pos) 
        for i in range(len(joint_angles)):
            p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_angles[i], force=100)
        for _ in range(100):  # 動作反映まで待つ
            p.stepSimulation()

        # グリッパーを閉じる
        for j in [9, 10]:
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=10)
        for _ in range(50):
            p.stepSimulation()
            # time.sleep(1. / 240.)  # PyBulletデフォルトのシミュレーション速度に合わせる

        # 判定
        success, obj_id = is_grasp_successful(panda_id, object_ids)
        if success:
            print(f"✔ Grasp succeeded on attempt {attempt + 1}")
            return True, obj_id
        else:
            print(f"✘ Grasp failed on attempt {attempt + 1}")

    print("❌ Grasp failed after all retries")
    return False, None
