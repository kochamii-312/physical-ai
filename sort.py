import numpy as np
import pybullet as p
import pybullet_data
import math

def sort_object_by_color(panda_id, picked_id, color_name, drop_height=0.2):
    """
    掴んだ物体を色ごとに分別位置まで移動して投下。

    Parameters:
        panda_id: PandaのID
        picked_id: 把持した物体のID（掴んだあとも連動している）
        color_name: 色ラベル（例: "red", "green", "blue"）
        drop_height: 分別位置での高さ
    """

    # 色ごとの分別位置を定義（y軸方向にずらす）
    sort_targets = {
        "red":   [0.3, -0.2, drop_height],
        "green": [0.3,  0.0, drop_height],
        "blue":  [0.3,  0.2, drop_height]
    }

    # 移動先座標を取得
    target_pos = sort_targets.get(color_name, [0.3, 0.0, drop_height])
    target_orient = p.getQuaternionFromEuler([math.pi, 0, 0])

    # 逆運動学で分別位置に移動
    joint_angles = p.calculateInverseKinematics(panda_id, 11, target_pos, target_orient)
    for i in range(len(joint_angles)):
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_angles[i], force=500)

    for _ in range(100):
        p.stepSimulation()

    # グリッパーを開いて投下
    for j in [9, 10]:
        p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=10)

    for _ in range(50):
        p.stepSimulation()

    print(f"✔ {color_name} の物体を分別位置に投下しました。")
