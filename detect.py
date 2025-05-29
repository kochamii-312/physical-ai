import numpy as np
import pybullet as p
import pybullet_data
from grasp import pixel_to_world


class_id_to_name = {
    0: "red",
    1: "green",
    2: "blue"
}

def get_nearest_detection(results, panda_id):
    ee_pos = p.getLinkState(panda_id, 11)[0]

    nearest_obj = None
    min_dist = float("inf")

    for box in results.boxes:
        cls = int(box.cls[0].item())
        color_name = class_id_to_name.get(cls, "unknown")

        # 画像座標中心 (u,v)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # 画像座標 → 仮のロボット座標（zは固定）
        est_world = pixel_to_world(u, v, depth=0.02)
        dist = np.linalg.norm(np.array(est_world) - np.array(ee_pos))

        if dist < min_dist:
            min_dist = dist
            nearest_obj = (u, v, color_name, cls)

    return nearest_obj