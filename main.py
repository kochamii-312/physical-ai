import pybullet as p
import pybullet_data
import time
import math
import cv2
import csv
import numpy as np
from PIL import Image
from ultralytics import YOLO
from detect import class_id_to_name
from grasp import pixel_to_world, bbox_to_world,try_grasp_with_retries
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
colors = [(0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, 1), (1, 0, 0, 1)]
positions = [[0.5, -0.4, 0.02], [0.5, -0.2, 0.02], [0.5, 0.0, 0.02], [0.5, 0.2, 0.02], [0.5, 0.4, 0.02]]
for color, pos in zip(colors, positions):
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02]*3, rgbaColor=color)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
    cube_id = p.createMultiBody(baseMass=0.1,
                      baseCollisionShapeIndex=collision_shape,
                      baseVisualShapeIndex=visual_shape,
                      basePosition=pos)
    object_ids.append(cube_id)



# ===== カメラ画像取得関数 =====
camera_position = [0.8, 0, 0.5] # カメラの位置
camera_target = [0.4, 0, 0] # カメラの注視点
camera_up_vector = [0, 0, 1] # カメラの上方向ベクトル
# カメラパラメータ
fov = 60 # 視野角
aspect = 1.0 # アスペクト比
nearVal = 0.01 # 近接クリッピング面
farVal = 2 # 遠方クリッピング面
def compute_view_matrix():
    return p.computeViewMatrix(
        cameraEyePosition=camera_position,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up_vector
    )
def compute_projection_matrix():
    return p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal
    )
def get_camera_image():
    width, height, rgb, _, _ = p.getCameraImage(
        640, 480,
        viewMatrix=compute_view_matrix(),
        projectionMatrix=compute_projection_matrix()
    )
    return np.reshape(rgb, (480, 640, 4))[:, :, :3]


# ===== 物体検出と分別処理 =====
model = YOLO("best.pt")
processed_positions = []
processed_object_ids = set()
# save_dir = "log/images"

# ログ初期化（ループ外で一度だけ）
log_file_path = "sorted_log.csv"
with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # ログヘッダーを修正 (object_id を追加)
    writer.writerow(["loop_id", "target_object_id", "picked_object_id", "u", "v", "color_name", "result_detail", "result"])

print(f"処理対象の初期オブジェクトIDリスト: {object_ids}")

for loop_id in range(10):  # 最大10ループまで実行
    print(f"\n--- ループ {loop_id} 開始 ---")
    print(f"現在処理済みのオブジェクトID: {processed_object_ids}")
    
    # 未処理のPybulletオブジェクトが残っているか確認
    remaining_pybullet_objects = [obj_id for obj_id in object_ids if obj_id not in processed_object_ids]
    if not remaining_pybullet_objects:
        print("すべての定義済みオブジェクトが処理されました。終了します。")
        break

    # 1. カメラ画像取得・保存
    img = get_camera_image()
    img_path = f"panda_view_loop{loop_id}.png"
    cv2.imwrite(img_path, img)

    # 2. YOLO推論
    results = model(img_path)[0]
    if len(results.boxes) == 0:
        print("YOLOが物体を検出しませんでした。")
        time.sleep(1)
        break

    # 3. 検出された複数物体の中心座標を取得
    candidates = []
    # 現在シーンに物理的に存在する未処理のPyBulletオブジェクトIDリスト
    current_unprocessed_pybullet_ids = [
        obj_id for obj_id in object_ids
        if obj_id not in processed_object_ids and p.getBasePositionAndOrientation(obj_id)[0][2] > -0.1
    ]

    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls not in class_id_to_name:
            continue
        color_name = class_id_to_name[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        try:
            detected_world_pos = bbox_to_world(x1, y1, x2, y2, depth=0.02, camera_position=camera_position, camera_target=camera_target)
        except Exception as e:
            print(f"bbox_to_world failed: {e}")
            continue

        print(f"検出: {color_name} @ pixel({u},{v}) -> world {np.round(detected_world_pos, 3)}")

        # 対象オブジェクト候補の中から最も近いPyBulletオブジェクトを探す
        best_match_obj_id = -1
        min_dist = float('inf')
        for obj_id in current_unprocessed_pybullet_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = np.linalg.norm(np.array(detected_world_pos[:2]) - np.array(obj_pos[:2]))
            if dist < min_dist:
                min_dist = dist
                best_match_obj_id = obj_id

        if best_match_obj_id != -1:
            if best_match_obj_id not in processed_object_ids:
                candidates.append((u, v, color_name, detected_world_pos, best_match_obj_id))
                print(f"  候補追加: YOLO検出({color_name}) -> PyBullet ID: {best_match_obj_id}")
            else:
                print(f"  PyBullet ID {best_match_obj_id} はすでに処理済み、スキップします")

    # 一番近い物体から順に処理
    candidates.sort(key=lambda x: np.linalg.norm(np.array(x[3]) - np.array(p.getLinkState(panda_id, 11)[0])))

    for (u, v, color_name, detected_world_pos, obj_id) in candidates:
        success, picked_id = try_grasp_with_retries(panda_id, detected_world_pos, [obj_id])
        if success:
            sort_object_by_color(panda_id, picked_id, color_name)
            processed_object_ids.add(picked_id)
            break  # 1ループ1物体のみ処理

    if not candidates:
        print("未処理の物体に対する有効な検出候補が見つかりませんでした。")
        # この場合、YOLOは何か検出したが、それが未処理のPyBulletオブジェクトに紐づかなかったケース
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([loop_id, "N/A", "N/A", "N/A", "N/A", "N/A", "no_valid_candidates_found", "fail"])
        time.sleep(1)
        continue

    # 4. Pandaに最も近い物体を選ぶ
    ee_link_state = p.getLinkState(panda_id, 11)
    ee_pos = ee_link_state[0]
    
    nearest_candidate_info = None
    min_dist_to_ee = float('inf')

    for cand_u, cand_v, cand_color, cand_detected_wp, cand_obj_id in candidates:
        # オブジェクトの実際の位置とエンドエフェクタの位置を比較
        obj_actual_pos, _ = p.getBasePositionAndOrientation(cand_obj_id)
        dist = np.linalg.norm(np.array(obj_actual_pos) - np.array(ee_pos))
        if dist < min_dist_to_ee:
            min_dist_to_ee = dist
            nearest_candidate_info = {
                "u": cand_u, "v": cand_v, "color_name": cand_color,
                "target_object_id": cand_obj_id # これが掴むべきPyBulletオブジェクトID
            }
    
    if nearest_candidate_info is None:
        print("選ぶべき最近傍候補が見つかりませんでした。(candidatesは空ではない)")
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([loop_id, "N/A", "N/A", "N/A", "N/A", "N/A", "no_nearest_candidate_found", "fail"])
        continue
    
    u_target = nearest_candidate_info["u"]
    v_target = nearest_candidate_info["v"]
    color_name_target = nearest_candidate_info["color_name"]
    object_id_to_try_grasp = nearest_candidate_info["target_object_id"]

    print(f"処理対象選択: PyBullet ID: {object_id_to_try_grasp}, 色: {color_name_target}, 検出ピクセル(u,v)=({u_target},{v_target}), EEからの距離: {min_dist_to_ee:.3f}")

    # 5. grasp（成功するまでリトライ）
    success_grasp = False
    actual_picked_is = -1
    grasp_detail_message = ""
    
    try:
        success_grasp, actual_picked_id = try_grasp_with_retries(panda_id, detected_world_pos, [object_id_to_try_grasp])
        
        if success_grasp:
            if actual_picked_id == object_id_to_try_grasp:
                grasp_detail_message = f"grasp_success_intended_obj_{actual_picked_id}"
                print(f"  把持成功: ID {actual_picked_id} (ターゲット通り)")
            else:
                # 意図しないものを掴んだ場合 (try_grasp_with_retries が対象を絞れなかった場合など)
                grasp_detail_message = f"grasp_success_unexpected_obj_{actual_picked_id}_expected_{object_id_to_try_grasp}"
                print(f"  把持成功: ID {actual_picked_id} (ターゲット {object_id_to_try_grasp} と異なる！)")
                # 掴んだ actual_picked_id が未処理なら処理済みとして扱う
                if actual_picked_id in object_ids and actual_picked_id not in processed_object_ids:
                     pass # 後続の処理で actual_picked_id を処理済みにする
                else: # 掴んだものが既に処理済みか、そもそもリストにない不正なIDの場合
                    success_grasp = False # 問題ありとして失敗扱いにする
                    grasp_detail_message = f"grasp_error_picked_invalid_or_processed_obj_{actual_picked_id}"
                    print(f"  エラー: 掴んだ物体 {actual_picked_id} は処理対象外または既に処理済み。")

        else:
            grasp_detail_message = f"grasp_fail_obj_{object_id_to_try_grasp}"
            print(f"  把持失敗: ID {object_id_to_try_grasp}")

    except Exception as e:
        print(f"try_grasp_with_retriesでエラー: {e}")
        success_grasp = False
        actual_picked_id = -1
        grasp_detail_message = f"grasp_exception_obj_{object_id_to_try_grasp}_{type(e).__name__}"

    log_u = u_target
    log_v = v_target
    log_color = color_name_target
    log_target_id = object_id_to_try_grasp
    log_picked_id = actual_picked_id if success_grasp else "N/A"
    log_result_status = "fail" # デフォルト

    if success_grasp:
        print(f"  物体 {actual_picked_id} (色: {color_name_target}) の分別処理を開始。")
        # 6. 分別動作
        success_sort = False
        sort_detail_message = ""
        try:
            sort_object_by_color(panda_id, actual_picked_id, color_name_target)
            success_sort = True
            sort_detail_message = f"sort_success_obj_{actual_picked_id}"
            print(f"    分別成功: ID {actual_picked_id}")
        except Exception as e:
            print(f"sort_object_by_colorでエラー: {e}")
            success_sort = False
            sort_detail_message = f"sort_exception_obj_{actual_picked_id}_{type(e).__name__}"
            print(f"    分別失敗: ID {actual_picked_id}")
        
        if success_sort:
            # 7. 処理済みIDを記録 (実際に掴んで分別まで成功した actual_picked_id を記録)
            processed_object_ids.add(actual_picked_id)
            log_result_status = "success"
            print(f"  物体 {actual_picked_id} を処理済みとして記録。処理済みリスト: {processed_object_ids}")
            final_detail_message = f"{grasp_detail_message}_{sort_detail_message}"
        else: # 分別に失敗した場合
            log_result_status = "fail"
            final_detail_message = f"{grasp_detail_message}_{sort_detail_message}"
            print(f"  物体 {actual_picked_id} は分別に失敗したため、未処理のままです。")
            # 安全のため、掴んでいるものを離す処理が必要な場合がある (sort_object_by_color内で処理されるか確認)
        print(f"[DEBUG] candidates: {candidates}")
    else: # 把持に失敗した場合
        log_result_status = "fail"
        final_detail_message = grasp_detail_message
        print(f"  物体 {object_id_to_try_grasp} は把持に失敗したため、未処理のままです。")


    # ログ記録
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([loop_id, log_target_id, log_picked_id, log_u, log_v, log_color, final_detail_message, log_result_status])
    
    print(f"--- ループ {loop_id} 終了 ---")
    time.sleep(0.5) # 次のループのための小休止

print("\n全ループ終了、または全オブジェクト処理完了。")
print(f"最終的に処理済みのオブジェクトID: {processed_object_ids}")
print(f"ログは {log_file_path} に保存されました。")

# ===== シミュレーション終了 =====
p.disconnect()
