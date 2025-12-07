import math
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from utils.utils_vector import Vector2
from utils.utils_bv import AABB2D, OBB2D
from utils.utils_algorithm import *



def draw_scene(objects: list[PhysicsObject2D], sdf_results: dict = None):
    """使用 Matplotlib 將場景中的所有物件畫出來"""
    fig, ax = plt.subplots(figsize=(10, 10))

    all_x = []
    all_y = []

    for obj in objects:
        # 收集所有點以自動調整視窗大小
        all_x.extend([v.x for v in obj.world_vertices])
        all_y.extend([v.y for v in obj.world_vertices])

        # 1.繪製物件的精確形狀 (凸體)
        poly_coords = [(v.x, v.y) for v in obj.world_vertices]
        ax.add_patch(Polygon(poly_coords, closed=True, facecolor='lightblue', edgecolor='blue', linewidth=2, label=f'{obj.name} Shape'))

        # 2.繪製 AABB (紅色虛線)
        aabb_width = obj.aabb.max.x - obj.aabb.min.x
        aabb_height = obj.aabb.max.y - obj.aabb.min.y
        ax.add_patch(Rectangle((obj.aabb.min.x, obj.aabb.min.y), aabb_width, aabb_height,
                               facecolor='none', edgecolor='red', linestyle='--', linewidth=1.5, label=f'{obj.name} AABB'))

        # 3.繪製 OBB (綠色實線)
        obb_center = obj.obb.center
        obb_half_extents = obj.obb.half_extents
        obb_angle_deg = math.degrees(obj.angle_rad)
        # Rectangle patch 旋轉的錨點在左下角，我們需要計算它
        corner_offset = Vector2(-obb_half_extents.x, -obb_half_extents.y).rotate(obj.angle_rad)
        obb_corner = obb_center + corner_offset
        ax.add_patch(Rectangle((obb_corner.x, obb_corner.y), obb_half_extents.x * 2, obb_half_extents.y * 2, angle=obb_angle_deg,
                               facecolor='none', edgecolor='green', linewidth=1.5, label=f'{obj.name} OBB', zorder=2))

    # 4.干涉區，確保它在最上層
    if sdf_results and sdf_results.get("interference_mask") is not None:
        grid = sdf_results.get("grid")
        interference_mask = sdf_results.get("interference_mask")
        if np.any(interference_mask): # 只在確實有干涉區時繪製
            
            # 繪製干涉區 (使用半透明紅色)
            interference_display = np.ma.masked_where(~interference_mask, np.ones_like(interference_mask))
            ax.imshow(interference_display, extent=(grid.bounds.min.x, grid.bounds.max.x, grid.bounds.min.y, grid.bounds.max.y),
                      origin='lower', cmap='Reds', alpha=0.7, interpolation='nearest', zorder=10)
            
            # 創建一個不可見的代理元素來代表干涉區
            ax.plot([], [], color='red', alpha=0.7, linewidth=10, label='Interference Area')

    # 根據所有點的範圍自動設定視窗邊界，並增加一些邊距
    if all_x and all_y:
        margin = 5  # 邊距大小
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 1. 定義不同形狀的本地頂點 (local vertices)
    
    # 矩形 (AGV)
    box_vertices = [
        Vector2(-1.5, -1), Vector2(1.5, -1),
        Vector2(1.5, 1), Vector2(-1.5, 1)
    ]
    
    # 三角形 (牆壁/障礙物)
    triangle_vertices = [
        Vector2(0, 1.5), Vector2(-1.5, -1), Vector2(1.5, -1)
    ]

    # 另一個不規則凸多邊形 
    ship_vertices = [
        Vector2(0, 2), Vector2(-1, -1), Vector2(1, -1)
    ]
    
    # 2. 創建多個物理物件
    agv = PhysicsObject2D(name="AGV", local_vertices=box_vertices)
    obstacle = PhysicsObject2D(name="Obstacle", local_vertices=triangle_vertices)
    package = PhysicsObject2D(name="Package", local_vertices=ship_vertices)

    # 3. 將物件放置在場景中的不同位置和角度
    agv.update_transform(Vector2(13.5, 9), math.radians(45))
    obstacle.update_transform(Vector2(15, 10), math.radians(-15))
    package.update_transform(Vector2(2, 2), math.radians(90))

    # 4. 將所有物件放入一個列表以便管理
    scene_objects = [agv, obstacle, package]

    # 5. 打印所有物件的初始狀態
    print("--- 場景中所有物件的初始狀態 ---")
    for obj in scene_objects:
        print(f"- 物件: {obj.name:<12} | AABB: {obj.aabb}")
    print("\n" + "="*50 + "\n")

    # 6. 粗檢測-使用 BVH 找出潛在的碰撞對
    print("--- 【粗檢測】使用 BVH ---")
    
    # 建立 BVH 樹
    bvh_root = build_bvh(scene_objects)
    print("BVH 樹已建立")

    # 查詢可能碰撞的物件對
    candidate_pairs = []
    query_bvh_pairs(bvh_root, candidate_pairs)

    if not candidate_pairs:
        print("-> BVH 檢測後，沒有發現任何可能碰撞的物件對")
    else:
        pair_names = [f"({p1.name}, {p2.name})" for p1, p2 in candidate_pairs]
        print(f"-> BVH 發現潛在碰撞對: {', '.join(pair_names)}")

    # 7. 細檢測-對潛在碰撞對使用 GJK 進行精確檢測
    print("\n" + "--- 【細檢測】使用 GJK ---")
    real_collisions = []
    sdf_results_to_draw = {}

    if not candidate_pairs:
        print("-> 沒有需要進行細檢測的物件對")
    else:
        for obj1, obj2 in candidate_pairs:
            print(f"正在對 ({obj1.name}, {obj2.name}) 進行 GJK/EPA 檢測...")
            simplex = gjk(obj1, obj2)
            if simplex:
                normal, depth = epa(simplex, obj1, obj2)
                real_collisions.append((obj1, obj2, (normal, depth)))
                print(f"--> 結果：確認碰撞！ 穿透深度: {depth:.2f}, 法線: {normal}")

                # 8. 干涉區計算 - 使用 Voxelization-SDF
                print("\n" + "--- 【干涉區計算】使用 Voxelization-SDF ---")
                
                # a. 創建一個足夠大的網格來包圍兩個碰撞物體
                combined_aabb_min_x = min(obj1.aabb.min.x, obj2.aabb.min.x)
                combined_aabb_min_y = min(obj1.aabb.min.y, obj2.aabb.min.y)
                combined_aabb_max_x = max(obj1.aabb.max.x, obj2.aabb.max.x)
                combined_aabb_max_y = max(obj1.aabb.max.y, obj2.aabb.max.y)
                bounds = AABB2D(Vector2(combined_aabb_min_x - 1, combined_aabb_min_y - 1), 
                                Vector2(combined_aabb_max_x + 1, combined_aabb_max_y + 1))
                resolution = 0.1 # 網格解析度，值越小越精確，計算量越大
                
                # b. 為每個物體計算 SDF
                print(f"為 {obj1.name} 計算 SDF...")
                sdf1 = compute_sdf_for_object(obj1, SDFGrid(bounds, resolution))
                print(f"為 {obj2.name} 計算 SDF...")
                sdf2 = compute_sdf_for_object(obj2, SDFGrid(bounds, resolution))

                # c. 計算干涉區
                interference_mask = calculate_interference_from_sdfs(sdf1, sdf2)
                print(f"-> 計算完成！發現 {np.sum(interference_mask)} 個干涉單元")
                sdf_results_to_draw = {"grid": sdf1, "interference_mask": interference_mask}

            else:
                print(f"--> 結果：未碰撞 (偽陽性)")

    # 9. 將場景與干涉區繪製出來
    draw_scene(scene_objects, sdf_results_to_draw)
