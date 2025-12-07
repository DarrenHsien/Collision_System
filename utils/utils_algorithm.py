from utils.utils_bv import AABB2D
from utils.utils_vector import Vector2
from utils.utils_physic import PhysicsObject2D
import numpy as np
import math


# ==============================================================================
# BVH
# ==============================================================================


class BVHNode:
    """BVH 樹中的一個節點"""
    def __init__(self, objects: list[PhysicsObject2D]):
        self.objects = objects
        self.left: 'BVHNode' = None
        self.right: 'BVHNode' = None
        self.aabb: AABB2D = self._calculate_node_aabb()

    def _calculate_node_aabb(self) -> AABB2D:
        """計算能包圍節點內所有物件的 AABB"""
        if not self.objects:
            return None
        
        min_x = min(obj.aabb.min.x for obj in self.objects)
        min_y = min(obj.aabb.min.y for obj in self.objects)
        max_x = max(obj.aabb.max.x for obj in self.objects)
        max_y = max(obj.aabb.max.y for obj in self.objects)
        return AABB2D(Vector2(min_x, min_y), Vector2(max_x, max_y))

def build_bvh(objects: list[PhysicsObject2D]) -> BVHNode:
    """遞迴地建構 BVH 樹"""
    if not objects:
        return None

    node = BVHNode(objects)

    # 如果物件數量小於等於一個，則為葉節點，停止遞迴
    if len(objects) <= 1:
        return node

    # 選擇最長的軸進行分割
    span_x = node.aabb.max.x - node.aabb.min.x
    span_y = node.aabb.max.y - node.aabb.min.y
    axis = 'x' if span_x > span_y else 'y'

    # 根據選擇的軸的中點進行排序和分割
    objects.sort(key=lambda obj: obj.aabb.center.x if axis == 'x' else obj.aabb.center.y)
    mid = len(objects) // 2
    
    node.left = build_bvh(objects[:mid])
    node.right = build_bvh(objects[mid:])
    
    return node

def query_bvh_pairs(node: BVHNode, pairs: list):
    """在 BVH 樹中查詢所有可能碰撞的物件對"""
    if not node or (not node.left and not node.right):
        return

    # 如果左右子樹的 AABB 相交，則它們內部的物件可能碰撞
    if node.left and node.right and node.left.aabb.intersects(node.right.aabb):
        # 遍歷左子樹的所有物件，與右子樹的所有物件進行比較
        for obj_left in node.left.objects:
            for obj_right in node.right.objects:
                if obj_left.aabb.intersects(obj_right.aabb):
                    pairs.append((obj_left, obj_right))
    
    # 遞迴查詢左右子樹
    query_bvh_pairs(node.left, pairs)
    query_bvh_pairs(node.right, pairs)

# ==============================================================================
# 細檢測演算法：GJK (Gilbert-Johnson-Keerthi) 
# ==============================================================================

def support(obj: PhysicsObject2D, direction: Vector2) -> Vector2:
    """GJK support 函式：找出在指定方向上最遠的頂點"""
    max_dot = -float('inf')
    farthest_vertex = None
    for v in obj.world_vertices:
        dot = v.dot(direction)
        if dot > max_dot:
            max_dot = dot
            farthest_vertex = v
    return farthest_vertex

def triple_product(a: Vector2, b: Vector2, c: Vector2) -> Vector2:
    """計算三重積 (a × b) × c，用於 2D GJK"""
    # 在 2D 中，a × b 是純量（z 分量）
    # (a × b) × c = b(a·c) - a(b·c)
    ac = a.dot(c)
    bc = b.dot(c)
    return Vector2(b.x * ac - a.x * bc, b.y * ac - a.y * bc)

def handle_simplex(simplex: list[Vector2], direction: Vector2) -> bool:
    """
    處理單純形並更新搜索方向
    返回 True 表示 GJK 應繼續搜索，返回 False 表示已找到包含原點的單純形（碰撞）
    """
    if len(simplex) == 2:
        # 線段情況：simplex = [A, B]
        a = simplex[1]  # 最新加入的點
        b = simplex[0]
        
        ab = b - a
        ao = Vector2(0, 0) - a  # 從 A 指向原點
        
        # 使用三重積計算垂直於 AB 且指向原點的方向
        ab_perp = triple_product(ab, ao, ab)
        
        if ab_perp.magnitude_sq() < 1e-10:
            # AB 通過原點，使用垂直方向
            ab_perp = Vector2(-ab.y, ab.x)
        
        direction.x, direction.y = ab_perp.x, ab_perp.y
        return True

    elif len(simplex) == 3:
        # 三角形情況：simplex = [A, B, C]
        a = simplex[2]  # 最新加入的點
        b = simplex[1]
        c = simplex[0]
        
        ab = b - a
        ac = c - a
        ao = Vector2(0, 0) - a
        
        # 計算垂直於各邊且指向外部的法線
        ab_perp = triple_product(ac, ab, ab)
        ac_perp = triple_product(ab, ac, ac)
        
        if ab_perp.dot(ao) > 0:
            # 原點在 AB 邊外側
            simplex.pop(0)  # 移除 C
            direction.x, direction.y = ab_perp.x, ab_perp.y
            return True
        
        if ac_perp.dot(ao) > 0:
            # 原點在 AC 邊外側
            simplex.pop(1)  # 移除 B
            direction.x, direction.y = ac_perp.x, ac_perp.y
            return True
        
        # 原點在三角形內部
        return False
    
    return True

def gjk(obj1: PhysicsObject2D, obj2: PhysicsObject2D, debug=False) -> list[Vector2]:
    """
    GJK 演算法 (2D) - 修正版
    如果相交，返回構成包圍原點的單純形的頂點列表；否則返回 None
    """
    def minkowski_support(d: Vector2) -> Vector2:
        return support(obj1, d) - support(obj2, Vector2(-d.x, -d.y))

    # 使用兩物體中心差作為初始方向
    direction = obj1.position - obj2.position
    if direction.magnitude_sq() < 1e-10:
        direction = Vector2(1, 0)
    else:
        direction = direction.normalized()
    
    # 獲取初始點
    simplex = [minkowski_support(direction)]
    direction = Vector2(-simplex[0].x, -simplex[0].y)
    
    if direction.magnitude_sq() < 1e-10:
        direction = Vector2(1, 0)

    max_iterations = 50
    for iteration in range(max_iterations):
        a = minkowski_support(direction)
        
        if debug:
            print(f"迭代 {iteration}: 方向={direction}, 新點={a}, dot={a.dot(direction)}")
        
        # 加入容差檢查
        if a.dot(direction) < -1e-6:
            if debug:
                print("找到分離軸，無碰撞")
            return None
        
        simplex.append(a)
        
        if not handle_simplex(simplex, direction):
            if debug:
                print(f"找到碰撞！單純形: {simplex}")
            return simplex
        
        if direction.magnitude_sq() < 1e-10:
            if debug:
                print("方向向量過小，判定為碰撞")
            return simplex
    
    if debug:
        print(f"達到迭代上限 ({max_iterations})，判定為無碰撞")
    return None

def epa(simplex: list[Vector2], obj1: PhysicsObject2D, obj2: PhysicsObject2D) -> tuple[Vector2, float]:
    """
    EPA 演算法 (2D)當 GJK 確認碰撞後，計算穿透深度和法線
    """
    # 閔可夫斯基差集的 support 函式
    def minkowski_support(d: Vector2) -> Vector2:
        return support(obj1, d) - support(obj2, Vector2(-d.x, -d.y))

    polytope = list(simplex)

    for _ in range(30): # 限制迭代次數
        # 1. 找到離原點最近的邊
        min_dist = float('inf')
        closest_edge_index = -1
        penetration_normal = None
        penetration_depth = 0.0

        for i in range(len(polytope)):
            p1 = polytope[i]
            p2 = polytope[(i + 1) % len(polytope)]
            edge = p2 - p1
            
            # 計算邊的法線，並確保它指向原點外部
            normal = edge.perpendicular().normalized()
            dist = p1.dot(normal)
            if dist < 0:
                dist *= -1
                normal = Vector2(-normal.x, -normal.y)

            if dist < min_dist:
                min_dist = dist
                closest_edge_index = i
                penetration_normal = normal
                penetration_depth = dist

        # 2. 在最近邊的法線方向上尋找新的 support 點
        p = minkowski_support(penetration_normal)
        
        # 3. 檢查新點是否能擴展多邊形
        d = p.dot(penetration_normal)
        if d - penetration_depth < 1e-4: # 容忍誤差
            # 新點沒有明顯超出邊界，我們找到了最小穿透向量
            return penetration_normal, penetration_depth
        else:
            # 將新點插入到多邊形中，擴展它
            polytope.insert(closest_edge_index + 1, p)
    
    # 如果迭代次數過多，返回一個近似值
    return penetration_normal, penetration_depth

# ==============================================================================
# 干涉區計算：Voxelization-SDF
# ==============================================================================

class SDFGrid:
    """用於計算和儲存 2D SDF 的網格"""
    def __init__(self, bounds: AABB2D, resolution: float):
        self.bounds = bounds
        self.resolution = resolution
        self.width = int(math.ceil((bounds.max.x - bounds.min.x) / resolution))
        self.height = int(math.ceil((bounds.max.y - bounds.min.y) / resolution))
        # 使用 numpy array 以方便進行數學運算
        self.data = np.full((self.height, self.width), float('inf'), dtype=np.float32)

    def world_to_grid(self, world_pos: Vector2) -> tuple[int, int]:
        """將世界座標轉換為網格索引"""
        x = int((world_pos.x - self.bounds.min.x) / self.resolution)
        y = int((world_pos.y - self.bounds.min.y) / self.resolution)
        # 確保索引在邊界內
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return x, y

    def get_grid_cell_center(self, x: int, y: int) -> Vector2:
        """獲取網格單元中心的世界座標"""
        world_x = self.bounds.min.x + (x + 0.5) * self.resolution
        world_y = self.bounds.min.y + (y + 0.5) * self.resolution
        return Vector2(world_x, world_y)

def point_in_polygon(point: Vector2, polygon_vertices: list[Vector2]) -> bool:
    """使用 Ray Casting 演算法判斷點是否在多邊形內部"""
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0].x, polygon_vertices[0].y
    for i in range(n + 1):
        p2x, p2y = polygon_vertices[i % n].x, polygon_vertices[i % n].y
        if point.y > min(p1y, p2y):
            if point.y <= max(p1y, p2y):
                if point.x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point.x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def compute_sdf_for_object(obj: PhysicsObject2D, grid: SDFGrid) -> SDFGrid:
    """為單一物件計算 SDF"""
    # 1. 體素化 (Voxelization): 找出所有在物體內部的網格單元
    inside_mask = np.zeros_like(grid.data, dtype=bool)
    for y in range(grid.height):
        for x in range(grid.width):
            cell_center = grid.get_grid_cell_center(x, y)
            if point_in_polygon(cell_center, obj.world_vertices):
                inside_mask[y, x] = True

    # 2. 計算到邊緣的距離
    # 這是一個簡化的距離變換，更高效的方法是使用 scipy.ndimage.distance_transform_edt
    from scipy.ndimage import distance_transform_edt

    # 計算從外部到最近內部點的距離
    dist_outside = distance_transform_edt(~inside_mask) * grid.resolution
    # 計算從內部到最近外部點的距離
    dist_inside = distance_transform_edt(inside_mask) * grid.resolution

    # 組合為有向距離場
    grid.data = dist_outside - dist_inside
    return grid

def calculate_interference_from_sdfs(sdf1: SDFGrid, sdf2: SDFGrid) -> np.ndarray:
    """根據兩個 SDF 計算干涉區域"""
    # 干涉區是兩個 SDF 值都為負的區域
    interference_mask = (sdf1.data <= 0) & (sdf2.data <= 0)
    return interference_mask
