from utils.utils_vector import Vector2
from utils.utils_bv import AABB2D, OBB2D
import math


class PhysicsObject2D:
    """
    一個 2D 物理物件，包含足以支撐完整碰撞檢測流程的資訊
    """
    def __init__(self, name: str, local_vertices: list[Vector2]):
        """
        初始化一個 2D 物理物件
        :param name: 物件名稱
        :param local_vertices: 定義物體形狀的本地空間頂點列表（圍繞原點(0,0)）
        """
        self.name = name
        
        # --- 核心幾何 (本地空間) ---
        # 這是物體的原始形狀定義，不會改變
        self.local_vertices = local_vertices
        
        # --- 變換資訊 (世界空間) ---
        self.position = Vector2(0, 0)  # 物件在世界中的位置
        self.angle_rad = 0.0           # 物件在世界中的旋轉角度（弧度）

        # --- 衍生數據 (用於碰撞偵測) ---
        self.world_vertices: list[Vector2] = []

        self.aabb: AABB2D = None

        self.obb: OBB2D = None

        # 首次創建時，進行一次完整的計算
        self.update_transform(self.position, self.angle_rad)

    def update_transform(self, new_position: Vector2, new_angle_rad: float):
        """
        更新物件的位置和旋轉，並重新計算所有衍生數據
        """
        self.position = new_position
        self.angle_rad = new_angle_rad

        # 1. 計算世界空間頂點 (world_vertices)
        # 這是 GJK/EPA 和 SDF 所需的精確幾何數據
        self.world_vertices = [v.rotate(self.angle_rad) + self.position for v in self.local_vertices]

        # 2. 計算 AABB
        # 這是 BVH/Quadtree 所需的粗略邊界框
        if not self.world_vertices:
            self.aabb = AABB2D(self.position, self.position)
        else:
            min_x = min(v.x for v in self.world_vertices)
            min_y = min(v.y for v in self.world_vertices)
            max_x = max(v.x for v in self.world_vertices)
            max_y = max(v.y for v in self.world_vertices)
            self.aabb = AABB2D(Vector2(min_x, min_y), Vector2(max_x, max_y))
            
        # 3. 計算 OBB (這是一個簡化的計算方法)
        # OBB 的軸就是物體旋轉後的本地座標軸
        axis1 = Vector2(1, 0).rotate(self.angle_rad)
        axis2 = Vector2(0, 1).rotate(self.angle_rad)
        
        # 計算本地頂點在本地軸上的投影範圍，以得到 half_extents
        min_local_x = min(v.x for v in self.local_vertices)
        max_local_x = max(v.x for v in self.local_vertices)
        min_local_y = min(v.y for v in self.local_vertices)
        max_local_y = max(v.y for v in self.local_vertices)
        
        half_extents = Vector2(
            (max_local_x - min_local_x) / 2.0,
            (max_local_y - min_local_y) / 2.0
        )

        # 計算本地 AABB 的中心點，這就是本地空間的偏移量
        local_center_offset = Vector2(
            (min_local_x + max_local_x) / 2.0,
            (min_local_y + max_local_y) / 2.0
        )
        
        # OBB 的世界中心點 = 物件世界位置 + 旋轉後的本地偏移量
        world_center = self.position + local_center_offset.rotate(self.angle_rad)
        self.obb = OBB2D(world_center, half_extents, axis1, axis2)

    def __repr__(self):
        return (f"PhysicsObject2D(name='{self.name}', position={self.position}, angle={math.degrees(self.angle_rad):.1f}°)\n"
                f"  - AABB: {self.aabb}\n"
                f"  - OBB: {self.obb}\n"
                f"  - World Vertices: {[str(v) for v in self.world_vertices]}")