from utils.utils_vector import Vector2


class AABB2D:
    """
    軸對齊邊界框 (Axis-Aligned Bounding Box)
    用途：主要用於【粗檢測 (Broad Phase)】，例如 BVH 或 Quadtree（2D的Octree）
    優點：相交測試極快
    """
    def __init__(self, min_point: Vector2, max_point: Vector2):
        self.min = min_point
        self.max = max_point

    def intersects(self, other: 'AABB2D') -> bool:
        """檢查此 AABB 是否與另一個 AABB 相交"""
        return (self.min.x <= other.max.x and self.max.x >= other.min.x) and \
               (self.min.y <= other.max.y and self.max.y >= other.min.y)

    @property
    def center(self) -> 'Vector2':
        """計算並返回 AABB 的中心點"""
        return (self.min + self.max) / 2.0

    def __repr__(self):
        return f"AABB2D(min={self.min}, max={self.max})"

class OBB2D:
    """
    定向邊界框 (Oriented Bounding Box)
    用途：可用於更精確的粗檢測或中間階段檢測
    優點：對於旋轉物體的包裹比 AABB 更緊密，能減少誤判
    """
    def __init__(self, center: Vector2, half_extents: Vector2, axis1: Vector2, axis2: Vector2):
        self.center = center          # 中心點
        self.half_extents = half_extents  # 沿著軸向的半長度
        self.axes = [axis1, axis2]    # 兩個互相垂直的單位向量軸

    def __repr__(self):
        return f"OBB2D(center={self.center}, half_extents={self.half_extents})"
