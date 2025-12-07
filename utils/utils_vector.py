import math

class Vector2:
    """一個簡單的 2D 向量類，用於表示位置和方向"""
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def rotate(self, angle_rad: float) -> 'Vector2':
        """將向量繞原點旋轉指定角度（弧度）"""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        new_x = self.x * cos_a - self.y * sin_a
        new_y = self.x * sin_a + self.y * cos_a
        return Vector2(new_x, new_y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def dot(self, other: 'Vector2') -> float:
        return self.x * other.x + self.y * other.y

    def magnitude_sq(self) -> float:
        return self.x**2 + self.y**2

    def magnitude(self) -> float:
        return math.sqrt(self.magnitude_sq())

    def normalized(self) -> 'Vector2':
        mag = self.magnitude()
        return self if mag == 0 else self / mag

    def perpendicular(self) -> 'Vector2':
        return Vector2(-self.y, self.x)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __truediv__(self, scalar: float):
        return Vector2(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Vector2({self.x:.2f}, {self.y:.2f})"