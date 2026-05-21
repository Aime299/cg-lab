import math
import numpy as np

def normalize(v):
    """向量归一化"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotate_x(angle):
    """绕X轴旋转矩阵（右手系）"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)

def rotate_y(angle):
    """绕Y轴旋转矩阵"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)

def rotate_z(angle):
    """绕Z轴旋转矩阵"""
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=np.float32)

def translate(x, y, z):
    """平移矩阵"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def perspective(fov, aspect, near, far):
    """透视投影矩阵
    fov: 垂直视场角（度）
    aspect: 宽高比
    near: 近裁剪面
    far: 远裁剪面
    """
    f = 1.0 / math.tan(math.radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def look_at(eye, target, up):
    """视图矩阵（相机变换）"""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = normalize(np.array(up, dtype=np.float32))

    forward = normalize(eye - target)
    right = normalize(np.cross(up, forward))
    new_up = np.cross(forward, right)

    view_matrix = np.array([
        [right[0], right[1], right[2], -np.dot(right, eye)],
        [new_up[0], new_up[1], new_up[2], -np.dot(new_up, eye)],
        [forward[0], forward[1], forward[2], -np.dot(forward, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return view_matrix

def lerp_matrix(a, b, t):
    """矩阵线性插值（用于姿态过渡）"""
    return a * (1 - t) + b * t