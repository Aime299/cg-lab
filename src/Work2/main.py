import pygame
import numpy as np
import math
# ==========================
# 终端打印控制说明
# ==========================
print("==================================================")
print("3D 立方体旋转程序")
print("==================================================")
print("控制说明：")
print("  R       - 重置角度")
print("  Space   - 暂停/继续自动旋转")
print("  A/D     - 绕 Y 轴旋转")
print("  W/S     - 绕 X 轴旋转")
print("  Q/E     - 绕 Z 轴旋转")
print("  ESC     - 退出程序")
print("==================================================")

# ==========================
# 变换矩阵工具
# ==========================
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotate_x(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

def rotate_y(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

def rotate_z(angle):
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov)/2)
    return np.array([
        [f/aspect,0,0,0],
        [0,f,0,0],
        [0,0,(far+near)/(near-far),(2*far*near)/(near-far)],
        [0,0,-1,0]
    ], dtype=np.float32)

def look_at(eye, target, up):
    eye = np.array(eye)
    target = np.array(target)
    up = normalize(np.array(up))
    forward = normalize(eye-target)
    right = normalize(np.cross(up, forward))
    new_up = np.cross(forward, right)
    return np.array([
        [right[0],right[1],right[2],-np.dot(right,eye)],
        [new_up[0],new_up[1],new_up[2],-np.dot(new_up,eye)],
        [forward[0],forward[1],forward[2],-np.dot(forward,eye)],
        [0,0,0,1]
    ], dtype=np.float32)

# ==========================
# 创建立方体
# ==========================
def create_cube():
    vertices = [
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    return vertices, edges

# ==========================
# 主程序
# ==========================
pygame.init()
w, h = 800, 600
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("3D Cube 手动/自动旋转控制")
clock = pygame.time.Clock()

vertices, edges = create_cube()

# 角度
angle_x = 0
angle_y = 0
angle_z = 0

# 自动旋转
auto_rotate = True
rotate_speed = 0.5

# 相机
view = look_at([0,0,5], [0,0,0], [0,1,0])
proj = perspective(60, w/h, 0.1, 100)

# ==========================
# 主循环
# ==========================
running = True
while running:
    screen.fill((0,0,0))

    # 事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                angle_x = angle_y = angle_z = 0
            if event.key == pygame.K_SPACE:
                auto_rotate = not auto_rotate

    # 按键持续控制
    keys = pygame.key.get_pressed()
    step = 2

    if keys[pygame.K_w]: angle_x += step
    if keys[pygame.K_s]: angle_x -= step
    if keys[pygame.K_a]: angle_y -= step
    if keys[pygame.K_d]: angle_y += step
    if keys[pygame.K_q]: angle_z -= step
    if keys[pygame.K_e]: angle_z += step

    # 自动旋转
    if auto_rotate:
        angle_x += rotate_speed * 0.3
        angle_y += rotate_speed * 0.5
        angle_z += rotate_speed * 0.2

    # 模型矩阵 = 旋转组合
    model = rotate_z(angle_z) @ rotate_y(angle_y) @ rotate_x(angle_x)
    mvp = proj @ view @ model

    # 变换顶点
    points = []
    for v in vertices:
        vh = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
        clip = mvp @ vh
        
        # 透视除法
        if clip[3] != 0:
            x_ndc = clip[0] / clip[3]
            y_ndc = clip[1] / clip[3]
        else:
            x_ndc = clip[0]
            y_ndc = clip[1]

        # 转为屏幕坐标（普通float，避免numpy格式报错）
        x = float((x_ndc + 1) / 2 * w)
        y = float((1 - y_ndc) / 2 * h)
        points.append((x, y))

    # 绘制立方体线框
    for a, b in edges:
        start = points[a]
        end = points[b]
        pygame.draw.line(screen, (0, 255, 255), (start[0], start[1]), (end[0], end[1]), 2)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()