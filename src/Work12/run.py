import taichi as ti
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ti.init(arch=ti.cpu)

# 物理参数
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_sh = 5000.0
k_b = 2000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 碰撞球体
sphere_center = ti.Vector([0.0, -0.5, 0.0])
sphere_radius = 0.5
enable_collision = True

# 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

spring_pairs = ti.Vector.field(2, dtype=int, shape=N * N * 12)
spring_lengths = ti.field(dtype=float, shape=N * N * 12)
num_springs = ti.field(dtype=int, shape=())


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        
        # 结构弹簧
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
        
        # 剪切弹簧
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
        
        if i < N - 1 and j > 0:
            idx_diag2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
        
        # 弯曲弹簧
        if i < N - 2:
            idx_skip = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip])
            spring_lengths[c] = (x[idx] - x[idx_skip]).norm()
        
        if j < N - 2:
            idx_skip2 = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip2])
            spring_lengths[c] = (x[idx] - x[idx_skip2]).norm()


def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    print(f"弹簧数量: {num_springs[None]}")


@ti.func
def handle_collision(pos, vel):
    new_pos = pos
    new_vel = vel
    if enable_collision:
        dir_to_center = pos - sphere_center
        dist = dir_to_center.norm()
        if dist < sphere_radius:
            normal = dir_to_center / dist
            new_pos = sphere_center + normal * sphere_radius
            new_vel = vel - 0.5 * vel.dot(normal) * normal
    return new_pos, new_vel


@ti.func
def compute_forces_on(pos, vel, force):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)


@ti.func
def clamp_velocity(vel, idx):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            new_x, new_v = handle_collision(x[i], v[i])
            x[i] = new_x
            v[i] = new_v


init_cloth()

# 预运行稳定
for _ in range(100):
    step_semi_implicit()

# 获取顶点位置
def get_positions():
    pos = []
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            pos.append([x[idx][0], x[idx][1], x[idx][2]])
    return np.array(pos)

# 获取弹簧连接
spring_edges = []
num = int(num_springs[None])
for i in range(num):
    a = int(spring_pairs[i][0])
    b = int(spring_pairs[i][1])
    spring_edges.append((a, b))

# 创建图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mass-Spring System - Structural + Shear + Bending + Collision')

# 绘制球体
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
sphere_x = 0 + sphere_radius * np.outer(np.cos(u), np.sin(v))
sphere_y = -0.5 + sphere_radius * np.outer(np.sin(u), np.sin(v))
sphere_z = 0 + sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(sphere_x, sphere_y, sphere_z, color='green', alpha=0.3)

# 初始化绘图
lines = []
for _ in spring_edges:
    line, = ax.plot([], [], [], 'b-', linewidth=0.5)
    lines.append(line)
points = ax.scatter([], [], [], c='r', s=5)

frame_count = 0

def update(frame):
    global frame_count
    for _ in range(5):
        step_semi_implicit()
    
    pos = get_positions()
    
    # 更新弹簧线
    for i, (a, b) in enumerate(spring_edges):
        lines[i].set_data([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]])
        lines[i].set_3d_properties([pos[a][2], pos[b][2]])
    
    # 更新质点
    points._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    
    # 旋转视角
    frame_count += 1
    ax.view_init(elev=30, azim=frame_count % 360)
    
    return lines + [points]

print("开始动画...")
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False)
plt.show()