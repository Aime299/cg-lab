import taichi as ti
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ti.init(arch=ti.cpu)

N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

spring_pairs = ti.Vector.field(2, dtype=int, shape=N * N * 4)
spring_lengths = ti.field(dtype=float, shape=N * N * 4)
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


def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()


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


init_cloth()

# 预运行让布料稳定
for _ in range(100):
    step_semi_implicit()

# 获取顶点位置用于绘图
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

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mass-Spring System - Cloth Simulation')

# 初始化绘图
lines = []
for _ in spring_edges:
    line, = ax.plot([], [], [], 'b-', linewidth=0.5)
    lines.append(line)
points = ax.scatter([], [], [], c='r', s=5)

def update(frame):
    for _ in range(5):
        step_semi_implicit()
    
    pos = get_positions()
    
    # 更新弹簧线
    for i, (a, b) in enumerate(spring_edges):
        lines[i].set_data([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]])
        lines[i].set_3d_properties([pos[a][2], pos[b][2]])
    
    # 更新质点
    points._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    
    ax.view_init(elev=30, azim=frame % 360)
    
    return lines + [points]

ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False)
plt.show()