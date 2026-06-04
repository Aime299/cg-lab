import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# 物理参数
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0      # 结构弹簧系数
k_sh = 5000.0      # 剪切弹簧系数
k_b = 2000.0       # 弯曲弹簧系数
k_d = 1.0          # 阻尼系数
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

# 弹簧数据
spring_pairs = ti.Vector.field(2, dtype=int, shape=N * N * 12)  # 更多弹簧
spring_lengths = ti.field(dtype=float, shape=N * N * 12)
num_springs = ti.field(dtype=int, shape=())

# 屏幕参数
WIDTH, HEIGHT = 800, 600


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_springs():
    """初始化三种弹簧：结构、剪切、弯曲"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        
        # 1. 结构弹簧 (Structural) - 相邻网格点
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
        
        # 2. 剪切弹簧 (Shear) - 对角线方向
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
        
        # 3. 弯曲弹簧 (Bending) - 间隔一个点的结构弹簧
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
    """球体碰撞处理"""
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
    """计算所有力（使用不同的弹簧系数）"""
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
            # 根据弹簧类型选择系数（简化：统一使用 k_s）
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
    """半隐式欧拉积分 + 碰撞处理"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            
            # 碰撞处理
            x[i], v[i] = handle_collision(x[i], v[i])


def main():
    init_cloth()

    gui = ti.GUI("Mass-Spring System - 付雅婷 (Shear+Bending+Collision)", (WIDTH, HEIGHT))
    paused = False
    show_springs = True

    print("=" * 60)
    print("质点弹簧模型 - 完整版 (结构弹簧 + 剪切弹簧 + 弯曲弹簧 + 球体碰撞)")
    print("=" * 60)
    print("  P - 暂停/恢复    R - 重置布料")
    print("  C - 切换碰撞开关    S - 切换弹簧显示")
    print("  ESC - 退出")
    print("=" * 60)

    # 预运行稳定
    for _ in range(100):
        step_semi_implicit()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'p':
                paused = not paused
                print("暂停" if paused else "继续")
            elif gui.event.key == 'r':
                init_cloth()
                print("布料已重置")
            elif gui.event.key == 'c':
                global enable_collision
                enable_collision = not enable_collision
                print(f"碰撞: {'开启' if enable_collision else '关闭'}")
            elif gui.event.key == 's':
                show_springs = not show_springs
                print(f"弹簧显示: {'开启' if show_springs else '关闭'}")

        if not paused:
            for _ in range(20):
                step_semi_implicit()

        gui.clear(0x111111)

        # 计算屏幕坐标
        points_2d = []
        for idx in range(N * N):
            px = x[idx][0]
            py = x[idx][1]
            pz = x[idx][2]
            screen_x = (px + 0.8) / 1.6 * WIDTH
            screen_y = (1.0 - (py + 0.2) / 1.8) * HEIGHT
            points_2d.append((screen_x, screen_y))

        # 绘制弹簧
        if show_springs:
            num = int(num_springs[None])
            for i in range(num):
                a = int(spring_pairs[i][0])
                b = int(spring_pairs[i][1])
                p1 = points_2d[a]
                p2 = points_2d[b]
                gui.line(p1, p2, color=0x4488AA, radius=1)

        # 绘制质点
        for idx in range(N * N):
            p = points_2d[idx]
            if 0 <= p[0] <= WIDTH and 0 <= p[1] <= HEIGHT:
                gui.circle(p, radius=2, color=0xFF9933)

        # 绘制碰撞球体（投影为圆形）
        # 计算球体在屏幕上的投影
        sphere_px = sphere_center[0]
        sphere_py = sphere_center[1]
        sphere_pz = sphere_center[2]
        sphere_screen_x = (sphere_px + 0.8) / 1.6 * WIDTH
        sphere_screen_y = (1.0 - (sphere_py + 0.2) / 1.8) * HEIGHT
        sphere_screen_r = sphere_radius / 1.6 * WIDTH * 0.5
        gui.circle((sphere_screen_x, sphere_screen_y), radius=sphere_screen_r, color=0x88FF88)

        # 显示信息
        gui.text(f"Springs: Structural + Shear + Bending ({num_springs[None]})", (10, 10), color=0xFFFFFF)
        gui.text(f"Collision: {'ON' if enable_collision else 'OFF'} (C)", (10, 30), color=0xFFFF00 if enable_collision else 0x888888)
        gui.text("P: Pause  R: Reset  S: Toggle Springs  C: Toggle Collision  ESC: Exit", (10, HEIGHT - 20), color=0xAAAAAA)

        gui.show()


if __name__ == "__main__":
    main()