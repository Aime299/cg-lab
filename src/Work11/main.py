import taichi as ti

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

WIDTH, HEIGHT = 800, 600


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
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt


def main():
    init_cloth()

    gui = ti.GUI("Mass-Spring System - 付雅婷", (WIDTH, HEIGHT))
    current_method = 1
    paused = False
    method_names = ["Explicit", "Semi-Implicit", "Implicit"]

    print("=" * 50)
    print("质点弹簧模型 - 布料模拟")
    print("=" * 50)
    print("  E - 显式欧拉    S - 半隐式欧拉    I - 隐式欧拉")
    print("  P - 暂停/恢复   R - 重置布料")
    print("  ESC - 退出")
    print("=" * 50)

    # 先运行几帧让布料稳定
    for _ in range(10):
        step_semi_implicit()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'e':
                current_method = 0
                init_cloth()
                print("切换到: 显式欧拉")
            elif gui.event.key == 's':
                current_method = 1
                init_cloth()
                print("切换到: 半隐式欧拉")
            elif gui.event.key == 'i':
                current_method = 2
                init_cloth()
                print("切换到: 隐式欧拉")
            elif gui.event.key == 'p':
                paused = not paused
                print("暂停" if paused else "继续")
            elif gui.event.key == 'r':
                init_cloth()
                print("布料已重置")

        if not paused:
            for _ in range(20):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_semi_implicit()

        gui.clear(0x000000)

        # 先画一个白色矩形测试 GUI 是否正常
        gui.rect((100, 100), (200, 150), color=0xFF0000)

        # 计算屏幕坐标并绘制
        points_2d = []
        for idx in range(N * N):
            px = x[idx][0]
            py = x[idx][1]
            pz = x[idx][2]
            # 简单正交投影
            screen_x = (px + 0.8) / 1.6 * WIDTH
            screen_y = (1.0 - (py + 0.2) / 1.8) * HEIGHT
            points_2d.append((screen_x, screen_y))

        # 绘制弹簧线
        num = int(num_springs[None])
        for i in range(num):
            a = int(spring_pairs[i][0])
            b = int(spring_pairs[i][1])
            p1 = points_2d[a]
            p2 = points_2d[b]
            gui.line(p1, p2, color=0x66CCFF, radius=1)

        # 绘制质点
        for idx in range(N * N):
            p = points_2d[idx]
            if 0 <= p[0] <= WIDTH and 0 <= p[1] <= HEIGHT:
                gui.circle(p, radius=2, color=0xFF9933)

        gui.text(f"Method: {method_names[current_method]} (E/S/I)", (10, 10), color=0xFFFFFF)
        gui.text("P: Pause  R: Reset  ESC: Exit", (10, 30), color=0xAAAAAA)
        gui.text(f"Status: {'PAUSED' if paused else 'RUNNING'}", (10, 50), color=0xFFFF00)

        gui.show()


if __name__ == "__main__":
    main()