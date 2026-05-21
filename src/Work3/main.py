import taichi as ti
import numpy as np

# 初始化 - 使用 CPU 后端
ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# 存储控制点
control_points = []


def de_casteljau(points, t):
    """De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append((x, y))
    return de_casteljau(next_points, t)


def compute_curve_points(points):
    """计算曲线上的所有点"""
    if len(points) < 2:
        return []
    
    curve = []
    for i in range(NUM_SEGMENTS + 1):
        t = i / NUM_SEGMENTS
        pt = de_casteljau(points, t)
        curve.append(pt)
    return curve


def main():
    gui = ti.GUI("Bezier Curve - 付雅婷", res=(WIDTH, HEIGHT))
    
    while gui.running:
        # 处理鼠标点击
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.LMB:
                # 获取鼠标位置
                mouse_pos = gui.get_cursor_pos()
                control_points.append((mouse_pos[0], mouse_pos[1]))
                print(f"添加控制点: {mouse_pos}")
            elif gui.event.key == 'c':
                control_points.clear()
                print("清空所有控制点")
            elif gui.event.key == ti.GUI.ESCAPE:
                break
        
        # 清屏
        gui.clear(0x000000)
        
        # 绘制控制点之间的连线（灰色）
        if len(control_points) >= 2:
            for i in range(len(control_points) - 1):
                p1 = control_points[i]
                p2 = control_points[i + 1]
                gui.line(p1, p2, color=0x666666, radius=2)
        
        # 绘制控制点（红色圆点）
        for pt in control_points:
            gui.circle(pt, radius=5, color=0xFF0000)
        
        # 绘制贝塞尔曲线（绿色）
        if len(control_points) >= 2:
            curve = compute_curve_points(control_points)
            for i in range(len(curve) - 1):
                gui.line(curve[i], curve[i + 1], color=0x00FF00, radius=2)
        
        # 显示提示文字
        gui.text(f"控制点数量: {len(control_points)}", (10, 10), color=0xFFFFFF)
        gui.text("鼠标左键: 添加控制点", (10, 30), color=0xAAAAAA)
        gui.text("按 C 键: 清空控制点", (10, 50), color=0xAAAAAA)
        gui.text("按 ESC: 退出程序", (10, 70), color=0xAAAAAA)
        
        gui.show()


if __name__ == "__main__":
    print("=" * 50)
    print("贝塞尔曲线绘制程序")
    print("=" * 50)
    print("操作说明：")
    print("  鼠标左键 - 添加控制点")
    print("  C 键     - 清空所有控制点")
    print("  ESC      - 退出程序")
    print("=" * 50)
    main()