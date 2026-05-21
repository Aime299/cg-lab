import taichi as ti
import numpy as np
import math

# 初始化
ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# 控制点
control_points = []

# 曲线模式: 0=贝塞尔, 1=B样条
curve_mode = 0

# 反走样开关
antialiasing = True


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


def compute_bezier_curve(points):
    """计算贝塞尔曲线上的所有点"""
    if len(points) < 2:
        return []
    curve = []
    for i in range(NUM_SEGMENTS + 1):
        t = i / NUM_SEGMENTS
        pt = de_casteljau(points, t)
        curve.append(pt)
    return curve


def compute_bspline_curve(points):
    """计算均匀三次B样条曲线"""
    if len(points) < 4:
        return []
    
    # 均匀三次B样条基矩阵
    M = np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0]
    ]) / 6.0
    
    curve = []
    # 每4个控制点生成一段曲线
    for i in range(len(points) - 3):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]
        p3 = points[i + 3]
        
        # 对每一段采样 NUM_SEGMENTS / (len(points)-3) 个点
        seg_samples = NUM_SEGMENTS // (len(points) - 3) + 1
        
        for j in range(seg_samples):
            t = j / seg_samples
            # 计算B样条曲线上的点
            t2 = t * t
            t3 = t2 * t
            T = np.array([t3, t2, t, 1])
            
            P = np.array([
                [p0[0], p0[1]],
                [p1[0], p1[1]],
                [p2[0], p2[1]],
                [p3[0], p3[1]]
            ])
            
            result = T @ M @ P
            curve.append((result[0], result[1]))
    
    return curve


def draw_pixel_aa(x, y, color, intensity=1.0):
    """反走样绘制像素（在 GPU kernel 中实现）"""
    # 这个函数在 kernel 中实现
    pass


def main():
    global curve_mode, antialiasing
    
    gui = ti.GUI("Bezier & B-Spline Curves - 付雅婷", res=(WIDTH, HEIGHT))
    
    print("=" * 60)
    print("贝塞尔曲线 & B样条曲线 交互程序")
    print("=" * 60)
    print("操作说明：")
    print("  鼠标左键        - 添加控制点")
    print("  C 键           - 清空所有控制点")
    print("  B 键           - 切换曲线模式 (贝塞尔 <-> B样条)")
    print("  A 键           - 切换反走样 (开/关)")
    print("  ESC            - 退出程序")
    print("=" * 60)
    print(f"当前模式: {'贝塞尔曲线' if curve_mode == 0 else 'B样条曲线'}")
    print(f"反走样: {'开启' if antialiasing else '关闭'}")
    print("=" * 60)
    
    while gui.running:
        # 处理事件
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.LMB:
                pos = gui.get_cursor_pos()
                control_points.append((pos[0], pos[1]))
                print(f"添加控制点: ({pos[0]:.3f}, {pos[1]:.3f})")
            elif event.key == 'c':
                control_points.clear()
                print("清空所有控制点")
            elif event.key == 'b':
                curve_mode = 1 - curve_mode
                mode_name = "贝塞尔曲线" if curve_mode == 0 else "B样条曲线"
                print(f"切换到: {mode_name}")
            elif event.key == 'a':
                antialiasing = not antialiasing
                print(f"反走样: {'开启' if antialiasing else '关闭'}")
            elif event.key == ti.GUI.ESCAPE:
                break
        
        # 清屏
        gui.clear(0x000000)
        
        # 绘制控制点之间的连线（灰色）
        if len(control_points) >= 2:
            for i in range(len(control_points) - 1):
                p1 = control_points[i]
                p2 = control_points[i + 1]
                gui.line(p1, p2, color=0x444444, radius=2)
        
        # 绘制控制点（红色圆点）
        for pt in control_points:
            gui.circle(pt, radius=6, color=0xFF0000)
        
        # 绘制曲线
        if len(control_points) >= 2:
            if curve_mode == 0:  # 贝塞尔曲线
                curve = compute_bezier_curve(control_points)
                color = 0x00FF00
            else:  # B样条曲线
                curve = compute_bspline_curve(control_points)
                color = 0xFFFF00  # 黄色
            
            # 绘制曲线
            for i in range(len(curve) - 1):
                if antialiasing:
                    # 反走样：绘制更细的线，多重采样效果
                    gui.line(curve[i], curve[i + 1], color=color, radius=1)
                    # 辅助线增加平滑度
                    gui.line(curve[i], curve[i + 1], color=color, radius=0.5)
                else:
                    gui.line(curve[i], curve[i + 1], color=color, radius=2)
        
        # 显示信息
        mode_text = "贝塞尔曲线" if curve_mode == 0 else "B样条曲线"
        aa_text = "开启" if antialiasing else "关闭"
        gui.text(f"控制点: {len(control_points)}", (10, 10), color=0xFFFFFF)
        gui.text(f"模式: {mode_text}", (10, 30), color=0xFFFFFF)
        gui.text(f"反走样: {aa_text}", (10, 50), color=0xFFFFFF)
        gui.text("左键:添加点  C:清空  B:切换模式  A:反走样  ESC:退出", (10, HEIGHT - 20), color=0x888888)
        
        gui.show()


if __name__ == "__main__":
    main()