import taichi as ti
import numpy as np

# 初始化
ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 800, 600

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 材质参数
Ka = 0.2   # 环境光系数
Kd = 0.7   # 漫反射系数
Ks = 0.5   # 高光系数
shininess = 32.0  # 高光指数


@ti.func
def normalize(v):
    return v / (v.norm() + 1e-8)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
    """光线与球体相交"""
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 1e-5:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """光线与圆锥相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    ro_local = ro - apex
    
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    if ti.abs(A) > 1e-8:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            if t1 > t2:
                t1, t2 = t2, t1
                
            y1 = ro_local.y + t1 * rd.y
            if t1 > 1e-5 and -H <= y1 <= 0:
                t = t1
            else:
                y2 = ro_local.y + t2 * rd.y
                if t2 > 1e-5 and -H <= y2 <= 0:
                    t = t2
                    
            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
    return t, normal


@ti.kernel
def render(ka: ti.f32, kd: ti.f32, ks: ti.f32, ns: ti.f32):
    """渲染主函数"""
    for i, j in pixels:
        # 计算射线方向
        u = (i - WIDTH / 2.0) / HEIGHT * 2.0
        v = (j - HEIGHT / 2.0) / HEIGHT * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))
        
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 球体（左侧）
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
        
        # 圆锥（右侧）
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])
        
        # 背景色
        color = ti.Vector([0.05, 0.15, 0.15])
        
        # 如果击中物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)
            
            # 环境光
            ambient = ka * light_color * hit_color
            
            # 漫反射
            diff = ti.max(0.0, N.dot(L))
            diffuse = kd * diff * light_color * hit_color
            
            # 高光
            R = normalize(reflect(-L, N))
            spec = ti.max(0.0, R.dot(V)) ** ns
            specular = ks * spec * light_color
            
            color = ambient + diffuse + specular
            color = ti.math.clamp(color, 0.0, 1.0)
        
        pixels[i, j] = color


def main():
    global Ka, Kd, Ks, shininess
    
    gui = ti.GUI("Phong Shading - 付雅婷", (WIDTH, HEIGHT))
    
    print("=" * 50)
    print("Phong 光照模型")
    print("=" * 50)
    print("左侧：红色球体")
    print("右侧：紫色圆锥")
    print("")
    print("控制说明：")
    print("  Q/A - 增加/减少 环境光(Ka)")
    print("  W/S - 增加/减少 漫反射(Kd)")
    print("  E/D - 增加/减少 高光(Ks)")
    print("  R/F - 增加/减少 高光指数(Shininess)")
    print("  ESC - 退出")
    print("=" * 50)
    
    while gui.running:
        # 键盘控制
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'q':
                Ka = min(1.0, Ka + 0.05)
                print(f"Ka (环境光): {Ka:.2f}")
            elif gui.event.key == 'a':
                Ka = max(0.0, Ka - 0.05)
                print(f"Ka (环境光): {Ka:.2f}")
            elif gui.event.key == 'w':
                Kd = min(1.0, Kd + 0.05)
                print(f"Kd (漫反射): {Kd:.2f}")
            elif gui.event.key == 's':
                Kd = max(0.0, Kd - 0.05)
                print(f"Kd (漫反射): {Kd:.2f}")
            elif gui.event.key == 'e':
                Ks = min(1.0, Ks + 0.05)
                print(f"Ks (高光): {Ks:.2f}")
            elif gui.event.key == 'd':
                Ks = max(0.0, Ks - 0.05)
                print(f"Ks (高光): {Ks:.2f}")
            elif gui.event.key == 'r':
                shininess = min(128.0, shininess + 5)
                print(f"Shininess (高光指数): {shininess:.0f}")
            elif gui.event.key == 'f':
                shininess = max(1.0, shininess - 5)
                print(f"Shininess (高光指数): {shininess:.0f}")
        
        # 渲染
        render(Ka, Kd, Ks, shininess)
        
        # 显示像素
        gui.set_image(pixels)
        
        # 显示参数
        gui.text(f"Ka (环境光): {Ka:.2f}  [Q/A]", (10, 10), color=0xFFFFFF)
        gui.text(f"Kd (漫反射): {Kd:.2f}  [W/S]", (10, 30), color=0xFFFFFF)
        gui.text(f"Ks (高光): {Ks:.2f}    [E/D]", (10, 50), color=0xFFFFFF)
        gui.text(f"Shininess: {shininess:.0f}  [R/F]", (10, 70), color=0xFFFFFF)
        gui.text("ESC: 退出", (10, HEIGHT - 20), color=0x888888)
        
        gui.show()


if __name__ == "__main__":
    main()