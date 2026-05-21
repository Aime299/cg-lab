import taichi as ti

ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

Ka = 0.2
Kd = 0.7
Ks = 0.5
shininess = 32.0
use_blinn_phong = True
use_shadow = True


@ti.func
def normalize(v):
    return v / (v.norm() + 1e-8)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
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


@ti.func
def check_shadow(p, light_pos):
    light_dir = normalize(light_pos - p)
    shadow_ro = p + light_dir * 1e-4
    dist_to_light = (light_pos - p).norm()
    result = 0.0
    t_sphere, _ = intersect_sphere(shadow_ro, light_dir, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
    if t_sphere > 1e-5 and t_sphere < dist_to_light:
        result = 1.0
    t_cone, _ = intersect_cone(shadow_ro, light_dir, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
    if t_cone > 1e-5 and t_cone < dist_to_light:
        result = 1.0
    return result


@ti.kernel
def render(ka: ti.f32, kd: ti.f32, ks: ti.f32, ns: ti.f32, blinn_mode: ti.i32, shadow_mode: ti.i32):
    for i, j in pixels:
        u = (i - WIDTH / 2.0) / HEIGHT * 2.0
        v = (j - HEIGHT / 2.0) / HEIGHT * 2.0
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))
        
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        
        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])
            hit_pos = ro + rd * min_t
        
        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])
            hit_pos = ro + rd * min_t
        
        color = ti.Vector([0.05, 0.15, 0.15])
        
        if min_t < 1e9:
            N = hit_normal
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])
            L = normalize(light_pos - hit_pos)
            V = normalize(ro - hit_pos)
            ambient = ka * light_color * hit_color
            
            in_shadow = 0.0
            if shadow_mode == 1:
                in_shadow = check_shadow(hit_pos, light_pos)
            
            if in_shadow > 0.5:
                color = ambient
            else:
                diff = ti.max(0.0, N.dot(L))
                diffuse = kd * diff * light_color * hit_color
                
                spec_val = 0.0
                if blinn_mode == 1:
                    H = normalize(L + V)
                    spec_val = ti.max(0.0, N.dot(H)) ** ns
                else:
                    R = normalize(reflect(-L, N))
                    spec_val = ti.max(0.0, R.dot(V)) ** ns
                
                specular = ks * spec_val * light_color
                color = ambient + diffuse + specular
            
            color = ti.math.clamp(color, 0.0, 1.0)
        
        pixels[i, j] = color


def main():
    global Ka, Kd, Ks, shininess, use_blinn_phong, use_shadow
    gui = ti.GUI("Phong vs Blinn-Phong + Shadow - 付雅婷", (WIDTH, HEIGHT))
    
    print("=" * 60)
    print("Phong / Blinn-Phong 光照模型 + 硬阴影")
    print("=" * 60)
    print("  Q/A - 环境光(Ka)    W/S - 漫反射(Kd)")
    print("  E/D - 高光(Ks)      R/F - 高光指数")
    print("  P   - 切换光照模型   H   - 切换阴影")
    print("  ESC - 退出")
    print("=" * 60)
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'q':
                Ka = min(1.0, Ka + 0.05)
                print(f"Ka: {Ka:.2f}")
            elif gui.event.key == 'a':
                Ka = max(0.0, Ka - 0.05)
                print(f"Ka: {Ka:.2f}")
            elif gui.event.key == 'w':
                Kd = min(1.0, Kd + 0.05)
                print(f"Kd: {Kd:.2f}")
            elif gui.event.key == 's':
                Kd = max(0.0, Kd - 0.05)
                print(f"Kd: {Kd:.2f}")
            elif gui.event.key == 'e':
                Ks = min(1.0, Ks + 0.05)
                print(f"Ks: {Ks:.2f}")
            elif gui.event.key == 'd':
                Ks = max(0.0, Ks - 0.05)
                print(f"Ks: {Ks:.2f}")
            elif gui.event.key == 'r':
                shininess = min(128.0, shininess + 5)
                print(f"Shininess: {shininess:.0f}")
            elif gui.event.key == 'f':
                shininess = max(1.0, shininess - 5)
                print(f"Shininess: {shininess:.0f}")
            elif gui.event.key == 'p':
                use_blinn_phong = not use_blinn_phong
                mode = "Blinn-Phong" if use_blinn_phong else "Phong"
                print(f"切换到: {mode}")
            elif gui.event.key == 'h':
                use_shadow = not use_shadow
                print(f"阴影: {'开启' if use_shadow else '关闭'}")
        
        render(Ka, Kd, Ks, shininess, 1 if use_blinn_phong else 0, 1 if use_shadow else 0)
        gui.set_image(pixels)
        
        mode_text = "Blinn-Phong" if use_blinn_phong else "Phong"
        shadow_text = "ON" if use_shadow else "OFF"
        gui.text(f"Ka:{Ka:.2f} Kd:{Kd:.2f} Ks:{Ks:.2f} N:{shininess:.0f}", (10, 10), color=0xFFFFFF)
        gui.text(f"Mode: {mode_text}  Shadow: {shadow_text}", (10, 30), color=0xFFFF00)
        gui.text("Q/A:Ka  W/S:Kd  E/D:Ks  R/F:N  P:Mode  H:Shadow", (10, HEIGHT - 20), color=0x888888)
        
        gui.show()


if __name__ == "__main__":
    main()