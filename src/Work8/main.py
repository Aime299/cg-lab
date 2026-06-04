import taichi as ti

ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

light_pos_x = 2.0
light_pos_y = 4.0
light_pos_z = 3.0
max_bounces = 3
samples_per_pixel = 1

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

IOR_AIR = 1.0
IOR_GLASS = 1.5


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
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t = (plane_y - ro.y) / rd.y
        if t < 0:
            t = -1.0
    return t, normal


@ti.func
def get_ground_color(p):
    ix = ti.floor(p.x * 2.0)
    iz = ti.floor(p.z * 2.0)
    result = ti.Vector([0.0, 0.0, 0.0])
    if (ix + iz) % 2 == 0:
        result = ti.Vector([0.3, 0.3, 0.3])
    else:
        result = ti.Vector([0.8, 0.8, 0.8])
    return result


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 1.0])
        hit_mat = MAT_GLASS

    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        hit_c = get_ground_color(p)

    return min_t, hit_n, hit_c, hit_mat


@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32, bounces: ti.i32, samples: ti.i32):
    light_pos = ti.Vector([light_x, light_y, light_z])
    camera_pos = ti.Vector([0.0, 1.0, 5.0])
    
    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        for s in range(samples):
            offset_x = (ti.random() - 0.5) / WIDTH
            offset_y = (ti.random() - 0.5) / HEIGHT
            
            u = (i + offset_x - WIDTH / 2.0) / HEIGHT * 2.0
            v = (j + offset_y - HEIGHT / 2.0) / HEIGHT * 2.0
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
            
            color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            current_ro = camera_pos
            current_rd = rd
            bg_color = ti.Vector([0.05, 0.15, 0.2])
            
            for bounce in range(bounces):
                t, N, obj_color, mat_id = scene_intersect(current_ro, current_rd)
                
                if t > 1e9:
                    color += throughput * bg_color
                    break
                
                p = current_ro + current_rd * t
                
                if mat_id == MAT_MIRROR:
                    current_ro = p + N * 1e-4
                    current_rd = normalize(reflect(current_rd, N))
                    throughput *= 0.8 * obj_color
                
                elif mat_id == MAT_GLASS:
                    # 简化：玻璃当做镜面处理（反射）
                    current_ro = p + N * 1e-4
                    current_rd = normalize(reflect(current_rd, N))
                    throughput *= 0.9 * obj_color
                
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    
                    shadow_ro = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ro, L)
                    dist_to_light = (light_pos - p).norm()
                    
                    ambient = 0.2 * obj_color
                    
                    if shadow_t > dist_to_light or shadow_t < 0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        color += throughput * (ambient + diffuse)
                    else:
                        color += throughput * ambient
                    break
            
            final_color += color
        
        final_color /= samples
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


def main():
    global light_pos_x, light_pos_y, light_pos_z, max_bounces, samples_per_pixel
    gui = ti.GUI("Ray Tracing - Glass - 付雅婷", (WIDTH, HEIGHT))
    
    print("=" * 60)
    print("光线追踪 - 玻璃材质 (反射效果)")
    print("=" * 60)
    print("左侧：玻璃球体（反射）")
    print("右侧：银色镜面球体")
    print("底部：黑白棋盘格地面")
    print("")
    print("控制说明：")
    print("  Q/A - 光源 X 轴")
    print("  W/S - 光源 Y 轴")
    print("  E/D - 光源 Z 轴")
    print("  R/F - 增加/减少 最大弹射次数")
    print("  Z/X - 增加/减少 MSAA 采样次数")
    print("  ESC - 退出")
    print("=" * 60)
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'q':
                light_pos_x = max(-5.0, light_pos_x - 0.2)
            elif gui.event.key == 'a':
                light_pos_x = min(5.0, light_pos_x + 0.2)
            elif gui.event.key == 'w':
                light_pos_y = max(1.0, light_pos_y - 0.2)
            elif gui.event.key == 's':
                light_pos_y = min(8.0, light_pos_y + 0.2)
            elif gui.event.key == 'e':
                light_pos_z = max(-5.0, light_pos_z - 0.2)
            elif gui.event.key == 'd':
                light_pos_z = min(5.0, light_pos_z + 0.2)
            elif gui.event.key == 'r':
                max_bounces = min(5, max_bounces + 1)
                print(f"Bounces: {max_bounces}")
            elif gui.event.key == 'f':
                max_bounces = max(1, max_bounces - 1)
                print(f"Bounces: {max_bounces}")
            elif gui.event.key == 'z':
                samples_per_pixel = min(4, samples_per_pixel + 1)
                print(f"MSAA: {samples_per_pixel}x")
            elif gui.event.key == 'x':
                samples_per_pixel = max(1, samples_per_pixel - 1)
                print(f"MSAA: {samples_per_pixel}x")
        
        render(light_pos_x, light_pos_y, light_pos_z, max_bounces, samples_per_pixel)
        gui.set_image(pixels)
        gui.text(f"Light: ({light_pos_x:.1f}, {light_pos_y:.1f}, {light_pos_z:.1f})", (10, 10), color=0xFFFFFF)
        gui.text(f"Bounces: {max_bounces}", (10, 30), color=0xFFFFFF)
        gui.text(f"MSAA: {samples_per_pixel}x", (10, 50), color=0xFFFFFF)
        gui.text("Glass (Left) | Mirror (Right)", (10, HEIGHT - 35), color=0xFFFF00)
        gui.text("Q/A:X  W/S:Y  E/D:Z  R/F:Bounces  Z/X:MSAA", (10, HEIGHT - 20), color=0x888888)
        gui.show()


if __name__ == "__main__":
    main()