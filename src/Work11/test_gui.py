import taichi as ti

# 尝试不同的后端
ti.init(arch=ti.cpu, cpu_max_num_threads=4)
gui = ti.GUI("Test", (800, 600))

print("GUI created successfully")

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
    
    gui.clear(0xFFFFFF)  # 白色背景
    gui.circle((400, 300), radius=50, color=0xFF0000)
    gui.text("Hello!", (380, 290), color=0x000000)
    gui.show()