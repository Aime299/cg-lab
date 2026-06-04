"""
可微渲染 - 联合纹理优化 (选做)
拟合 RGB 图像和剪影，同时优化网格顶点和纹理
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ========== 1. 加载目标奶牛模型 ==========
print("加载奶牛模型...")
obj_filename = "cow.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)

# 归一化
verts = mesh.verts_packed()
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
mesh.scale_verts_((1.0 / float(scale)))

# ========== 2. 创建多视角数据集 ==========
print("创建多视角数据集...")
num_views = 20
elev = torch.linspace(0, 360, num_views)
azim = torch.linspace(-180, 180, num_views)

# 光源
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# 摄像机
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 渲染设置
raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 纹理渲染器
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights)
)

# 剪影渲染器
sigma = 1e-4
raster_settings_silhouette = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50,
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings_silhouette),
    shader=SoftSilhouetteShader()
)

# 渲染目标图像
meshes = mesh.extend(num_views)
target_images = renderer_textured(meshes, cameras=cameras, lights=lights)
target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

target_silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
target_silhouette = [target_silhouette_images[i, ..., 3] for i in range(num_views)]

target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], T=T[None, i, ...]) 
                  for i in range(num_views)]

print(f"数据集创建完成，共 {num_views} 个视角")

# ========== 3. 初始化球体 ==========
print("初始化球体...")
src_mesh = ico_sphere(4, device)

# 可视化函数
def visualize_prediction(predicted_mesh, target_image, title=''):
    with torch.no_grad():
        predicted_images = renderer_textured(predicted_mesh)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., :3].cpu().detach().numpy())
    plt.title("预测")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title("目标")
    plt.axis("off")
    plt.suptitle(title)
    plt.show()

# ========== 4. 优化循环（纹理+形状） ==========
print("开始优化...")
verts_shape = src_mesh.verts_packed().shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)

Niter = 1000
num_views_per_iteration = 2

# 损失函数权重
loss_weights = {
    "rgb": 1.0,
    "silhouette": 1.0,
    "edge": 1.0,
    "normal": 0.01,
    "laplacian": 1.0,
}

losses_history = {"rgb": [], "silhouette": [], "edge": [], "normal": [], "laplacian": []}

for i in tqdm(range(Niter)):
    optimizer.zero_grad()
    
    # 变形网格
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    
    # 正则化损失
    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # 渲染损失
    loss_rgb = torch.tensor(0.0, device=device)
    loss_silhouette = torch.tensor(0.0, device=device)
    
    for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
        images_pred = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)
        
        # RGB 损失
        pred_rgb = images_pred[..., :3]
        loss_rgb += ((pred_rgb - target_rgb[j]) ** 2).mean() / num_views_per_iteration
        
        # 剪影损失
        pred_sil = renderer_silhouette(new_src_mesh, cameras=target_cameras[j], lights=lights)[..., 3]
        loss_silhouette += ((pred_sil - target_silhouette[j]) ** 2).mean() / num_views_per_iteration
    
    # 总损失
    total_loss = (loss_weights["rgb"] * loss_rgb + 
                  loss_weights["silhouette"] * loss_silhouette +
                  loss_weights["edge"] * loss_edge +
                  loss_weights["normal"] * loss_normal +
                  loss_weights["laplacian"] * loss_laplacian)
    
    total_loss.backward()
    optimizer.step()
    
    # 记录损失
    losses_history["rgb"].append(loss_rgb.item())
    losses_history["silhouette"].append(loss_silhouette.item())
    losses_history["edge"].append(loss_edge.item())
    losses_history["normal"].append(loss_normal.item())
    losses_history["laplacian"].append(loss_laplacian.item())
    
    # 每 200 步显示结果
    if i % 200 == 0:
        print(f"\n迭代 {i}: 总损失={total_loss.item():.4f}, RGB={loss_rgb.item():.4f}, 剪影={loss_silhouette.item():.4f}")
        visualize_prediction(new_src_mesh, target_rgb[1], title=f"迭代 {i}")

print("优化完成！")

# ========== 5. 保存结果 ==========
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts * scale + center

os.makedirs("output_meshes_advanced", exist_ok=True)
save_obj("output_meshes_advanced/final_model.obj", final_verts, final_faces)
print("模型已保存到 output_meshes_advanced/final_model.obj")

# 绘制损失曲线
plt.figure(figsize=(12, 5))
for name, values in losses_history.items():
    plt.plot(values, label=name)
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.legend()
plt.title("损失函数变化曲线")
plt.savefig("loss_curve.png")
plt.show()
print("损失曲线已保存到 loss_curve.png")