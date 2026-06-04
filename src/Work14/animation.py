"""
实验八选做：LBS 蒙皮 - 姿态动画
让关节从 0 逐渐旋转到某个角度，生成动画 GIF
"""

import os
import sys
import types
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

# ========== Chumpy 兼容补丁 ==========
class _ChumpyArrayShim:
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data")
    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)
    @property
    def shape(self):
        return np.asarray(self).shape
    def __len__(self):
        return len(np.asarray(self))
    def __getitem__(self, item):
        return np.asarray(self)[item]

def install_chumpy_pickle_shim():
    if "chumpy.ch" in sys.modules:
        return
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module

install_chumpy_pickle_shim()

# ========== 加载 SMPL 模型 ==========
import smplx

os.makedirs("outputs_animation", exist_ok=True)

device = torch.device("cpu")
dtype = torch.float32

model_path = "./models"
model = smplx.create(
    model_path=model_path,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
).to(device)

print("模型加载成功")
print(f"顶点数: {model.v_template.shape[0]}")
print(f"关节数: {model.lbs_weights.shape[1]}")

# ========== 固定 shape 参数 ==========
betas = torch.zeros((1, 10), dtype=dtype, device=device)
betas[0, 0] = 1.5
betas[0, 1] = -0.8

global_orient = torch.zeros((1, 3), dtype=dtype, device=device)

joint_names = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]

joint_idx = 16  # left_shoulder
start_angle = 0
end_angle = 90
num_frames = 30

print(f"\n动画设置:")
print(f"  关节: {joint_names[joint_idx]} (索引 {joint_idx})")
print(f"  角度范围: {start_angle}° → {end_angle}°")
print(f"  帧数: {num_frames}")

angles_deg = np.linspace(start_angle, end_angle, num_frames)
angles_rad = np.deg2rad(angles_deg)

lbs_weights = model.lbs_weights.detach().cpu().numpy()
faces = model.faces

frames = []

def get_vertex_colors_by_joint_weights(weights, joint_id):
    return plt.cm.hot(weights[:, joint_id])

def smpl_to_plot_coords(points):
    return points[:, [0, 2, 1]]

def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def draw_mesh(ax, vertices, faces, joint_weights, joint_idx, title=""):
    plot_vertices = smpl_to_plot_coords(vertices)
    colors = get_vertex_colors_by_joint_weights(joint_weights, joint_idx)
    face_colors = colors[faces].mean(axis=1)
    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=face_colors,
        linewidths=0.05,
        edgecolors=(0, 0, 0, 0.1),
        alpha=0.9
    )
    ax.add_collection3d(mesh)
    set_axes_equal(ax, plot_vertices)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

print("\n生成动画帧...")

for i, angle_rad in enumerate(angles_rad):
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)
    body_pose[0, joint_idx * 3: (joint_idx + 1) * 3] = torch.tensor([0.0, 0.0, angle_rad], dtype=dtype)
    
    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True
        )
        verts = output.vertices[0].detach().cpu().numpy()
        joints = output.joints[0].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    draw_mesh(ax, verts, faces, lbs_weights, joint_idx, 
              title=f"Left Shoulder Rotation: {angles_deg[i]:.1f}°")
    plot_joints = smpl_to_plot_coords(joints)
    ax.scatter(plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
               c='red', s=20, marker='o', alpha=0.8, edgecolors='white')
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)
    frames.append(frame)
    plt.close(fig)
    
    if (i + 1) % 5 == 0:
        print(f"  生成帧 {i+1}/{num_frames}")

print("\n保存 GIF...")
gif_path = "outputs_animation/lbs_animation.gif"
imageio.mimsave(gif_path, frames, fps=10, loop=0)
print(f"动画已保存: {gif_path}")

print("\n保存关键帧...")
keyframes = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
for idx in keyframes:
    angle = angles_deg[idx]
    angle_rad = angles_rad[idx]
    
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)
    body_pose[0, joint_idx * 3: (joint_idx + 1) * 3] = torch.tensor([0.0, 0.0, angle_rad], dtype=dtype)
    
    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True
        )
        verts = output.vertices[0].detach().cpu().numpy()
        joints = output.joints[0].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    draw_mesh(ax, verts, faces, lbs_weights, joint_idx, 
              title=f"Left Shoulder: {angle:.0f}°")
    plot_joints = smpl_to_plot_coords(joints)
    ax.scatter(plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
               c='red', s=20, marker='o', alpha=0.8, edgecolors='white')
    plt.savefig(f"outputs_animation/keyframe_{angle:.0f}deg.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: keyframe_{angle:.0f}deg.png")

print("\n完成！")
print(f"GIF 位置: {gif_path}")