# CG-Lab 计算机图形学课程作业

## 学生信息
- **姓名**：付雅婷
- **班级**：24级计算机科学与技术（公费师范）
- **课程**：计算机图形学
- **教师**：张鸿文
- **助教**：张怡冉

## 实验一：图形学开发工具

### 环境配置

#### 开发工具
- **IDE**：Trae (VS Code 内核)
- **包管理器**：uv (Rust 编写的高性能 Python 包管理器)
- **版本控制**：Git

#### 依赖安装
```bash
uv add taichi
项目结构（src 布局）
text
CG-Lab/
├── src/
│   ├── Work0/                    # 实验一：粒子系统
│   │   ├── __init__.py
│   │   ├── config.py             # 配置参数
│   │   ├── physics.py            # 物理计算
│   │   └── main.py               # 主程序入口
│   └── Work1/                    # 实验二：旋转与变换
│       ├── __init__.py
│       ├── main.py               # 主程序入口
│       └── transform.py          # 变换矩阵实现
├── .gitignore
├── pyproject.toml
├── README.md
├── 屏幕录制 2026-05-21 192003.mp4   # 实验一演示视频
└── 屏幕录制 2026-05-21 193933.mp4   # 实验二演示视频
运行方法
实验一：粒子系统
bash
uv run -m src.Work0.main
实验二：旋转与变换
bash
uv run -m src.Work1.main
实验一：万有引力粒子群仿真
GPU 加速情况 ✅
程序成功调用 NVIDIA CUDA 进行 GPU 并行计算：

text
[Taichi] version 1.7.4, llvm 15.0.1, commit b4b956fd, win, python 3.12.13
[Taichi] Starting on arch=cuda
说明：这是最理想的 GPU 调用状态，粒子群模拟性能最优。

运行效果演示
点击观看实验一演示视频

功能说明
万有引力模拟：粒子之间根据距离和质量产生引力

GPU 并行计算：使用 Taichi 将计算部署到 GPU

实时交互：鼠标移动产生力场影响粒子运动

平滑渲染：60fps 实时可视化

代码分层设计
config.py：集中管理配置参数（粒子数量、引力常数、窗口大小等）

physics.py：实现万有引力计算和粒子位置更新（GPU 并行）

main.py：主程序入口，负责 GUI 初始化、渲染循环和鼠标交互

实验二：旋转与变换
实验目标
深入理解 3D 空间中的坐标变换流程（MVP 变换），独立推导并用代码实现模型变换、视图变换和投影变换矩阵。

三角形顶点坐标
v0: (2.0, 0.0, -2.0)

v1: (0.0, 2.0, -2.0)

v2: (-2.0, 0.0, -2.0)

核心功能
1. 模型变换 (Model Transformation)
绕 Z 轴旋转的变换矩阵：

python
def get_model_matrix(angle):
    rad = angle * π / 180.0
    return [[cos(rad), -sin(rad), 0, 0],
            [sin(rad),  cos(rad), 0, 0],
            [0,         0,        1, 0],
            [0,         0,        0, 1]]
2. 视图变换 (View Transformation)
将相机平移到世界坐标系原点：

python
def get_view_matrix(eye_pos):
    return [[1, 0, 0, -eye_pos[0]],
            [0, 1, 0, -eye_pos[1]],
            [0, 0, 1, -eye_pos[2]],
            [0, 0, 0, 1]]
3. 投影变换 (Projection Transformation)
透视投影矩阵，分为两步：

透视平截头体挤压为正交长方体

正交投影至规范立方体 [-1, 1]³

交互操作
按 A 键：三角形顺时针旋转

按 D 键：三角形逆时针旋转

按 ESC 键：退出程序

运行效果演示
点击观看实验二演示视频

技术要点
角度转弧度：三角函数使用弧度制

齐次坐标：4D 向量 (x, y, z, w)

透视除法：除以 w 得到 NDC 坐标

视口变换：NDC [-1,1] → 屏幕 [0, width/height]

实验任务完成情况
实验一
基础图形学开发环境搭建（uv + Taichi）

src 布局重构和代码分层（config, physics, main）

GPU 粒子系统实现（万有引力仿真）

Git 代码托管（GitHub + README + 演示视频）

实验二
模型变换矩阵（绕 Z 轴旋转）

视图变换矩阵（相机平移）

透视投影矩阵

三角形线框渲染

A/D 键控制旋转

代码提交到 GitHub

演示视频上传

技术栈
工具/库	用途	版本
Python	编程语言	3.12.13
Taichi	GPU 并行计算图形学库	1.7.4
uv	包管理器	最新
Git	版本控制	2.54.0
参考资料
Taichi 官方文档

uv 包管理器文档

实验课程主页

src 布局说明

仓库地址
https://github.com/Aime299/cg-lab