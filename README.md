# CG-Lab 计算机图形学课程作业

## 实验信息
- **课程**：计算机图形学
- **教师**：张鸿文
- **助教**：张怡冉
- **实验名称**：图形学开发工具

## 环境配置

### 开发工具
- **IDE**：Trae (VS Code 内核)
- **包管理器**：uv (Rust 编写的高性能 Python 包管理器)
- **版本控制**：Git

### 依赖安装
```bash
uv add taichi
项目结构（src 布局）
text
CG-Lab/
├── src/
│   └── Work0/
│       ├── __init__.py      # 包初始化文件
│       ├── config.py        # 配置参数（粒子数量、引力常数、窗口大小等）
│       ├── physics.py       # 物理计算（万有引力、速度更新）
│       └── main.py          # 主程序（GUI 初始化、渲染循环、交互处理）
├── .gitignore               # Git 忽略文件配置
├── pyproject.toml           # 项目依赖配置
├── README.md                # 项目说明文档
└── 屏幕录制 2026-05-21 192003.mp4  # 运行效果演示视频
运行方法
1. 同步依赖
bash
uv sync
2. 运行程序
bash
uv run -m src.Work0.main
GPU 加速情况 ✅
程序成功调用 NVIDIA CUDA 进行 GPU 并行计算：

text
[Taichi] version 1.7.4, llvm 15.0.1, commit b4b956fd, win, python 3.12.13
[Taichi] Starting on arch=cuda
说明：这是最理想的 GPU 调用状态，粒子群模拟性能最优。

运行效果演示
点击观看演示视频

或直接在 GitHub 上点击视频文件播放。

演示内容：

粒子系统实时运行（60fps）

鼠标移动产生力场影响粒子运动

粒子间万有引力相互作用

GPU 加速的平滑渲染

功能说明
核心特性
万有引力模拟：粒子之间根据距离和质量产生引力

GPU 并行计算：使用 Taichi 将计算部署到 GPU，支持大规模粒子群

实时交互：鼠标位置作为力场源，影响粒子运动轨迹

平滑渲染：60fps 实时可视化

代码分层设计
config.py：集中管理所有配置参数（粒子数量、引力常数、窗口大小、颜色等）

physics.py：实现万有引力计算和粒子位置/速度更新（GPU 并行）

main.py：主程序入口，负责 GUI 初始化、渲染循环和鼠标交互

实验任务完成情况
任务1：基础图形学开发环境搭建（uv + Taichi）

任务2：src 布局重构和代码分层（config, physics, main）

任务3：GPU 粒子系统实现（万有引力仿真）

任务4：Git 代码托管（GitHub + README + 演示视频）

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