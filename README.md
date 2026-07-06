# 🔥 FireCheck — 视频AI智能识别及预警管理系统

<div align="center">

**基于 YOLOv11 的实时火焰/烟雾检测与预警平台**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF)](https://docs.ultralytics.com/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)](https://opencv.org/)
[![RK3588](https://img.shields.io/badge/NPU-RK3588S-orange)](https://www.rock-chips.com/)

[功能特性](#功能特性) · [系统架构](#系统架构) · [快速开始](#快速开始) · [模型训练](#模型训练) · [API 文档](#api-文档) · [部署指南](#部署指南)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [模型训练](#模型训练)
- [API 文档](#api-文档)
- [部署指南](#部署指南)
- [性能评估](#性能评估)
- [开发计划](#开发计划)

---

## 项目简介

FireCheck 是一套**端到端的智能火焰烟雾检测及预警管理系统**，由边缘设备端和中心 Web 服务端两部分组成。边缘设备基于 Orange Pi 5 Pro (RK3588S) 运行 YOLOv11 模型实时检测摄像头画面，检测到火情后自动截图、录像并上报至中心服务端；管理员通过 Web 数据大屏实时监控、处理和审核每一条告警事件。

> 🎯 **设计目标**：低成本、高可靠、易部署的工业/校园级火灾预警解决方案。

---

## 功能特性

### 🖥️ 边缘设备端
| 功能 | 说明 |
|------|------|
| **YOLOv11 实时检测** | 火焰 + 烟雾双类别检测，支持 PyTorch / RKNN(NPU) 双推理后端 |
| **多路视频源** | 支持本地摄像头、RTSP 网络流、本地视频文件 |
| **隔帧推理优化** | 每 5 帧执行一次推理，帧间平滑，边缘设备 CPU 负载降低 60%+ |
| **告警自动录像** | 触发告警时自动回溯缓冲帧 + 持续录像，事后 ffmpeg 转码 H.264 |
| **WebSocket 实时推流** | 内建 WebSocket 服务端 (端口按 camera_id 自动分配)，供前端实时预览 |
| **设备心跳上报** | 定时上报 CPU/内存使用率、在线状态至服务端 |
| **崩溃自上报** | 全局异常捕获 hook，程序崩溃时自动 POST 堆栈到服务端 |
| **模拟模式** | 无摄像头环境下自动生成测试画面，便于开发调试 |

### 🌐 中心 Web 服务端
| 功能 | 说明 |
|------|------|
| **数据大屏仪表盘** | ECharts + Leaflet 地图 (OpenStreetMap)，实时展示报警统计、月度趋势、区域分布、设备地图 |
| **告警处理工作流** | 三级状态流转：待处理 → 已处理 → 已审核 |
| **角色权限控制** | 超级管理员 / 处理人 / 审核人，基于装饰器的 RBAC 鉴权 |
| **部门 & 用户管理** | 树形部门结构，用户与角色绑定，SHA256 密码哈希 |
| **设备 & 摄像头管理** | AI 分析盒注册、摄像头发现、在线/离线状态自动检测 |
| **操作日志审计** | 全量操作日志 (增删改) + 登录日志，可追溯 |
| **故障自动诊断** | 心跳超时 90s 自动生成设备离线 + 摄像头故障记录 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      边缘设备 (Orange Pi 5 Pro / RK3588S)          │
│                                                                   │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │ 摄像头/RTSP │───▶│  flame_detect.py  │───▶│  WebSocket :9999+  │  │
│  │ 视频文件   │    │  YOLOv11 推理引擎  │    │  实时推流 (JPEG)    │  │
│  └──────────┘    │  · 检测 + 标注     │    └────────────────────┘  │
│                   │  · 告警截图/录像   │                           │
│                   │  · 隔帧推理优化    │    ┌────────────────────┐  │
│                   │  · 帧缓冲区回溯    │───▶│  HTTP POST 上报     │  │
│                   └──────────────────┘    │  · 告警数据+图片+视频│  │
│                                           │  · 心跳(CPU/内存)   │  │
│  ┌──────────────────────────────────┐    │  · 崩溃堆栈         │  │
│  │          main.py (启动器)         │    └─────────┬──────────┘  │
│  │  交互菜单 · 全局异常捕获 · 配置加载 │              │             │
│  └──────────────────────────────────┘              │             │
└────────────────────────────────────────────────────┼─────────────┘
                                                     │
                    HTTP POST / WebSocket             │
                                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    中心 Web 服务端 (Flask :5000)                    │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │  数据大屏仪表盘  │  │   管理后台 CRUD  │  │   REST API 接口    │  │
│  │  ECharts 图表   │  │  设备·用户·角色  │  │  /api/heartbeat   │  │
│  │  Leaflet 地图标注│  │  告警·审核·日志  │  │  /api/alarm       │  │
│  └────────────────┘  └────────────────┘  │  /api/device/error │  │
│                                           │  /api/stats        │  │
│  ┌────────────────────────────────────┐  └────────────────────┘  │
│  │         SQLite 数据库 (13 张表)      │                         │
│  │  T_Site · T_User · T_Device         │                         │
│  │  T_Camera · T_DetectResult · ...    │                         │
│  └────────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

### 数据流
```
摄像头帧采集 → YOLOv11 推理 → 检测到火焰/烟雾?
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                  无目标        冷却期内      触发告警
                  继续循环      跳过          ├─ 保存截图 (标注框)
                                            ├─ 帧缓冲区回溯 + 录像
                                            ├─ ffmpeg 转码 H.264
                                            └─ multipart POST → /api/alarm
```

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **AI 推理** | Ultralytics YOLOv11 + PyTorch | 火焰/烟雾目标检测 |
| **NPU 加速** | RKNN Toolkit2 / rknnlite | RK3588 硬件推理加速 |
| **图像处理** | OpenCV | 摄像头采集、图像标注、视频编码 |
| **边缘端框架** | Python 3.10+, threading, asyncio | 并发检测、心跳、WebSocket |
| **服务端框架** | Flask 3.x | Web 管理后台 + REST API |
| **数据库** | SQLite (WAL 模式) | 设备、用户、告警、日志存储 |
| **前端** | Bootstrap / Tailwind CSS + ECharts + Leaflet (OpenStreetMap) | 数据大屏可视化 |
| **通信协议** | HTTP REST + WebSocket | 数据上报 + 实时推流 |
| **视频编码** | ffmpeg (libx264) | MJPG → H.264 安全转码 |

---

## 目录结构

```
Firecheck/
├── .gitignore                          # Git 忽略规则
│
├── board/                              # ====== 边缘设备端 ======
│   ├── flame_config.json              # 运行时配置文件
│   ├── flame_detect.py                # 🔥 核心检测引擎 (1,100 行)
│   │   ├── Config                     #   配置管理类
│   │   ├── FlameDetector              #   检测器主类
│   │   ├── train_fire_model()         #   集成训练函数
│   │   └── convert_to_rknn()          #   .pt → .rknn 转换
│   ├── main.py                        # 🚀 交互式启动器 (170 行)
│   │   ├── global_exception_handler() #   全局崩溃捕获
│   │   └── show_interactive_menu()    #   交互配置菜单
│   ├── train.py                       # 🏋️ 模型训练脚本 (401 行)
│   │   ├── download_fire_dataset()    #   自动下载公开数据集
│   │   ├── train_model()              #   YOLOv11 训练
│   │   ├── validate_model()           #   模型评估
│   │   ├── export_rknn()              #   RKNN 导出
│   │   └── test_model()               #   单图/摄像头测试
│   ├── test_model.py                  # 🖥️ Qt 可视化测试平台 (357 行)
│   │   ├── YOLOThread(QThread)        #   后台推理线程
│   │   └── MainWindow(QMainWindow)    #   测试 GUI 窗口
│   ├── models/                        # 模型文件
│   │   ├── fire_yolov11.pt            #   训练好的 YOLOv11 模型
│   │   ├── fire_yolov11_final.pt      #   最终版模型
│   │   └── fire_yolov11_eval.json     #   评估指标 (mAP50: 0.751)
│   ├── yolo11n_rknn_model/            # RKNN NPU 模型 (COCO 预训练)
│   │   ├── metadata.yaml              #   80 类 COCO 标签
│   │   └── yolo11n-rk3588.rknn        #   编译后的 RKNN 模型
│   └── runs/detect/fire_detect/       # 训练结果 (图表 + 权重)
│       ├── args.yaml                  #   训练超参数存档
│       ├── results.csv                #   40 轮训练指标
│       ├── confusion_matrix.png       #   混淆矩阵
│       ├── BoxF1_curve.png            #   F1 曲线
│       └── weights/best.pt            #   最佳权重
│
├── server/                             # ====== 中心服务端 ======
│   └── web_server.py                  # Flask Web 服务 (5,787 行)
│       ├── init_db()                  #   13 张表的数据库初始化
│       ├── seed_data()                #   预设用户/角色/权限种子数据
│       ├── login_required / admin_required  # RBAC 鉴权装饰器
│       ├── /dashboard                 #   数据大屏仪表盘
│       ├── /admin/*                   #   管理后台 CRUD (45 个路由)
│       └── /api/*                     #   边缘设备通信 API
│
└── video/                              # ====== 测试视频 ======
    ├── 火灾1.mp4                       # 测试用火焰视频
    ├── 火灾2.mp4
    ├── 火灾3.mp4
    └── 火灾4.mp4
```

---

## 快速开始

### 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.10+ | 边缘设备和服务端均需 |
| ffmpeg | 4.x+ | 视频转码，`sudo apt install ffmpeg` |
| PyTorch | 2.0+ | GPU/CPU 推理 |
| OpenCV | 4.5+ | 视频采集与处理 |

### 1. 克隆项目

```bash
git clone <repo-url>
cd Firecheck
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 需包含以下核心依赖：
> ```
> ultralytics>=8.0.0
> opencv-python>=4.5.0
> numpy>=1.21.0
> flask>=3.0.0
> requests>=2.28.0
> websockets>=11.0
> pyyaml>=6.0
> Pillow>=9.0.0
> ```

### 3. 启动中心服务端

```bash
cd server
python web_server.py
# 访问 http://localhost:5000
# 管理员: admin / 123456
# 处理人: chuli001 / 123456
# 审核人: shenhe001 / 123456
```

### 4. 启动边缘检测设备

```bash
cd board

# 方式一: 交互式菜单启动
python main.py

# 方式二: 命令行直接启动
python flame_detect.py --config flame_config.json

# 方式三: 使用视频文件测试 (无需摄像头)
python flame_detect.py --camera "../video/火灾1.mp4"
```

### 5. Qt 可视化测试 (可选)

```bash
cd board
python test_model.py
# 左侧面板选择: 摄像头实时 / 图片检测 / 视频检测
```

---

## 配置说明

### 边缘设备配置 (`board/flame_config.json`)

```jsonc
{
    // ── 设备标识 ──
    "device_mac": "AAABBBCCCDDD",         // MAC 地址, 自动获取或手动指定
    "device_id": 1,                        // 设备编号
    "camera_id": 1,                        // 摄像头编号 (影响 WebSocket 端口)

    // ── 服务端连接 ──
    "server_url": "http://127.0.0.1:5000", // 中心服务端地址

    // ── 视频源 ──
    "camera_url": 0,                       // 0=本地摄像头, "rtsp://..."=网络流, "xxx.mp4"=文件

    // ── AI 模型 ──
    "model_path": "runs/detect/fire_detect/weights/best.pt", // YOLOv11 权重路径
    "use_npu": false,                      // 启用 RK3588 NPU 硬件加速
    "rknn_model_path": "models/fire_yolov11.rknn", // RKNN 模型路径

    // ── 检测阈值 ──
    "conf_threshold": 0.35,                // 置信度阈值 (低于此值的框被过滤)
    "iou_threshold": 0.45,                 // NMS IoU 阈值
    "detect_classes": [0, 1],              // 检测类别 [0=fire, 1=smoke], null=全部
    "image_size": 640,                     // NPU 推理输入尺寸

    // ── 告警 ──
    "video_duration": 5,                   // 告警录像时长 (秒)
    "heartbeat_interval": 60,              // 心跳间隔 (秒)
    "save_dir": "alarm_data",              // 告警数据保存目录

    // ── 位置信息 ──
    "location": "重庆理工大学花溪校区",
    "longitude": "106.529813",
    "latitude": "29.452537",

    // ── 实时推流 ──
    "ws_port": 9999                        // WebSocket 端口 (默认自动分配 9990+camera_id)
}
```

> 💡 **提示**：命令行参数 `--camera`、`--model`、`--server`、`--conf` 可覆盖配置文件中的对应项。

---

## 模型训练

### 训练新模型

```bash
cd board

# 1. 自动下载公开火焰数据集并训练
python train.py --download --epochs 80 --model-size n

# 2. 使用自定义数据集
python train.py --data /path/to/data.yaml --epochs 100 --model-size m

# 3. 验证已有模型
python train.py --validate models/fire_yolov11.pt --data-yaml fire_dataset/data.yaml

# 4. 导出 RKNN 格式 (Orange Pi 5 Pro NPU 部署)
python train.py --export-rknn models/fire_yolov11.pt

# 5. 快速测试
python train.py --test models/fire_yolov11.pt --test-image test.jpg
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-size` | `n` | 模型规模: `n`(超轻) / `s`(轻量) / `m`(中) / `l`(大) / `x`(超大) |
| `--epochs` | `80` | 训练轮数 |
| `--imgsz` | `640` | 输入图像尺寸 |
| `--download` | — | 自动从 Roboflow 下载公开火焰数据集 |

### 训练超参数 (来自 `args.yaml`)

| 参数 | 值 | 参数 | 值 |
|------|------|------|------|
| 优化器 | AdamW | 初始学习率 | 0.01 |
| 批次大小 | 30 | 学习率调度 | 余弦退火 |
| 预热轮数 | 3 | 早停耐心 | 15 轮 |
| Mosaic 增强 | 1.0 | MixUp 增强 | 0.1 |
| HSV 增强 | h=0.015, s=0.7, v=0.4 | 随机擦除 | 0.4 |

---

## API 文档

### 设备心跳上报

```http
POST /api/device/heartbeat
Content-Type: application/json

{
    "device_mac": "AAABBBCCCDDD",
    "device_id": 1,
    "camera_id": 1,
    "status": "online",
    "model_info": "YOLOv11-FireDetect",
    "cpu_usage": 35.2,
    "memory_usage": 62.8,
    "websocket_port": 9991,
    "location": "重庆理工大学花溪校区",
    "latitude": "29.452537",
    "longitude": "106.529813"
}
```

**响应：**
```json
{
    "code": 200,
    "msg": "ok",
    "config": {
        "thresh": 0.35,
        "width": 640,
        "height": 480,
        "video_times": 5,
        "heartBeat": 1
    }
}
```

> 服务端根据 heartbeat 间隔判断在线状态：超过 90 秒无心跳 → 自动生成离线故障记录。

### 告警上报

```http
POST /api/alarm
Content-Type: multipart/form-data

device_mac: AAABBBCCCDDD
device_id: 1
camera_id: 1
area_id: 1
longitude: 106.529813
latitude: 29.452537
location: 重庆理工大学花溪校区
timestamp: 2026-07-06T15:30:00
description: 检测到: 火焰 (95.3%), 烟雾 (87.1%)
detections: [{"bbox":[320,240,480,400],"confidence":0.953,"class_id":0,"class_name":"fire","center":[400,320]}]
picture: (binary JPEG)
video: (binary MP4)
```

**响应：**
```json
{
    "code": 200,
    "msg": "ok",
    "alarm_id": 42
}
```

### 设备错误上报

```http
POST /api/device/error
Content-Type: application/json

{
    "device_id": 1,
    "device_mac": "AAABBBCCCDDD",
    "error_code": "算法崩溃",
    "error_msg": "YOLOv11 推理线程异常退出 (exit status 139: segmentation fault)"
}
```

### 摄像头自动发现

```http
POST /api/camera/discover
Content-Type: application/json

{
    "camera_id": 1,
    "ws_port": 9991,
    "ip": "192.168.1.100",
    "location": "花溪校区·东门",
    "camera_name": "东门监控摄像头"
}
```

### 数据统计 (前端仪表盘使用)

```http
GET /api/stats
Authorization: session cookie required

{
    "area": [...],        // 各区域报警分布
    "time_trend": [...],  // 近 30 天趋势
    "category": {...},    // 真报警/误报/漏报统计
    "recent_alarms": [...], // 最新 30 条报警
    "monthly_ranking": [...] // 区域月度排行
}
```

### 告警处理工作流

```
 Status='1'          Status='2'            Status='3'
 (待处理) ──处理──▶ (已处理/待审核) ──审核通过──▶ (已审核)
    ▲                     │                      │
    │                     │ 审核驳回              │
    └─────────────────────┘ (回退至待处理)        │
                                                 ▼
                                            审核完成
```

---

## 部署指南

### 方案一：Orange Pi 5 Pro 边缘部署 (推荐)

```bash
# 1. 刷写 Ubuntu 22.04 系统镜像
# 2. 安装系统依赖
sudo apt update && sudo apt install -y python3-pip ffmpeg

# 3. 安装 Python 依赖 (建议使用虚拟环境)
python3 -m venv venv && source venv/bin/activate
pip install ultralytics opencv-python numpy requests websockets

# 4. 复制项目文件到设备
scp -r board/ orangepi@192.168.1.xxx:/home/orangepi/firecheck/

# 5. (可选) 编译 RKNN 模型用于 NPU 推理
# 注意: NPU 推理帧率约 25 FPS，CPU 推理约 10-15 FPS

# 6. 设置开机自启 (systemd)
sudo cp firecheck.service /etc/systemd/system/
sudo systemctl enable --now firecheck
```

### 方案二：Linux 通用设备部署

```bash
# 与 Orange Pi 5 Pro 部署流程基本一致, 仅需跳过 NPU 相关步骤
# 将配置文件中 use_npu 设为 false 即可使用 CPU 推理
```

### 方案三：服务器端部署

```bash
# 1. 安装 Python 依赖
pip install flask

# 2. 使用 gunicorn 生产部署 (替代 Flask 内置开发服务器)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_server:app

# 3. Nginx 反向代理 (可选)
# 配置 Nginx 指向 gunicorn, 添加 SSL 证书
```

---

## 性能评估

### 模型性能

| 指标 | 数值 |
|------|------|
| 数据集 | Smoke-Fire-Detection-YOLO |
| 训练图片 | 14,122 张 |
| 验证图片 | 3,099 张 |
| mAP@0.5 | **0.751** |
| mAP@0.5:0.95 | **0.436** |
| 精确率 (Precision) | **0.766** |
| 召回率 (Recall) | **0.679** |
| 训练时间 | 2026-06-11 |

### 边缘设备推理性能

| 推理后端 | 分辨率 | FPS | 延迟 |
|----------|--------|------|------|
| PyTorch CPU (RK3588) | 480 | ~10 FPS | ~95ms |
| RKNN NPU (RK3588) | 640 | ~25 FPS | ~40ms |

> ⚡ **隔帧推理优化**：每 5 帧做一次推理，其余帧复用结果。实际 CPU 负载降低 60%+，画面流畅度不受影响。

---

## 开发计划

- [ ] 支持多摄像头并发检测
- [ ] RTSP 断流自动重连
- [ ] 告警推送 (企业微信/钉钉/邮件)
- [ ] Docker 一键部署
- [ ] 移动端告警查看 App
- [ ] 检测模型增量学习 (在线更新)

---

## 许可证

本项目仅供学习研究使用。

---

<div align="center">

**FireCheck** — 让每一缕火光，都在掌控之中 🔥

</div>
