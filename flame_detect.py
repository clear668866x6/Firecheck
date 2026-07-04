#!/usr/bin/env python3
"""
视频AI智能识别及预警管理系统 - 边缘设备火焰检测模块。
功能概述:
  1. 基于 YOLOv11 实现火焰/烟雾实时检测
  2. 支持 NPU (RK3588S) 硬件加速推理
  3. 检测到火焰后自动保存告警图片和录像视频
  4. 通过 HTTP 将告警数据推送到中心 Web 服务端
  5. 内建 WebSocket 服务向外推送实时检测画面
  6. 定时心跳上报设备状态(CPU/内存/在线状态)
部署目标: Orange Pi 5 (RK3588S) / 通用 Linux 设备。
运行方式:
  python flame_detect.py --config flame_config.json
  python flame_detect.py --train fire_dataset/data.yaml
  python flame_detect.py --convert-rknn best.pt
"""

# ---------- 标准库导入 ----------
import os           # 文件系统操作: 路径拼接、文件存在性检查
import sys          # 系统参数, 目前未直接使用, 保留用于扩展
import time         # 时间戳、休眠控制、录像计时
import json         # JSON 序列化/反序列化, 用于配置文件加载和告警数据打包
import uuid         # 生成唯一事件 ID, 确保每次告警有全局唯一标识
import base64       # Base64 编解码, 备用(如 HTTP Basic Auth 或图片编码)
import hashlib      # 哈希摘要, 备用(如文件校验或签名)
import logging      # 结构化日志输出, 同时写入控制台和文件
import threading    # 多线程并发: 心跳线程、WebSocket 服务线程、告警处理线程
import subprocess   # 调用外部进程(ffmpeg 视频转码)
from pathlib import Path         # 面向对象的文件路径操作, 创建目录
from datetime import datetime    # 获取当前时间, 生成格式化时间戳字符串
from io import BytesIO           # 内存中的二进制 I/O, 备用(如内存中处理图片)
from collections import deque    # 定长双端队列: 帧缓冲区和检测历史

# ---------- 第三方库导入 ----------
import cv2                      # OpenCV: 视频流采集、图像处理、视频编码写入
import numpy as np              # 数值计算: 图像数组操作、推理输入预处理
import requests                 # HTTP 客户端: 发送告警数据和心跳到服务端
import websockets               # WebSocket 服务端: 推送实时检测画面帧给前端
from ultralytics import YOLO    # Ultralytics YOLO: YOLOv11 模型加载、训练、推理

# ---------- 日志系统配置 ----------
# 同时输出到控制台和 flame_detect.log 文件, 方便现场排查和长期归档
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("flame_detect.log")]
)
logger = logging.getLogger("FlameDetector")


# ============================================================================
# Config 配置管理类
# 功能: 统一管理所有运行参数, 支持 JSON 文件覆盖默认值, 属性式访问
# 设计: 默认值内置在代码中, 外置配置文件可选择性覆盖, 兼顾开箱即用和灵活定制
# ============================================================================
class Config:
    """
    设备运行配置管理器。
    内部使用字典 _cfg 存储所有配置项, 通过 __getattr__ 实现属性式访问(如 cfg.server_url)。
    支持从 JSON 配置文件加载并覆盖默认值, 未在配置文件中的项保持默认值不变。
    """

    def __init__(self, config_file="flame_config.json"):
        """
        初始化配置, 设置所有默认值, 若配置文件存在则覆盖。
        :param config_file: JSON 配置文件路径, 默认为当前目录下的 flame_config.json
        """
        self._cfg = {
            # ---- 设备标识 ----
            "device_mac": self._get_mac(),  # 自动获取网卡 MAC 地址作为设备唯一标识

            # ---- 服务端通信 ----
            "server_url": "http://127.0.0.1:5000",  # 中心 Web 服务端地址

            # ---- 摄像头 ----
            "camera_url": 0,  # 摄像头地址: 整数=本地摄像头编号, 字符串=RTSP/视频文件路径

            # ---- 模型 ----
            "model_path": os.path.join(os.path.dirname(__file__), "runs", "detect", "fire_detect", "weights", "best.pt"),
            "use_npu": False,  # 是否启用 NPU (瑞芯微 RK3588) 硬件加速推理
            "rknn_model_path": os.path.join(os.path.dirname(__file__), "models", "fire_yolov11.rknn"),

            # ---- 检测参数 ----
            "conf_threshold": 0.35,   # 置信度阈值: 低于此值的检测框将被过滤
            "iou_threshold": 0.45,    # IoU 阈值: NMS 去重时的交并比门限
            "detect_classes": [0, 1], # 仅检测这些类别 ID (0=fire, 1=smoke), None 表示检测所有类别
            "image_size": 640,        # 推理输入图像尺寸(仅 NPU/RKNN 模式使用)

            # ---- 告警参数 ----
            "video_duration": 5,     # 告警录像时长(秒)

            # ---- 心跳 ----
            "heartbeat_interval": 60, # 心跳上报间隔(秒)

            # ---- 存储 ----
            "save_dir": "alarm_data", # 告警数据(图片+视频)本地保存目录

            # ---- 地理信息 ----
            "longitude": "106.551556",
            "latitude": "29.563009",
            "location": "重庆理工大学",

            # ---- 业务标识 ----
            "camera_id": 1,
            "device_id": 1,
            "area_id": 1,

            # ---- WebSocket ----
            "ws_port": 9999,  # WebSocket 服务监听端口, 供前端实时预览
        }
        # 如果配置文件存在, 则从 JSON 加载并合并到默认配置上 (配置文件中的值覆盖默认值)
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                self._cfg.update(json.load(f))
        # 确保告警数据保存目录存在
        Path(self._cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        """
        实现属性式配置访问, 例如 cfg.server_url 等价于 cfg._cfg["server_url"]。
        若配置项不存在则抛出 AttributeError, 避免静默返回 None 导致难以排查的 bug。
        :param name: 配置项名称
        :return: 配置项的值
        :raises AttributeError: 配置项不存在时抛出
        """
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"Config has no attribute: {name}")

    @staticmethod
    def _get_mac():
        """
        获取设备 MAC 地址作为唯一标识符。
        优先读取 eth0 网卡物理地址, 失败则使用 Python 内置 uuid.getnode() 回退方案。
        为什么要用 MAC: 在无固定 IP 的边缘设备上, MAC 是相对稳定的唯一标识,
        服务端据此关联设备、推送配置和统计历史数据。
        :return: 12 位大写十六进制 MAC 字符串(不含分隔符), 如 "AABBCCDDEEFF"
        """
        try:
            # 尝试从 sysfs 读取 eth0 的 MAC 地址 (Linux 标准接口)
            with open("/sys/class/net/eth0/address") as f:
                return f.read().strip().replace(":", "").upper()
        except Exception:
            # 回退: 使用 Python 的 uuid.getnode() 获取硬件地址
            # zfill(12) 确保长度统一, 因为在某些虚拟化环境中返回的地址可能较短
            import uuid as _uuid
            return str(_uuid.getnode()).zfill(12)


# ============================================================================
# FlameDetector 火焰检测器主类
# 核心流程: 模型加载 → 摄像头打开 → 主循环采集帧 → 检测 → 告警处理 → 心跳+WebSocket 并发运行
# 线程模型: 主线程(采集+检测) + 心跳线程 + WebSocket 服务线程 + 告警处理子线程(按需创建)
# ============================================================================
class FlameDetector:
    """
    火焰检测器, 整合视频采集、YOLOv11 推理、告警触发、数据上报、实时推流全部功能。
    典型用法:
        cfg = Config("flame_config.json")
        detector = FlameDetector(cfg)
        detector.run()  # 阻塞运行, Ctrl+C 停止
    """

    def __init__(self, config: Config):
        """
        初始化检测器, 仅保存配置和初始化状态变量, 不加载模型或打开摄像头。
        :param config: Config 配置对象, 包含所有运行参数
        """
        self.cfg = config               # 配置对象引用
        # Dynamically set a unique ws_port based on camera_id if it's default 9999 or None
        ws_port = getattr(config, "ws_port", 9999)
        if ws_port == 9999 or ws_port is None:
            config._cfg["ws_port"] = 9990 + config.camera_id
        self.model = None               # YOLO 模型对象 (PyTorch 或 RKNN)
        self.cap = None                 # OpenCV VideoCapture 摄像头采集对象
        self.running = False            # 主循环运行标志, 设为 False 时所有线程退出

        # ---- 告警冷却机制: 防止同一类目短时间重复告警 ----
        self.alarm_cooldown = {}        # {class_id: last_alarm_timestamp}, 记录每个类目上次告警时间
        self.cooldown_seconds = 15      # 冷却间隔(秒), 同一类目在此时间内不再触发

        # ---- 帧缓冲区: 用于录像回溯 ----
        self.frame_buffer = deque(maxlen=60)  # 保留最近 60 帧, 告警时回写进录像开头, 确保不丢失检测前画面

        # ---- 录像状态 ----
        self.video_writer = None        # OpenCV VideoWriter 对象, 非 None 表示正在录像
        self.recording = False          # 录像状态标志
        self.record_start_time = 0      # 录像开始时间戳, 用于计算录像时长
        self.active_writers = []        # 存放并发活动中的录像写入器 (线程安全)

        # ---- 检测历史: 用于后续扩展平滑滤波(如连续 N 帧确认) ----
        self.detection_history = deque(maxlen=10)  # 最近 10 帧的检测结果

        self.lock = threading.Lock()          # 通用线程锁, 保护可能并发访问的状态
        self._latest_frame = None             # 最新帧数据(已标注), 供 WebSocket 推送
        self._frame_lock = threading.Lock()   # 帧数据专用锁, 减少锁竞争

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load_model(self):
        """
        加载 YOLOv11 火焰检测模型。
        加载策略(按优先级):
          1. 如果启用 NPU 且 RKNN 模型文件存在 → 走 NPU 加载路径
          2. 如果 PyTorch 模型文件存在 → 直接加载 .pt 文件
          3. 都不存在 → 下载官方 yolo11n.pt 基础模型作为回退

        注意: 模型路径支持相对路径, 会自动相对于本脚本所在目录解析。
        """
        model_path = self.cfg.model_path
        # 路径解析: 相对路径 → 相对于脚本所在 board 目录的绝对路径
        if not os.path.isabs(model_path):
            board_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.abspath(os.path.join(board_dir, model_path))
        else:
            resolved_path = model_path

        logger.info(f"Loading YOLOv11 model: {resolved_path}")

        # 优先 NPU: 如果配置启用 NPU 且 RKNN 模型文件存在
        if self.cfg.use_npu and os.path.exists(self.cfg.rknn_model_path):
            self._load_rknn_model(resolved_path)
        # 其次 PyTorch: 直接加载 .pt 文件
        elif os.path.exists(resolved_path):
            self.model = YOLO(resolved_path)
            self.cfg.model_path = resolved_path
        # 最后回退: 下载官方基础模型
        else:
            logger.warning(f"Fire model not found at {resolved_path}, downloading YOLOv11n base model")
            self.model = YOLO("yolo11n.pt")

        # 验证模型是否加载成功, 打印可检测的类别名称
        if hasattr(self, "model") and self.model is not None:
            logger.info(f"Model loaded successfully. Classes detected: {self.model.names}")
        else:
            logger.info("Model loaded")

    def _load_rknn_model(self, resolved_path):
        """
        加载 RKNN 模型到瑞芯微 NPU (如 RK3588 的 NPU_CORE_0)。
        只使用 NPU 核心 0, 保持其他核心空闲给其他任务(如 ISP 图像处理)。
        若 rknnlite 库不可用或初始化失败, 自动回退到 PyTorch CPU/GPU 推理。

        :param resolved_path: PyTorch .pt 模型路径, 仅在回退时使用
        """
        try:
            # 动态导入 rknnlite, 避免在没有 RKNN 环境的机器上报错
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            # 加载预转换好的 .rknn 模型文件
            self.rknn.load_rknn(self.cfg.rknn_model_path)
            # 初始化 NPU 运行时, 仅使用核心 0 (RK3588 有 3 个 NPU 核心)
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                raise RuntimeError("RKNN runtime init failed")
            logger.info("RKNN model loaded on NPU")
        except ImportError:
            # rknnlite 未安装(如开发机), 回退 PyTorch
            logger.warning("rknnlite not available, falling back to PyTorch")
            self.model = YOLO(resolved_path) if os.path.exists(resolved_path) else YOLO("yolo11n.pt")
        except Exception as e:
            # 其他初始化错误(如 NPU 驱动问题), 回退 PyTorch
            logger.error(f"RKNN init failed: {e}, falling back to PyTorch")
            self.model = YOLO(resolved_path) if os.path.exists(resolved_path) else YOLO("yolo11n.pt")

    # ========================================================================
    # 摄像头与帧采集
    # ========================================================================

    def open_camera(self):
        """
        打开摄像头或视频流。
        支持多种输入方式:
          - 整数 (如 0): 本地 USB/CSI 摄像头设备编号
          - 数字字符串 (如 "0"): 自动转为整数
          - RTSP URL (如 "rtsp://..."): 网络摄像头流
          - 文件路径 (如 "test.mp4"): 本地视频文件 (用于离线测试)
        若摄像头无法打开, 自动进入模拟画面模式(is_mock=True), 用于无摄像头的开发调试。
        """
        cam = self.cfg.camera_url
        try:
            # 判断摄像头地址类型: 纯数字 → 本地摄像头索引, 其他 → 视频流 URL/文件路径
            if isinstance(cam, int) or (isinstance(cam, str) and cam.isdigit()):
                self.cap = cv2.VideoCapture(int(cam))
            else:
                self.cap = cv2.VideoCapture(cam)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {cam}")

            # 设置摄像头参数: 分辨率 640x480 和帧率 25fps
            # 注意: 部分摄像头可能不支持这些参数, 设置失败会静默忽略
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25  # 获取实际帧率, 失败则默认 25
            self.is_mock = False  # 标记为真实摄像头模式
            logger.info(f"Camera opened, FPS: {self.fps}")
        except Exception as e:
            # 摄像头不可用时进入模拟模式, 在画布上绘制虚拟目标, 允许无硬件环境测试程序流程
            logger.warning(f"Failed to open camera ({cam}): {e}. Falling back to mock camera stream simulation.")
            self.cap = None
            self.is_mock = True
            self.fps = 25

    # ========================================================================
    # 检测推理
    # ========================================================================

    def detect_frame(self, frame):
        """
        对单帧图像执行火焰/烟雾检测。
        根据配置自动选择推理后端: NPU (RKNN) 或 CPU/GPU (PyTorch)。
        之所以把 imgsz 设为 480 而非默认 640, 是为了在边缘设备上平衡速度和精度。

        :param frame: OpenCV BGR 格式的 numpy 数组, shape (H, W, 3)
        :return: ultralytics Results 对象(包含检测框、置信度等) 或 None
        """
        # NPU 推理路径
        if self.cfg.use_npu and hasattr(self, "rknn"):
            return self._detect_rknn(frame)
        # PyTorch 推理路径: 指定置信度和 IoU 阈值, 限制检测类别, 降低分辨率提速
        results = self.model(frame, conf=self.cfg.conf_threshold, iou=self.cfg.iou_threshold,
                             classes=self.cfg.detect_classes or None, imgsz=480, verbose=False)
        return results[0] if results else None

    def _detect_rknn(self, frame):
        """
        NPU 推理实现(瑞芯微 RK3588)。
        预处理流程: 缩放 → RGB 转换 → 添加 batch 维度 → float32 类型转换 → 送入 NPU 推理。
        NPU 要求固定输入尺寸(由 cfg.image_size 指定, 默认 640x640)。

        :param frame: OpenCV BGR 图像
        :return: NPU 推理原始输出 (后续需自行解析, 格式取决于模型导出时的配置)
        """
        # 1. 缩放到 NPU 要求的固定输入尺寸
        img = cv2.resize(frame, (self.cfg.image_size, self.cfg.image_size))
        # 2. BGR → RGB (YOLO 模型训练时使用 RGB 色彩空间)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. 添加 batch 维度并转为 float32 (NPU 要求的输入格式)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # 4. NPU 推理
        outputs = self.rknn.inference(inputs=[img])
        return outputs

    # ========================================================================
    # 检测结果处理
    # ========================================================================

    def has_flame(self, result):
        """
        判断检测结果中是否包含火焰或烟雾目标。
        仅做快速布尔判断, 用于轻量级检查(如实时状态指示)。

        :param result: ultralytics Results 对象
        :return: True 表示至少有一个检测目标, False 表示无目标或输入为 None
        """
        if result is None:
            return False
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            return True
        return False

    def get_detection_info(self, result, frame_shape):
        """
        从检测结果中提取结构化的检测信息列表。
        将模型原始输出转换为方便后续处理(保存图片标注、打包 JSON 上报服务端)的字典格式。

        :param result: ultralytics Results 对象
        :param frame_shape: 原始帧尺寸 (H, W, C), 用于坐标参考(虽然当前未做尺寸映射)
        :return: 检测信息列表, 每项包含 bbox, confidence, class_id, class_name, center
        """
        if not hasattr(result, "boxes") or result.boxes is None:
            return []
        detections = []
        h, w = frame_shape[:2]
        for box in result.boxes:
            # 提取边界框坐标 (左上角和右下角)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 提取置信度和类别 ID
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # 从模型 names 字典获取类别名称 (如 0 → "fire", 1 → "smoke")
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name,
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],  # 边界框中心点, 用于空间分析
            })
        return detections

    # ========================================================================
    # 告警数据保存: 图片 + 视频
    # ========================================================================

    def save_alarm_image(self, frame, detections, event_id):
        """
        保存告警图片: 在原始帧上绘制检测框和标签后写入文件。
        图片用于服务端快速预览告警内容, 无需打开视频即可判断火情严重程度。

        :param frame: 原始帧图像 (numpy 数组)
        :param detections: get_detection_info() 返回的检测信息列表
        :param event_id: 告警事件唯一 ID
        :return: (filepath, filename) 文件完整路径和文件名
        """
        img = frame.copy()  # 拷贝以避免修改原始帧
        h, w = img.shape[:2]
        # 在图像上绘制每个检测目标的红色边界框和标签
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 165, 255) if det["class_name"] == "smoke" else (0, 0, 255)  # Smoke: Orange, Fire: Red
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 生成带时间戳的文件名: 事件ID_日期时间.jpg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_id}_{ts}.jpg"
        filepath = os.path.join(self.cfg.save_dir, filename)
        cv2.imwrite(filepath, img)
        logger.info(f"Alarm image saved: {filepath}")
        return filepath, filename

    def start_recording(self, frame_w=1280, frame_h=720):
        """
        开始录制告警视频（线程安全）。
        先将帧缓冲区中的历史帧回写到录像开头, 确保视频包含检测触发前的画面上下文。
        返回该次录像的 writer_entry 信息字典。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alarm_{ts}.mp4"
        filepath = os.path.join(self.cfg.save_dir, filename)

        fps = getattr(self, "fps", 20.0)
        if fps <= 0:
            fps = 20.0

        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_w, frame_h))
            if not writer.isOpened():
                raise Exception("avc1 writer not opened")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_w, frame_h))

        # 将帧缓冲区中尺寸匹配的历史帧写入视频开头, 实现"回溯录像"
        for f in list(self.frame_buffer):
            fh, fw = f.shape[:2]
            if fw == frame_w and fh == frame_h:
                try:
                    writer.write(f)
                except Exception as e:
                    logger.error(f"Error writing buffered frame: {e}")
        self.frame_buffer.clear()

        writer_entry = {
            "writer": writer,
            "w": frame_w,
            "h": frame_h,
            "filepath": filepath,
            "filename": filename,
            "start_time": time.time()
        }

        with self._frame_lock:
            self.active_writers.append(writer_entry)
            self.recording = True

        logger.info(f"Recording started with size ({frame_w}x{frame_h}) at {fps} FPS: {filepath}")
        return writer_entry

    def stop_recording(self, writer_entry=None):
        """
        停止录像并释放资源（线程安全）。
        如果传入 writer_entry，则只停止该特定录像；否则停止所有活跃录像。
        停止后调用 ffmpeg 进行 H.264 转码（优先使用系统原生 ffmpeg，并支持 libopenh264 备用）。
        """
        targets = []
        if writer_entry is not None:
            targets = [writer_entry]
        else:
            with self._frame_lock:
                targets = list(self.active_writers)

        last_filepath = ""
        last_filename = ""

        for entry in targets:
            writer = entry["writer"]
            filepath = entry["filepath"]
            filename = entry["filename"]

            try:
                writer.release()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")

            with self._frame_lock:
                if entry in self.active_writers:
                    self.active_writers.remove(entry)
                if not self.active_writers:
                    self.recording = False

            last_filepath = filepath
            last_filename = filename
            logger.info(f"Recording stopped, file: {filepath}")

            # 检查文件是否存在，大小是否大于0
            if filepath and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                temp_path = filepath + ".tmp.mp4"
                
                # 寻找支持完整编码器的 ffmpeg
                ffmpeg_cmd = "ffmpeg"
                for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
                    if os.path.exists(p) and os.access(p, os.X_OK):
                        ffmpeg_cmd = p
                        break

                transcode_success = False
                for encoder in ["libx264", "libopenh264"]:
                    try:
                        cmd = [
                            ffmpeg_cmd, "-y", "-i", filepath,
                            "-vcodec", encoder,
                            "-pix_fmt", "yuv420p",
                            "-profile:v", "baseline",
                            "-level", "3.0",
                            temp_path
                        ]
                        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35)
                        if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            os.replace(temp_path, filepath)
                            logger.info(f"Video transcoded successfully to H.264 via ffmpeg ({encoder})")
                            transcode_success = True
                            break
                        else:
                            err_msg = res.stderr.decode('utf-8', errors='ignore')
                            logger.warning(f"FFmpeg transcode with {encoder} failed (code={res.returncode}): {err_msg[:120]}")
                    except Exception as e:
                        logger.warning(f"FFmpeg transcode exception with {encoder}: {e}")

                    if os.path.exists(temp_path):
                        try: os.remove(temp_path)
                        except: pass

                if not transcode_success:
                    logger.error(f"All H.264 transcode attempts failed for {filepath}. Video remains in raw format.")

        return last_filepath, last_filename

    # ========================================================================
    # 告警上报: 将图片、视频、检测信息通过 HTTP POST 发送到中心服务端
    # ========================================================================

    def send_alarm_to_server(self, image_path, image_filename, video_path, video_filename, detections):
        """
        通过 multipart/form-data 将告警图片和视频上传到服务端 /api/alarm 接口。
        同时携带设备元信息(位置、MAC、相机ID等)和结构化的检测结果 JSON。
        使用 requests 库的 files 参数处理大文件上传, 支持流式传输。

        :param image_path: 告警图片本地路径
        :param image_filename: 图片文件名
        :param video_path: 告警视频本地路径
        :param video_filename: 视频文件名
        :param detections: 检测信息列表
        :return: True 表示上报成功, False 表示失败(网络异常或服务端返回非 200)
        """
        try:
            # 构建 multipart 文件部分: 图片必传, 视频可选(存在则传)
            files = {}
            with open(image_path, "rb") as f:
                files["picture"] = (image_filename, f.read(), "image/jpeg")
            if os.path.exists(video_path):
                with open(video_path, "rb") as f:
                    files["video"] = (video_filename, f.read(), "video/mp4")

            # 生成人类可读的检测描述文本, 如 "检测到: 火焰 (95.3%), 烟雾 (87.1%)"
            detected_items = []
            for d in detections:
                # 将英文类别名映射为中文, 方便在服务端告警页面直接展示
                c_name = "烟雾" if d["class_name"] == "smoke" else ("火焰" if d["class_name"] == "fire" else d["class_name"])
                detected_items.append(f"{c_name} ({d['confidence'] * 100:.1f}%)")
            description = "检测到: " + ", ".join(detected_items)

            # 构建设备元信息表单字段(非文件部分)
            data = {
                "device_mac": self.cfg.device_mac,
                "device_id": self.cfg.device_id,
                "camera_id": self.cfg.camera_id,
                "area_id": self.cfg.area_id,
                "longitude": self.cfg.longitude,
                "latitude": self.cfg.latitude,
                "location": self.cfg.location,
                "detections": json.dumps(detections),  # JSON 序列化检测结果
                "timestamp": datetime.now().isoformat(),  # ISO 8601 时间格式
                "description": description,
            }

            url = f"{self.cfg.server_url}/api/alarm"
            # timeout=30: 视频文件可能较大, 需要较长超时时间
            resp = requests.post(url, data=data, files=files, timeout=30, proxies={"http": None, "https": None})
            if resp.status_code == 200:
                logger.info(f"Alarm sent to server: {resp.json()}")
                return True
            else:
                logger.error(f"Server returned {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send alarm: {e}")
            return False

    # ========================================================================
    # 告警触发决策
    # ========================================================================

    def should_trigger_alarm(self, detections):
        """
        告警冷却判断: 同一类别 ID 在冷却时间内不会重复触发告警。
        为什么要冷却: 连续帧中可能会持续检测到同一目标, 如果不加冷却,
        每一帧都触发告警会导致大量重复告警和录像, 浪费存储和带宽。

        :param detections: 检测信息列表
        :return: True 表示应触发告警, False 表示在冷却中应忽略
        """
        if not detections:
            return False
        now = time.time()
        # 遍历所有检测到的类别, 只要有任意一个类别未在冷却期内, 就触发告警
        for d in detections:
            key = str(d["class_id"])  # 用类别 ID 作为冷却键
            if key not in self.alarm_cooldown or now - self.alarm_cooldown[key] > self.cooldown_seconds:
                # 更新该类别的最新告警时间, 重新开始冷却期
                self.alarm_cooldown[key] = now
                return True
        # 所有类别都在冷却期内, 不触发
        return False

    def process_alarm(self, frame, result):
        """
        告警事件处理主流程(在独立线程中运行):
          1. 提取检测信息
          2. 冷却判断(避免重复告警)
          3. 保存告警图片(带检测框标注)
          4. 录制告警视频(包含检测前后的缓冲区帧)
          5. 将图片和视频上报到中心服务端

        整个过程在单独线程中运行, 不阻塞主检测循环。

        :param frame: 触发告警的当前帧图像 (numpy 数组)
        :param result: ultralytics 检测结果对象
        """
        try:
            detections = self.get_detection_info(result, frame.shape)
            if not detections:
                self.recording = False
                return
            if not self.should_trigger_alarm(detections):
                self.recording = False
                return

            # 生成全局唯一的告警事件 ID: FLAME_日期时间_6位随机Hex
            event_id = f"FLAME_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
            logger.info(f"FLAME DETECTED! Event: {event_id}, Objects: {len(detections)}")

            # 步骤1: 保存告警图片
            image_path, image_filename = self.save_alarm_image(frame, detections, event_id)

            # 步骤2: 开始录制视频, 持续 cfg.video_duration 秒
            h, w = frame.shape[:2]
            writer_entry = self.start_recording(w, h)
            time.sleep(self.cfg.video_duration)  # 等待指定时长采集足够画面
            video_path, video_filename = self.stop_recording(writer_entry)

            # 步骤3: 上报告警到服务端
            success = self.send_alarm_to_server(image_path, image_filename, video_path, video_filename, detections)
            if success:
                logger.info(f"Alarm event {event_id} processed")
            else:
                logger.warning(f"Alarm event {event_id} saved locally but not sent to server")
        except Exception as e:
            logger.error(f"Error in process_alarm: {e}")
            self.recording = False

    # ========================================================================
    # 设备注册与心跳
    # 心跳机制: 定时向服务端报告设备在线状态和资源使用情况
    # 用途: 服务端据此判断设备是否离线、是否需要下发维护指令、统计设备负载
    # ========================================================================

    def register_device(self):
        """
        向服务端发送设备心跳, 携带设备标识、在线状态、CPU/内存使用率、WebSocket 端口等信息。
        服务端通过心跳超时(例如连续 3 次未收到)判定设备离线。
        心跳同时也是一种设备注册: 首次心跳时服务端自动创建设备记录。

        :return: True 表示心跳发送成功, False 表示失败(服务端不可达等)
        """
        try:
            url = f"{self.cfg.server_url}/api/device/heartbeat"
            ws_port = getattr(self.cfg, "ws_port", 9999)
            data = {
                "device_mac": self.cfg.device_mac,
                "device_id": self.cfg.device_id,
                "camera_id": self.cfg.camera_id,
                "timestamp": datetime.now().isoformat(),
                "status": "online",
                "model_info": getattr(self.cfg, "model_info", "YOLOv11"),
                "cpu_usage": self._get_cpu_usage(),       # 实时 CPU 使用率
                "memory_usage": self._get_memory_usage(), # 实时内存使用率
                "websocket_port": ws_port,               # 告知服务端 WebSocket 端口, 供前端连接
                "location": self.cfg.location,
                "latitude": getattr(self.cfg, "latitude", None),
                "longitude": getattr(self.cfg, "longitude", None),
            }
            resp = requests.post(url, json=data, timeout=10, proxies={"http": None, "https": None})  # timeout=10: 心跳超时不宜过长
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    @staticmethod
    def _get_cpu_usage():
        """
        通过 top 命令获取系统 CPU 使用率(仅 Linux)。
        使用 os.popen 行缓冲读取, 解析 top 输出中 Cpu(s) 行的 user 时间百分比。
        返回浮点数百分比(如 35.2 表示 35.2%)。

        :return: CPU 使用率(0-100 的浮点数), 获取失败返回 0.0
        """
        try:
            # top -bn1: 单次批处理模式输出, 避免交互式 top 的输出问题
            # awk '{print $2}' 提取 Cpu(s) 行第二个字段(us, 用户态 CPU 占比)
            return float(os.popen("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'").read().strip().replace(",", ".") or 0)
        except Exception:
            return 0.0

    @staticmethod
    def _get_memory_usage():
        """
        通过读取 /proc/meminfo 获取系统内存使用率(仅 Linux)。
        计算方法: (MemTotal - MemAvailable) / MemTotal × 100。
        使用 MemAvailable 而非 MemFree, 因为前者包含了可回收的缓存/缓冲区,
        更能反映实际可用内存。

        :return: 内存使用率百分比(保留 1 位小数), 获取失败返回 0.0
        """
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            # 第 1 行: MemTotal, 第 3 行: MemAvailable (单位 kB)
            total = int(lines[0].split()[1])
            available = int(lines[2].split()[1])
            return round((1 - available / total) * 100, 1)
        except Exception:
            return 0.0

    def heartbeat_loop(self):
        """
        心跳循环线程: 在 self.running 为 True 期间, 每隔 heartbeat_interval 秒发送一次心跳。
        此方法在独立守护线程中运行, 不阻塞主检测循环。
        """
        while self.running:
            self.register_device()
            time.sleep(self.cfg.heartbeat_interval)

    # ========================================================================
    # WebSocket 实时推流服务
    # 原理: 在独立线程中启动 asyncio 事件循环, 运行 WebSocket 服务端,
    #       每当主循环产生新帧(已标注检测框), 就通过 WebSocket 推送给所有连接的前端页面
    # 帧率控制: asyncio.sleep(0.04) ≈ 25fps, 与摄像头帧率持平
    # ========================================================================

    def _websocket_server(self, port=9999):
        """
        WebSocket 服务端: 接受前端实时预览连接, 持续推送标注后的检测画面帧。
        在独立的 asyncio 事件循环中运行, 通过守护线程启动。

        协议细节:
          - 每 40ms 发送一帧 JPEG 压缩图像(质量 50%, 在画质和带宽之间平衡)
          - 使用 ping_interval/ping_timeout 维持长连接存活, 检测死连接
          - 连接断开时自动清理, 不影响其他连接和新连接

        :param port: 监听端口, 默认 9999
        """
        import asyncio
        import websockets

        async def handler(ws):
            """
            WebSocket 连接处理器, 每个客户端连接对应一个协程实例。
            首帧发送设备元数据 JSON，随后持续推送标注后的检测画面帧。
            """
            logger.info(f"[WS] New connection from {ws.remote_address}")
            try:
                # 发送第一帧元数据 JSON 供前端动态识别
                import json
                meta = {
                    "type": "camera_info",
                    "camera_id": self.cfg.camera_id,
                    "camera_name": self.cfg.location or f"摄像头{self.cfg.camera_id}",
                    "location": self.cfg.location,
                    "area_id": self.cfg.area_id,
                    "ws_port": port
                }
                await ws.send(json.dumps(meta))

                while self.running:
                    # 加锁读取最新帧(线程安全)
                    with self._frame_lock:
                        f = self._latest_frame.copy() if self._latest_frame is not None else None

                    if f is not None:
                        # 压缩为 JPEG, 质量 50%: 减少带宽, 降低推送延迟
                        _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        # 发送二进制 JPEG 数据(比起 base64 编码文本, 更省带宽)
                        await ws.send(jpg.tobytes())

                    # 帧率控制: 0.04s ≈ 25fps
                    await asyncio.sleep(0.04)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"[WS] Connection closed: {ws.remote_address}")
            except Exception as e:
                logger.error(f"[WS] Handler error: {e}")

        async def serve():
            """
            启动 WebSocket 服务并持续监听, 直到程序退出。
            使用 websockets.serve 的上下文管理器确保资源自动释放。
            """
            try:
                async with websockets.serve(handler, "0.0.0.0", port, ping_interval=20, ping_timeout=20):
                    logger.info(f"[WS] Server successfully bound to port {port}")
                    await asyncio.Future()  # 永久阻塞, 保持服务运行
            except Exception as e:
                logger.error(f"[WS] Server failed to start: {e}")

        # 创建新的 asyncio 事件循环(线程安全的), 与主线程的默认循环隔离
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())

    # ========================================================================
    # 帧共享: 将主循环的最新标注帧写入共享变量, 供 WebSocket 线程读取
    # ========================================================================

    def set_latest_frame(self, frame):
        """
        线程安全地设置最新帧(已标注检测框), 供 WebSocket 服务读取和推送。
        使用独立锁 _frame_lock 隔离帧数据的并发访问, 与业务锁 lock 分离, 减少锁竞争。

        :param frame: 已标注检测框的图像帧 (numpy 数组)
        """
        with self._frame_lock:
            self._latest_frame = frame

    # ========================================================================
    # 主运行入口: 启动所有子系统和主检测循环
    # ========================================================================

    def run(self):
        """
        检测器主入口, 执行完整的启动和检测流程:
          1. 加载模型 (YOLOv11 或 RKNN)
          2. 打开摄像头 (或进入模拟模式)
          3. 注册设备 (首次心跳)
          4. 启动心跳线程 (daemon, 后台运行)
          5. 启动 WebSocket 推流服务线程 (daemon, 后台运行)
          6. 进入主循环: 采集帧 → 帧缓冲 → 检测推理 → 标注绘制 → 告警触发
          7. 响应 Ctrl+C 优雅退出

        主循环每秒约处理 25 帧, 每帧执行:
          - 真实模式: 读取摄像头 → 检测 → 标注 → 推送 WebSocket
          - 模拟模式: 生成模拟画面(带网格线、时间戳、移动圆形目标) → 跳过检测
        """
        # ---- 初始化阶段 ----
        self.load_model()
        self.open_camera()
        self.register_device()  # 立即发送一次心跳宣告上线

        self.running = True

        # 启动心跳线程: 定时向服务端上报设备状态
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

        # 启动 WebSocket 服务线程: 提供前端实时预览画面
        ws_port = getattr(self.cfg, "ws_port", 9999)
        threading.Thread(target=self._websocket_server, args=(ws_port,), daemon=True).start()

        logger.info("Flame detection started. Main loop running.")
        frame_count = 0     # 帧计数器, 用于性能统计
        last_print = 0      # 上次打印检测信息的时间戳, 控制打印频率(最多每 2 秒一次)

        try:
            # ---- 主检测循环 ----
            while self.running:
                # ---------- 帧采集 ----------
                if getattr(self, "is_mock", False):
                    # 模拟模式: 在没有摄像头的环境下生成测试画面
                    # 绘制深色背景 + 网格线 + 时间戳 + 位置信息 + 移动圆形模拟目标
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    for x in range(0, 1280, 80):
                        cv2.line(frame, (x, 0), (x, 720), (15, 23, 42), 1)
                    for y in range(0, 720, 80):
                        cv2.line(frame, (0, y), (1280, y), (15, 23, 42), 1)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    cv2.putText(frame, "CAMERA SIMULATION ACTIVE", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (6, 182, 212), 2)
                    cv2.putText(frame, f"TIMESTAMP: {ts}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
                    cv2.putText(frame, f"LOCATION: {self.cfg.location}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)

                    # 移动圆形目标: 沿椭圆轨迹运动, 模拟检测目标
                    angle = (time.time() * 2) % (2 * np.pi)
                    cx = int(640 + 200 * np.cos(angle))
                    cy = int(360 + 100 * np.sin(angle))
                    cv2.circle(frame, (cx, cy), 15, (16, 185, 129), -1)
                    cv2.putText(frame, "SIMULATED TARGET", (cx - 60, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)

                    ret = True
                    time.sleep(1.0 / self.fps)  # 模拟帧率控制
                else:
                    # 真实摄像头模式: 读取一帧
                    ret, frame = self.cap.read()
                    if not ret:
                        # 读取失败处理: 如果是视频文件则回到开头循环播放
                        if self.cap is not None:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                        if not ret:
                            # 仍然失败则短暂休眠后重试(避免 CPU 空转)
                            time.sleep(0.01)
                            continue

                frame_count += 1

                # ---------- 帧缓冲: 保留最近 60 帧供告警录像回溯 ----------
                self.frame_buffer.append(frame.copy())

                # ---------- 检测推理 ----------
                if getattr(self, "is_mock", False):
                    # 模拟模式下不执行实际检测, 避免无模型时的错误
                    result = None
                    detections = []
                else:
                    result = self.detect_frame(frame)
                    detections = self.get_detection_info(result, frame.shape) if result else []

                # ---------- 绘制检测框 ----------
                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    color = (0, 165, 255) if det["class_name"] == "smoke" else (0, 0, 255)  # Smoke: Orange, Fire: Red
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{det['class_name']} {det['confidence']:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # ---------- 录像写入: 如果正在录像, 将绘制了边界框的帧写入所有活跃视频中 ----------
                if self.recording:
                    with self._frame_lock:
                        for entry in list(self.active_writers):
                            try:
                                entry["writer"].write(annotated)
                            except Exception as e:
                                logger.error(f"Error writing frame to safe video writer: {e}")

                # 更新 WebSocket 推送用的最新帧
                self.set_latest_frame(annotated)

                # ---------- 告警触发 ----------
                if detections:
                    # 控制打印频率: 最多每 2 秒打印一次检测信息, 避免刷屏
                    if time.time() - last_print > 2:
                        for det in detections:
                            print(f"  🔥 {det['class_name']} conf={det['confidence']:.2f}")
                        last_print = time.time()
                    # 非录像状态下检测到目标则触发告警处理(在独立线程中执行)
                    if not self.recording:
                        self.recording = True
                        threading.Thread(
                            target=self.process_alarm,
                            args=(frame.copy(), result),
                            daemon=True
                        ).start()

        except KeyboardInterrupt:
            # 用户按 Ctrl+C 优雅退出
            logger.info("Interrupted by user")
        finally:
            # 无论何种退出方式, 都要执行清理
            self.stop()

    def stop(self):
        """
        停止检测器并释放所有资源: 停止录像、释放摄像头、关闭 OpenCV 窗口。
        由 run() 的 finally 块或外部调用触发。
        """
        self.running = False         # 通知所有线程(心跳线程、WebSocket)退出
        if self.recording:
            self.stop_recording()    # 如果有进行中的录像, 先停止
        if self.cap:
            self.cap.release()       # 释放摄像头设备
        cv2.destroyAllWindows()      # 关闭所有 OpenCV 窗口
        logger.info("Flame detector stopped")


# ============================================================================
# 工具函数: 模型训练和格式转换
# ============================================================================

def train_fire_model(data_yaml="fire_dataset/data.yaml", epochs=50, imgsz=640):
    """
    训练 YOLOv11 火焰检测模型。
    基于 yolo11n.pt 预训练权重进行迁移学习, 加速收敛。
    训练参数:
      - patience=10: 验证集损失 10 个 epoch 不下降则早停, 防止过拟合
      - exist_ok=True: 允许覆盖已有训练结果目录
      - 训练结果自动保存在 runs/detect/fire_detect/ 目录下

    :param data_yaml: 数据集配置文件路径 (YOLO 格式, 包含 train/val 路径和类别定义)
    :param epochs: 最大训练轮数
    :param imgsz: 输入图像尺寸(像素)
    :return: ultralytics 训练结果对象, 包含最佳模型路径、训练曲线等
    """
    logger.info("Training YOLOv11 fire detection model...")
    # 加载 YOLOv11 nano 预训练模型作为起点 (nano 版本专为边缘设备优化)
    model = YOLO("yolo11n.pt")
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, patience=10,
                          name="fire_detect", exist_ok=True)
    logger.info(f"Training completed. Best model: {results.save_dir}/weights/best.pt")
    return results


def convert_to_rknn(pt_model_path, rknn_output_path, dataset_txt="dataset.txt"):
    """
    将 PyTorch (.pt) 模型转换为瑞芯微 RKNN 格式, 用于在 NPU 上加速推理。
    转换过程: PyTorch 加载 → 量化校准(使用 dataset.txt 中的样本) → 导出 .rknn

    :param pt_model_path: 输入的 PyTorch 模型路径 (.pt)
    :param rknn_output_path: 输出的 RKNN 模型路径 (.rknn)
    :param dataset_txt: 量化校准图片列表文件, 每行一个图片路径, 用于量化时的统计信息采集
    """
    try:
        # rknn-toolkit2 仅在安装了 RKNN 环境的机器上可用
        from rknn.api import RKNN
        rknn = RKNN()
        # 步骤 1: 加载 PyTorch 模型, 指定输入尺寸 [1, 3, 640, 640]
        ret = rknn.load_pytorch(pt_model_path, input_size_list=[[1, 3, 640, 640]])
        if ret != 0:
            raise RuntimeError("Load PyTorch model failed")
        # 步骤 2: 构建 RKNN 模型, do_quantization=True 启用 INT8 量化以加速和节省内存
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
        if ret != 0:
            raise RuntimeError("Build RKNN model failed")
        # 步骤 3: 导出 .rknn 文件
        ret = rknn.export_rknn(rknn_output_path)
        if ret != 0:
            raise RuntimeError("Export RKNN model failed")
        rknn.release()  # 释放 RKNN 资源
        logger.info(f"RKNN model exported: {rknn_output_path}")
    except ImportError:
        logger.error("rknn-toolkit2 not installed. Install: pip install rknn-toolkit2")
    except Exception as e:
        logger.error(f"RKNN conversion failed: {e}")


# ============================================================================
# 程序入口
# ============================================================================

def main():
    """
    命令行入口函数, 支持三种运行模式:
      1. 训练模式: --train data.yaml     → 训练火焰检测模型
      2. 转换模式: --convert-rknn model.pt → 将 .pt 转为 .rknn 格式
      3. 检测模式: 默认                  → 启动火焰检测器, 加载摄像头和模型进行实时检测

    检测模式下的命令行参数支持:
      --config:  指定配置文件路径
      --camera:  覆盖摄像头地址
      --model:   覆盖模型路径
      --server:  覆盖服务端 URL
      --use-npu: 启用 NPU 加速
      --conf:    覆盖置信度阈值
    """
    import argparse
    parser = argparse.ArgumentParser(description="火焰检测边缘设备")
    parser.add_argument("--config", default="flame_config.json", help="配置文件路径")
    parser.add_argument("--camera", type=str, help="摄像头地址 (RTSP URL 或 设备编号)")
    parser.add_argument("--model", type=str, help="YOLOv11模型路径")
    parser.add_argument("--server", type=str, help="服务端URL")
    parser.add_argument("--train", type=str, help="训练数据集yaml路径")
    parser.add_argument("--convert-rknn", type=str, help="转换模型为RKNN格式, 指定pt路径")
    parser.add_argument("--use-npu", action="store_true", help="使用NPU加速")
    parser.add_argument("--conf", type=float, help="检测置信度阈值")
    args = parser.parse_args()

    # 模式 1: 训练模型
    if args.train:
        train_fire_model(args.train)
        return
    # 模式 2: 转换模型格式
    if args.convert_rknn:
        convert_to_rknn(args.convert_rknn, args.convert_rknn.replace(".pt", ".rknn"))
        return

    # 模式 3: 火焰检测(默认)
    # 加载配置文件, 命令行参数可覆盖配置文件中的对应项
    cfg = Config(args.config if os.path.exists(args.config) else "flame_config.json")
    if args.camera:
        # 摄像头参数: 纯数字转为 int(本地摄像头), 否则保持字符串(URL/路径)
        cfg._cfg["camera_url"] = int(args.camera) if args.camera.isdigit() else args.camera
    if args.model:
        cfg._cfg["model_path"] = args.model
    if args.server:
        cfg._cfg["server_url"] = args.server
    if args.use_npu:
        cfg._cfg["use_npu"] = True
    if args.conf is not None:
        cfg._cfg["conf_threshold"] = args.conf

    detector = FlameDetector(cfg)
    detector.run()  # 阻塞运行, Ctrl+C 终止


if __name__ == "__main__":
    main()
