#!/usr/bin/env python3
"""
YOLOv11 火焰检测 Qt 可视化测试平台
支持三种检测模式：摄像头实时检测 / 图片检测 / 视频检测
直接运行: python3 test_model.py

依赖: PySide6 或 PyQt5, OpenCV, Ultralytics YOLO
"""

import sys
import cv2
import time
import os
from ultralytics import YOLO

# ==================== Qt 库智能导入 ====================
# 优先使用 PySide6（LGPL 协议更友好），若不可用则回退到 PyQt5
# 注意：两者 API 基本兼容，主要区别在于信号装饰器和导入路径
try:
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QFrame,
    )
    from PySide6.QtGui import QImage, QPixmap, QFont
    from PySide6.QtCore import QThread, Signal, Qt
except ImportError:
    # PySide6 不可用时回退到 PyQt5
    from PyQt5.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QFileDialog,
        QFrame,
    )
    from PyQt5.QtGui import QImage, QPixmap, QFont
    from PyQt5.QtCore import QThread, pyqtSignal as Signal, Qt  # PyQt5 使用 pyqtSignal 别名

# ==================== 全局配置 ====================
# 默认 YOLO 模型权重路径（按实际训练结果调整，若不存在则后续会提示错误）
model_path = "/home/value/Keshe/fire/board/runs/detect/fire_detect/weights/best.pt"


class YOLOThread(QThread):
    """
    通用 YOLO 后台推理线程

    负责在独立线程中执行模型推理，避免阻塞 Qt 主界面，
    支持摄像头、视频文件、单张图片三种数据源。

    信号:
        frame_signal (QImage): 每帧推理完成后发射，携带标注后的图像
        info_signal (str): 状态信息文本，如"摄像头掉线"、"视频播放结束"等
    """

    # 定义自定义信号，用于跨线程传递图像和状态信息
    frame_signal = Signal(QImage)
    info_signal = Signal(str)

    def __init__(self):
        """初始化推理线程：设置运行标志、数据源类型及路径，并加载模型"""
        super().__init__()
        # 运行标志，用于控制主循环是否继续
        self.running = False
        # 数据源类型: 'camera' / 'image' / 'video'
        self.source_type = None
        # 数据源路径：摄像头编号（int）、图片路径或视频路径
        self.source_path = None

        # 加载 YOLO 模型，若模型文件不存在则设为 None（后续 run 会提示错误）
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = None

    def set_source(self, source_type, path=0):
        """
        设置当前要检测的数据源

        参数:
            source_type (str): 数据源类型，'camera'/'image'/'video'
            path: 摄像头编号（int）、图片路径或视频路径
        """
        self.source_type = source_type
        self.source_path = path

    def process_frame(self, frame, start_time=None):
        """
        对单帧图像执行 YOLO 推理并绘制检测结果

        参数:
            frame (numpy.ndarray): OpenCV BGR 格式图像帧
            start_time (float, 可选): 帧接收时间戳，用于计算实时 FPS
        """
        # 执行推理：置信度阈值 0.35，输入尺寸 480（快速推理），不输出冗余信息
        results = self.model(frame, conf=0.35, imgsz=480, verbose=False)
        # 在图像上绘制检测框、类别标签和置信度
        annotated = results[0].plot()

        # 如果提供了时间戳，计算并显示实时 FPS
        if start_time:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 35),                      # 左上角位置
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,                           # 字体大小
                (255, 255, 0),                 # 青色
                2,
            )

        # 显示当前帧检测到的目标数量
        count = len(results[0].boxes)
        cv2.putText(
            annotated,
            f"Detected: {count}",
            (10, 75 if start_time else 35),    # 有 FPS 时下移避免重叠
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),                       # 绿色
            2,
        )

        # 将 OpenCV BGR 图像转为 Qt RGB 图像格式，然后通过信号发射到主线程
        rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        # 构造 QImage，注意数据拷贝方式（直接使用内存数据，但 QImage 会拷贝一份）
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.frame_signal.emit(qt_img)

    def run(self):
        """
        QThread 主循环：根据数据源类型执行不同的检测逻辑
        此方法在新线程中执行，不应直接调用，应通过 start() 启动
        """
        # 模型未加载时发出错误信息并退出
        if self.model is None:
            self.info_signal.emit("❌ 模型未找到，请检查路径！")
            return

        self.running = True

        # ---------- 图片检测模式：单次推理 ----------
        if self.source_type == "image":
            self.info_signal.emit(
                f"🖼️ 正在检测图片: {os.path.basename(self.source_path)}"
            )
            frame = cv2.imread(self.source_path)
            if frame is not None:
                self.process_frame(frame)
            self.running = False  # 图片只需处理一次，标记结束

        # ---------- 视频流模式：摄像头或视频文件连续检测 ----------
        else:
            # 打开视频捕获对象（摄像头或文件）
            cap = cv2.VideoCapture(self.source_path)

            if self.source_type == "camera":
                # 摄像头模式：设置较低分辨率以提高帧率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.info_signal.emit("📷 摄像头实时检测中...")
                delay = 10  # 延时最小化以实现高帧率（毫秒）
            else:
                # 视频文件模式：按原始帧率播放
                self.info_signal.emit(
                    f"🎞️ 正在播放视频: {os.path.basename(self.source_path)}"
                )
                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps) if fps > 0 else 30  # 计算每帧间隔（毫秒）

            # 主循环：逐帧读取并推理
            while self.running:
                start_time = time.time()  # 记录接收帧的时间，用于 FPS 计算
                ret, frame = cap.read()

                if not ret:
                    # 读取失败：视频播完或摄像头掉线
                    if self.source_type == "video":
                        self.info_signal.emit("✅ 视频播放结束。")
                    else:
                        self.info_signal.emit("❌ 摄像头掉线。")
                    break

                # 处理当前帧（推理 + 绘制）
                self.process_frame(frame, start_time)

                # 延时以控制帧率，同时让出 CPU 给 Qt 主界面渲染
                QThread.msleep(delay)

            # 释放视频资源
            cap.release()

    def stop(self):
        """
        安全停止线程：设置停止标志并等待线程结束
        调用后线程将在下一个循环条件检查时退出
        """
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """
    火焰烟雾智能监测系统主窗口

    左侧控制面板：摄像头/图片/视频/停止四个操作按钮
    右侧显示面板：实时视频画面和底部状态栏
    """

    def __init__(self):
        """初始化主窗口：设置标题、大小、UI、线程及信号连接"""
        super().__init__()
        self.setWindowTitle("🔥 YOLOv11 火焰烟雾智能监测系统")
        self.resize(1000, 700)
        # 缓存当前画面，用于窗口调整大小时重新缩放
        self.current_qimage = None

        self.init_ui()

        # 初始化后台推理线程并连接信号槽
        self.thread = YOLOThread()
        self.thread.frame_signal.connect(self.update_frame)   # 画面更新
        self.thread.info_signal.connect(self.update_status)   # 状态更新

    def init_ui(self):
        """初始化用户界面布局：左侧控制面板 + 右侧显示区域"""
        # 中央主挂载部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ----- 左侧控制面板 -----
        control_panel = QFrame()
        control_panel.setFixedWidth(200)
        control_panel.setStyleSheet(
            "background-color: #2b2b2b; color: white; border-radius: 10px;"
        )
        control_layout = QVBoxLayout(control_panel)

        # 面板标题
        title = QLabel("功能控制台")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        control_layout.addSpacing(20)

        # 按钮通用样式（绿色基底）
        btn_style = """
            QPushButton {
                background-color: #4CAF50; color: white; border: none; 
                padding: 12px; border-radius: 6px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #388E3C; }
        """

        # 摄像头按钮（绿色）
        self.btn_camera = QPushButton("📷 开启摄像头")
        self.btn_camera.setStyleSheet(btn_style)
        self.btn_camera.clicked.connect(self.start_camera)

        # 图片检测按钮（蓝色）
        self.btn_image = QPushButton("🖼️ 选择图片检测")
        self.btn_image.setStyleSheet(
            btn_style.replace("#4CAF50", "#2196F3").replace("#45a049", "#1E88E5")
        )
        self.btn_image.clicked.connect(self.start_image)

        # 视频检测按钮（橙色）
        self.btn_video = QPushButton("🎞️ 选择视频检测")
        self.btn_video.setStyleSheet(
            btn_style.replace("#4CAF50", "#FF9800").replace("#45a049", "#F57C00")
        )
        self.btn_video.clicked.connect(self.start_video)

        # 停止按钮（红色）
        self.btn_stop = QPushButton("🛑 停止当前任务")
        self.btn_stop.setStyleSheet(
            btn_style.replace("#4CAF50", "#F44336").replace("#45a049", "#E53935")
        )
        self.btn_stop.clicked.connect(self.stop_current)

        # 将按钮添加到布局中
        control_layout.addWidget(self.btn_camera)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_image)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_video)
        control_layout.addStretch()  # 弹性空间将停止按钮推到底部
        control_layout.addWidget(self.btn_stop)

        # ----- 右侧显示面板 -----
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)

        # 视频/图片显示标签
        self.video_label = QLabel("请在左侧选择检测模式...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888888; font-size: 24px; border-radius: 10px;"
        )

        # 底部状态栏
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet(
            "color: #333; font-size: 14px; font-weight: bold;"
        )
        self.status_label.setFixedHeight(30)

        # 将显示组件加入布局
        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.status_label)

        # 组装整体布局：左侧控制面板 + 右侧显示面板
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel)

    # ==================== 按钮事件槽函数 ====================
    def start_camera(self):
        """启动摄像头实时检测模式（默认摄像头编号 0）"""
        self.thread.stop()                        # 先停止当前任务
        self.thread.set_source("camera", 0)       # 设置数据源类型和编号
        self.thread.start()                       # 启动线程

    def start_image(self):
        """弹出文件对话框选择图片，启动单张图片检测模式"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要检测的图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.thread.stop()
            self.thread.set_source("image", file_path)
            self.thread.start()

    def start_video(self):
        """弹出文件对话框选择视频文件，启动视频检测模式"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要检测的视频", "", "视频文件 (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.thread.stop()
            self.thread.set_source("video", file_path)
            self.thread.start()

    def stop_current(self):
        """停止当前正在执行的检测任务，并清空显示"""
        self.thread.stop()
        self.video_label.clear()
        self.video_label.setText("已停止检测。")
        self.status_label.setText("系统空闲")

    # ==================== 信号槽接收函数 ====================
    def update_frame(self, qt_img):
        """
        刷新显示画面：将推理线程传来的 QImage 缩放并显示

        参数:
            qt_img (QImage): 推理线程处理并标注后的图像
        """
        self.current_qimage = qt_img  # 缓存原始图像，用于窗口缩放时重绘
        scaled_img = qt_img.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,         # 保持原始宽高比
            Qt.SmoothTransformation,     # 平滑缩放，避免锯齿
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))

    def update_status(self, text):
        """
        刷新底部状态栏文本

        参数:
            text (str): 状态信息
        """
        self.status_label.setText(text)

    # ==================== Qt 事件重写 ====================
    def resizeEvent(self, event):
        """
        窗口大小改变时重新缩放显示图像，保持比例且不失真
        重写 QWidget.resizeEvent
        """
        if self.current_qimage is not None:
            self.update_frame(self.current_qimage)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """
        关闭窗口时安全停止后台推理线程，避免资源泄漏
        重写 QWidget.closeEvent
        """
        self.thread.stop()
        event.accept()


# ==================== 程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())