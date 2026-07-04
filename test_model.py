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

# 智能导入 Qt 库，优先使用 PySide6，若不可用则回退到 PyQt5
# 两者的 API 基本一致，仅信号定义略有差异（PySide6 使用 Signal，PyQt5 使用 pyqtSignal）
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
    from PyQt5.QtCore import QThread, pyqtSignal as Signal, Qt  # PyQt5 的信号名不同

# 默认模型路径（需根据实际训练结果调整）
# 该路径指向训练脚本 run_train.py 导出的最佳模型文件
model_path = "/home/value/Keshe/fire/board/runs/detect/fire_detect/weights/best.pt"


class YOLOThread(QThread):
    """通用 YOLO 后台推理线程

    负责在独立线程中执行模型推理，避免阻塞 Qt 主界面，
    支持摄像头、视频文件、单张图片三种数据源。

    信号:
        frame_signal (QImage): 每帧推理完成后发射，携带标注后的图像
        info_signal (str): 状态信息文本，如"摄像头掉线"、"视频播放结束"等
    """

    frame_signal = Signal(QImage)
    info_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False          # 线程运行标志，用于控制循环退出
        self.source_type = None       # 数据源类型: 'camera' / 'image' / 'video'
        self.source_path = None       # 对应的路径或摄像头编号

        # 加载 YOLO 模型，若模型文件不存在则设为 None（后续 run 会提示错误）
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = None

    def set_source(self, source_type, path=0):
        """设置当前要检测的数据源

        参数:
            source_type (str): 数据源类型，'camera'/'image'/'video'
            path: 摄像头编号（int）、图片路径或视频路径
        """
        self.source_type = source_type
        self.source_path = path

    def process_frame(self, frame, start_time=None):
        """对单帧图像执行 YOLO 推理并绘制检测结果

        参数:
            frame (numpy.ndarray): OpenCV BGR 格式图像帧
            start_time (float, 可选): 帧接收时间戳，用于计算实时 FPS

        处理流程:
            1. 调用模型预测，置信度阈值 0.35，输入尺寸 480（兼顾速度与精度）
            2. 使用 results[0].plot() 将检测框和标签绘制到图像上
            3. 若提供了时间戳，计算并显示实时 FPS
            4. 显示检测到的目标数量
            5. 将 BGR 转换为 RGB 并封装为 QImage，通过信号发送到主线程
        """
        results = self.model(frame, conf=0.35, imgsz=480, verbose=False)
        annotated = results[0].plot()  # 在图像上绘制检测框和标签

        # 如果提供了时间戳，计算并显示实时 FPS
        if start_time:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

        # 显示检测到的目标数量
        count = len(results[0].boxes)
        cv2.putText(
            annotated,
            f"Detected: {count}",
            (10, 75 if start_time else 35),  # 有 FPS 显示时向下偏移，避免重叠
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # 将 OpenCV BGR 图像转为 Qt RGB 图像并发射信号到主线程
        rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.frame_signal.emit(qt_img)

    def run(self):
        """QThread 主循环：根据数据源类型执行不同的检测逻辑

        分支处理：
            - image: 单次读取并推理，完成后自动停止
            - camera / video: 循环读取帧，直至用户停止或流结束
        摄像头模式下设置较低延时（10ms）追求高帧率；
        视频模式下根据原始 FPS 计算延时，保持正常播放速度。
        """
        if self.model is None:
            self.info_signal.emit("❌ 模型未找到，请检查路径！")
            return

        self.running = True

        if self.source_type == "image":
            # --- 图片检测模式：单次推理 ---
            self.info_signal.emit(
                f"🖼️ 正在检测图片: {os.path.basename(self.source_path)}"
            )
            frame = cv2.imread(self.source_path)
            if frame is not None:
                self.process_frame(frame)
            self.running = False  # 图片只需处理一次

        else:
            # --- 视频流模式：摄像头或视频文件连续检测 ---
            cap = cv2.VideoCapture(self.source_path)

            if self.source_type == "camera":
                # 摄像头模式：设置分辨率并尽可能快速处理
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.info_signal.emit("📷 摄像头实时检测中...")
                delay = 10  # 延时最小化以实现高帧率
            else:
                # 视频文件模式：按原始帧率播放
                self.info_signal.emit(
                    f"🎞️ 正在播放视频: {os.path.basename(self.source_path)}"
                )
                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = int(1000 / fps) if fps > 0 else 30  # 计算每帧间隔（毫秒）

            while self.running:
                start_time = time.time()
                ret, frame = cap.read()

                if not ret:
                    # 视频播完或摄像头掉线处理
                    if self.source_type == "video":
                        self.info_signal.emit("✅ 视频播放结束。")
                    else:
                        self.info_signal.emit("❌ 摄像头掉线。")
                    break

                self.process_frame(frame, start_time)

                # 延时以控制帧率，同时让出 CPU 给 Qt 主界面渲染
                QThread.msleep(delay)

            cap.release()

    def stop(self):
        """安全停止线程：设置停止标志并等待线程结束"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """火焰烟雾智能监测系统主窗口

    左侧控制面板：摄像头/图片/视频/停止四个操作按钮
    右侧显示面板：实时视频画面和底部状态栏
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔥 YOLOv11 火焰烟雾智能监测系统")
        self.resize(1000, 700)
        self.current_qimage = None  # 缓存当前画面，用于窗口调整大小时重新缩放

        self.init_ui()

        # 初始化后台推理线程并连接信号槽
        self.thread = YOLOThread()
        self.thread.frame_signal.connect(self.update_frame)   # 画面更新
        self.thread.info_signal.connect(self.update_status)   # 状态更新

    def init_ui(self):
        """初始化用户界面布局

        布局结构：
            - 主布局为水平 QHBoxLayout
            - 左侧固定宽度的控制面板（QFrame），包含标题和四个功能按钮
            - 右侧显示面板（QWidget），包含视频显示区域和状态栏
        按钮颜色风格区分功能：摄像头（绿）、图片（蓝）、视频（橙）、停止（红）
        """
        # 中央主挂载部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧控制面板 ---
        control_panel = QFrame()
        control_panel.setFixedWidth(200)
        control_panel.setStyleSheet(
            "background-color: #2b2b2b; color: white; border-radius: 10px;"
        )
        control_layout = QVBoxLayout(control_panel)

        title = QLabel("功能控制台")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        control_layout.addSpacing(20)

        # 按钮通用样式
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

        control_layout.addWidget(self.btn_camera)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_image)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.btn_video)
        control_layout.addStretch()  # 弹性空间将停止按钮推到底部
        control_layout.addWidget(self.btn_stop)

        # --- 右侧显示面板 ---
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)

        self.video_label = QLabel("请在左侧选择检测模式...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888888; font-size: 24px; border-radius: 10px;"
        )

        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet(
            "color: #333; font-size: 14px; font-weight: bold;"
        )
        self.status_label.setFixedHeight(30)

        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.status_label)

        # 组装整体布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel)

    # ------------------ 槽函数（按钮响应） ------------------
    def start_camera(self):
        """启动摄像头实时检测模式"""
        self.thread.stop()
        self.thread.set_source("camera", 0)  # 0 = 系统默认摄像头
        self.thread.start()

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
        """停止当前正在执行的检测任务"""
        self.thread.stop()
        self.video_label.clear()
        self.video_label.setText("已停止检测。")
        self.status_label.setText("系统空闲")

    # ------------------ 信号处理槽 ------------------
    def update_frame(self, qt_img):
        """刷新显示画面：将推理线程传来的 QImage 缩放并显示

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
        """刷新底部状态栏文本

        参数:
            text (str): 状态信息
        """
        self.status_label.setText(text)

    # ------------------ 事件重载 ------------------
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放显示图像，保持比例且不失真"""
        if self.current_qimage is not None:
            self.update_frame(self.current_qimage)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """关闭窗口时安全停止后台推理线程，避免资源泄漏"""
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())