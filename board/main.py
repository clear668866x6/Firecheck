#!/usr/bin/env python3
"""
后端启动入口 — 火焰/烟雾检测 + 服务端通信
运行时提供交互式菜单，可自定义选择或初始化配置
"""

import os
import sys

# 设置 Qt 平台插件，确保在 Linux 下正常显示 GUI
# 某些基于 Qt 的图形库（如 OpenCV 的 highgui）在 Linux 下需要指定平台插件，
# 设置为 "xcb" 可避免 Wayland 等环境下的兼容性问题。
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 将当前目录加入模块搜索路径，便于导入同目录下的 flame_detect 模块
# 这样即使从其他目录执行本脚本，也能正确找到同级的自定义模块。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traceback
import requests

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    全局异常捕获处理器，用于拦截所有未捕获的异常（除 KeyboardInterrupt 外）。
    功能：
      1. 将异常信息打印到终端（默认行为）。
      2. 尝试将崩溃堆栈信息通过 HTTP 请求上报给 Web 服务端，便于远程监控设备状态。
    参数：
      exc_type: 异常类型
      exc_value: 异常实例
      exc_traceback: 异常堆栈对象
    """
    # 如果是用户主动按 Ctrl+C 触发的 KeyboardInterrupt，则调用默认处理器并返回，
    # 不进行上报，因为这是预期的正常退出行为。
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 首先调用系统默认的异常处理函数，将错误信息输出到 stderr，保证用户能看到错误。
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # 尝试将崩溃堆栈上报给 Web 服务端，用于远程监控和故障诊断。
    # 若上报过程本身出错（如网络不可达），则静默忽略，不影响主程序退出。
    try:
        from flame_detect import Config
        # 构建配置文件路径（与当前脚本同目录下的 flame_config.json）
        config_path = os.path.join(os.path.dirname(__file__), "flame_config.json")
        cfg = Config(config_path)

        # 动态解析服务端 IP：从配置中的 server_url 提取主机名或 IP。
        # 若 server_url 格式为 "http://192.168.1.100:5000" 等，则提取 IP 部分。
        server_ip = "127.0.0.1"
        if "://" in cfg.server_url:
            server_ip = cfg.server_url.split("//")[-1].split(":")[0]

        # 构造错误上报的 API 地址，固定端口 5000，路径为 /api/device/error
        report_url = f"http://{server_ip}:5000/api/device/error"
        # 格式化完整的异常堆栈字符串
        err_stack = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # 准备上报的 JSON 数据，包含设备标识和错误信息。
        # 注意错误消息截取最后 400 个字符，避免数据包过大。
        payload = {
            "device_id": getattr(cfg, "device_id", 1),
            "device_mac": getattr(cfg, "device_mac", "AAABBBCCCDDD"),
            "error_code": "算法崩溃",
            "error_msg": f"Python 异常退出: {exc_value}\n{err_stack[-400:]}"
        }
        # 发送 POST 请求，超时时间 3 秒，避免阻塞退出过程。
        requests.post(report_url, json=payload, timeout=3.0)
    except Exception:
        # 上报过程中任何异常都忽略，避免影响原始异常的处理流程。
        pass

# 将自定义的全局异常处理器设置为系统默认，此后所有未捕获异常都会进入该函数。
sys.excepthook = global_exception_handler

# 从 flame_detect 模块中导入核心类 FlameDetector 和配置类 Config
from flame_detect import FlameDetector, Config

def show_interactive_menu():
    """
    显示交互式启动菜单，允许用户选择启动模式或自定义配置参数。
    用户可以选择：
      - 快速启动（使用现有配置文件）
      - 自定义启动（交互式修改地点、端口、视频源、摄像头ID）
      - 退出系统

    返回值:
        Config: 根据用户交互生成的配置对象（可能修改了部分字段）
    """
    # 从与当前脚本同级目录加载 JSON 配置文件
    config_path = os.path.join(os.path.dirname(__file__), "flame_config.json")
    cfg = Config(config_path)

    # 打印启动菜单，提供清晰的操作指引
    print("\n" + "="*60)
    print("🔥 YOLOv11 火焰烟雾智能监测系统 - 设备端启动平台")
    print("="*60)
    print(" [1] 快速启动 (直接使用默认配置文件参数)")
    print(" [2] 自定义启动 (手动交互式修改地点、端口、视频源)")
    print(" [3] 退出系统")
    print("="*60)
    
    # 获取用户输入，默认为 "1"
    choice = input("请选择操作 [1-3] (默认: 1): ").strip() or "1"
    
    if choice == "3":
        print("已退出系统。")
        sys.exit(0)
        
    if choice == "2":
        # 自定义启动模式：逐项提示用户输入，若不输入则保留原配置值。
        print("\n--- 交互式初始化配置 ---")
        
        # 1. 配置安装地点
        default_loc = cfg.location
        loc_input = input(f"📍 请输入安装地点 (当前默认: '{default_loc}'): ").strip()
        if loc_input:
            cfg._cfg["location"] = loc_input   # 直接修改内部字典，需确保 Config 类支持
            
        # 2. 配置 WebSocket 服务端口（用于实时推流或消息通讯）
        default_port = getattr(cfg, "ws_port", 9999)
        port_input = input(f"🔌 请输入WebSocket服务端口 (当前默认: {default_port}): ").strip()
        if port_input:
            try:
                cfg._cfg["ws_port"] = int(port_input)
            except ValueError:
                print(f"⚠️ 端口格式无效，将采用默认端口: {default_port}")
                
        # 3. 配置视频源（支持摄像头编号、本地视频文件路径、RTSP 流地址）
        default_source = cfg.camera_url
        source_input = input(f"🎥 请输入视频源 (0表示默认摄像头，或输入本地视频路径/RTSP流地址, 默认: '{default_source}'): ").strip()
        if source_input:
            # 尝试转换为整数（如 "0" 表示摄像头 ID），否则保留字符串作为路径或 URL
            try:
                cfg._cfg["camera_url"] = int(source_input)
            except ValueError:
                cfg._cfg["camera_url"] = source_input
                
        # 4. 配置摄像头 ID（用于在服务端唯一标识该摄像头设备）
        default_cam_id = cfg.camera_id
        cam_id_input = input(f"🆔 请输入摄像头ID (当前默认: {default_cam_id}): ").strip()
        if cam_id_input:
            try:
                cfg._cfg["camera_id"] = int(cam_id_input)
            except ValueError:
                print(f"⚠️ 摄像头ID格式无效，将采用默认ID: {default_cam_id}")

    # 打印最终配置摘要，供用户确认启动参数
    print("\n" + "="*60)
    print("🚀 系统初始化配置完成，即将启动检测：")
    print(f"   📍 监控地点: {cfg.location}")
    print(f"   🔌 WebSocket服务端口: {cfg.ws_port}")
    print(f"   🎥 视频源: {cfg.camera_url}")
    print(f"   🆔 摄像头ID: {cfg.camera_id}")
    print(f"   📂 报警数据目录: {cfg.save_dir}")
    print("="*60 + "\n")
    
    return cfg

if __name__ == "__main__":
    import shutil
    try:
        # 显示交互式菜单并获取用户配置对象
        cfg = show_interactive_menu()
        
        # 诊断环境中的 ffmpeg 工具是否可用。
        # ffmpeg 用于将录制的视频转码为 H.264 格式，若不安装则 Web 端可能无法播放。
        if shutil.which("ffmpeg") is None:
            print("\n" + "⚠️ " * 25)
            print(" ⚠️  环境警告 (ENVIRONMENT WARNING):")
            print(" 发现系统未安装 'ffmpeg' 视频编码工具！")
            print(" 这会导致录像视频无法转码为 H.264 格式，导致 Web 端无法直接播放。")
            print(" 请在终端中运行以下命令安装 ffmpeg:")
            print("     👉 sudo apt update && sudo apt install -y ffmpeg")
            print("⚠️ " * 25 + "\n")
            
        # 创建火焰检测器实例（传入配置对象），并启动主循环。
        # FlameDetector 内部会初始化 YOLO 模型、视频捕获、WebSocket 客户端等，
        # 然后进入持续检测和报警处理的主循环。
        detector = FlameDetector(cfg)
        detector.run()
    except KeyboardInterrupt:
        # 用户通过 Ctrl+C 优雅终止，打印提示信息后退出。
        print("\n检测任务已被用户终止。")
        sys.exit(0)