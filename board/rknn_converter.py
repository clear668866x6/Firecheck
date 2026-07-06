#!/usr/bin/env python3
"""
YOLOv11 .pt -> .onnx -> .rknn 转换工具
注意：RKNN-Toolkit2 开发套件仅支持在 x86_64 架构的 Ubuntu PC 上运行，不能直接在香橙派上运行转换。
转换完成后，将生成的 .rknn 文件拷贝至香橙派即可部署。
"""

import os
import sys

# 定义转换函数，接收 PyTorch 模型路径、目标平台（默认为 rk3588）和输出 rknn 文件路径（可选）
def convert(pt_path, target_platform="rk3588", output_rknn_path=None):
    # 检查输入的 .pt 文件是否存在
    if not os.path.exists(pt_path):
        print(f"❌ 找不到 PyTorch 模型文件: {pt_path}")
        return

    print("="*60)
    print(f"📦 步骤 1: 正在使用 Ultralytics 将 {pt_path} 导出为 ONNX 格式...")
    print("="*60)
    
    try:
        # 导入 ultralytics 库中的 YOLO 类
        from ultralytics import YOLO
        # 加载 YOLO 模型（.pt 文件）
        model = YOLO(pt_path)
        # 导出为 ONNX，设置图像尺寸为 640，简化模型，opset=12（RKNN 推荐版本）
        onnx_name = os.path.splitext(pt_path)[0] + ".onnx"
        model.export(format="onnx", imgsz=640, simplify=True, opset=12)
        print(f"✅ ONNX 模型导出成功: {onnx_name}")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        return

    print("\n" + "="*60)
    print(f"📦 步骤 2: 正在将 {onnx_name} 转换为 {target_platform} 的 RKNN 格式...")
    print("="*60)

    # 如果未指定输出路径，则使用与输入 .pt 同名的 .rknn 文件
    if output_rknn_path is None:
        output_rknn_path = os.path.splitext(pt_path)[0] + ".rknn"

    try:
        # 导入 RKNN 核心 API
        from rknn.api import RKNN
        # 创建 RKNN 实例
        rknn = RKNN()
        
        # 1. 配置模型：设定输入均值、标准差和目标芯片平台
        print("-> 配置 RKNN 参数...")
        rknn.config(
            mean_values=[[0, 0, 0]],      # RGB 均值归一化值，YOLOv11 推荐 [0,0,0] 配合 std 255
            std_values=[[255, 255, 255]], # 归一化系数
            target_platform=target_platform  # 目标平台，如 rk3588 或 rk3568
        )
        
        # 2. 导入 ONNX 模型
        print(f"-> 导入 ONNX 模型: {onnx_name}...")
        ret = rknn.load_onnx(model=onnx_name)
        if ret != 0:
            print("❌ 读取 ONNX 模型失败")
            return
            
        # 3. 编译模型：可以选择开启量化 (do_quantization=True)，会让模型变小且在 NPU 上跑得更快
        print(f"-> 编译 RKNN 模型 (目标平台: {target_platform})...")
        ret = rknn.build(do_quantization=True)  # 默认开启量化加速
        if ret != 0:
            print("❌ RKNN 编译失败")
            return
            
        # 4. 导出为 RKNN 文件
        print(f"-> 导出目标模型: {output_rknn_path}...")
        ret = rknn.export_rknn(output_rknn_path)
        if ret != 0:
            print("❌ RKNN 模型导出失败")
            return
            
        # 释放 RKNN 资源
        rknn.release()
        print("\n" + "="*60)
        print(f"🎉 转换成功！生成的 RKNN 模型保存在:")
        print(f"👉 {output_rknn_path}")
        print("部署说明: 将此文件复制到香橙派的 board/models/ 目录下，")
        print("并在 board/flame_config.json 中配置 \"use_npu\": true 和 \"rknn_model_path\" 即可使用 NPU 加速！")
        print("="*60)

    except ImportError:
        # 捕获导入错误，提示用户未安装 rknn-toolkit2 开发套件
        print("\n⚠️ 无法导入 'rknn.api' (未安装 rknn-toolkit2 开发套件)")
        print("说明：转换工具需要在电脑 PC (x86_64 Ubuntu) 端运行，请确保您在电脑端执行了：")
        print("     pip install rknn-toolkit2")
        print(f"您生成的 ONNX 模型 '{onnx_name}' 已就绪，可以直接拷贝到带有 rknn-toolkit2 的电脑上完成最后一步转换。")

# 当脚本作为主程序运行时，解析命令行参数并执行转换
if __name__ == "__main__":
    import argparse
    # 创建命令行参数解析器，描述程序功能
    parser = argparse.ArgumentParser(description="YOLOv11 pt 转 rknn 转换器")
    # 必需参数：输入的 .pt 模型路径
    parser.add_argument("model", type=str, help="输入的 .pt 模型路径 (例如: yolo11n.pt)")
    # 可选参数：目标平台，默认为 rk3588，可选 rk3568
    parser.add_argument("--platform", type=str, default="rk3588", choices=["rk3588", "rk3568"], 
                        help="目标平台芯片 (默认: rk3588, 香橙派5使用 rk3588)")
    # 可选参数：输出的 .rknn 路径
    parser.add_argument("--output", type=str, help="输出的 .rknn 路径 (默认同名)")
    # 解析参数
    args = parser.parse_args()
    
    # 调用转换函数，传入解析后的参数
    convert(args.model, args.platform, args.output)