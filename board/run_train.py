#!/usr/bin/env python3
"""
YOLOv11 火焰检测 - 极致精度版 (断点续跑 + 从已有模型继续优化)
直接运行: python3 run_train.py

策略: 加载你已有的 best.pt 作为预训练权重, 用 yolo11m 继续训 200 轮
"""

import os
import torch
import json
import shutil
import yaml  # YAML 读写，用于生成数据集配置文件
from datetime import datetime
from ultralytics import YOLO  # Ultralytics YOLO 训练框架

# 获取当前脚本所在目录作为项目根路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_best_model():
    """查找所有可能的已训练模型文件，优先返回最新的

    按优先级依次检查多个可能路径：
    1. models/fire_yolov11.pt（最终导出模型）
    2. models/fire_yolov11_final.pt（最终版）
    3. runs/detect/fire_detect_pro/weights/best.pt（专业训练结果）
    4. runs/detect/fire_detect/weights/best.pt（基础训练结果）
    5. runs/detect/fire_detect_continue/weights/best.pt（续训结果）

    返回值:
        str 或 None: 找到则返回模型文件路径，否则返回 None
    """
    # 按照优先级从高到低列出候选路径
    candidates = [
        os.path.join(BASE_DIR, "models", "fire_yolov11.pt"),
        os.path.join(BASE_DIR, "models", "fire_yolov11_final.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "fire_detect_pro", "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "fire_detect", "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "fire_detect_continue", "weights", "best.pt"),
    ]
    # 遍历并检查文件是否存在，返回第一个存在的路径
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def detect_gpu_mem():
    """检测当前 GPU 的可用显存和总显存

    返回值:
        tuple: (可用显存GB, 总显存GB)，无 GPU 时返回 (0, 0)
    """
    try:
        # 使用 PyTorch 的 CUDA API 获取显存信息（单位：字节）
        free, total = torch.cuda.mem_get_info(0)
        # 转换为 GB 并返回
        return free / (1024 ** 3), total / (1024 ** 3)
    except Exception:
        # 若获取失败（如无 GPU 或 CUDA 不可用），返回零值
        return 0, 0


def train():
    """YOLOv11 火焰检测极致精度训练

    核心策略：
    1. 自动检测 GPU 显存，据此自适应选择模型大小和批量参数
    2. 支持断点续跑：检测到未完成的训练检查点则继续
    3. 支持从已有模型权重初始化并微调（低学习率）
    4. 训练完成后自动验证并导出评估指标 JSON
    """
    print("=" * 62)
    print("  YOLOv11 火焰检测 - 🔥 极致精度训练 (断点续跑)")
    print("=" * 62)

    # 定义数据集根目录（位于脚本同级目录下的 fire_dataset 文件夹）
    dataset_dir = os.path.join(BASE_DIR, "fire_dataset")

    # ================================================================
    # 第1步: 环境检测 —— 数据集、已有模型、GPU 显存
    # ================================================================
    print("\n[1/4] 检查环境...")

    # 扫描训练集目录（支持多种命名规范）：优先查找常见的子目录名
    train_img = val_img = None
    for d in ["data/train/images", "train/images", "images"]:
        p = os.path.join(dataset_dir, d)
        if os.path.isdir(p) and os.listdir(p):
            train_img = d
            break
    # 扫描验证集目录（支持多种命名规范）
    for d in ["data/val/images", "data/valid/images", "val/images", "valid/images"]:
        p = os.path.join(dataset_dir, d)
        if os.path.isdir(p) and os.listdir(p):
            val_img = d
            break

    # 若找不到训练或验证集，则报错退出
    if not train_img or not val_img:
        print("❌ 找不到数据集!")
        return

    # 统计图片数量
    train_count = len(os.listdir(os.path.join(dataset_dir, train_img)))
    val_count   = len(os.listdir(os.path.join(dataset_dir, val_img)))
    print(f"   训练集: {train_count} 张  |  验证集: {val_count} 张")

    # 查找是否存在之前训练好的模型
    prev_model = find_best_model()
    if prev_model:
        print(f"   ✅ 找到已有模型: {os.path.basename(prev_model)}")

    # 检测 GPU 显存以自适应配置训练参数
    free_gb, total_gb = detect_gpu_mem()
    if total_gb > 0:
        print(f"   GPU 显存: {free_gb:.1f}G 可用 / {total_gb:.1f}G 总计")

    # 根据显存大小自动选择模型规模：显存越大，模型越大、分辨率越高
    # 显存 >= 10GB：使用 YOLO11m（中等模型） + 800x800 分辨率 + 批次 12
    if total_gb >= 10:
        cfg = {"model_name": "yolo11m.pt", "imgsz": 800, "batch": 12}
    # 显存 >= 7GB：使用 YOLO11m + 640x640 分辨率 + 批次 8
    elif total_gb >= 7:
        cfg = {"model_name": "yolo11m.pt", "imgsz": 640, "batch": 8}
    # 显存 >= 5GB：使用 YOLO11s（小模型） + 640x640 + 批次 16
    elif total_gb >= 5:
        cfg = {"model_name": "yolo11s.pt", "imgsz": 640, "batch": 16}
    # 显存 < 5GB：使用 YOLO11n（超轻量模型） + 640x640 + 批次 8
    else:
        cfg = {"model_name": "yolo11n.pt", "imgsz": 640, "batch": 8}

    print(f"   🎯 自动选择: {cfg['model_name']}  imgsz={cfg['imgsz']}  batch={cfg['batch']}")

    # 定义类别映射（0:烟雾, 1:火焰）
    classes = {0: "smoke", 1: "fire"}
    # 生成 YOLO 数据集所需的 data.yaml 文件
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": str(dataset_dir),      # 数据集根目录
            "train": train_img,            # 训练集相对路径
            "val": val_img,                # 验证集相对路径
            "nc": 2,                       # 类别数量
            "names": classes               # 类别名称字典
        }, f, default_flow_style=False, allow_unicode=True)

    # ================================================================
    # 第2步: 确定训练模式 —— 从头训 / 断点续跑 / 从已有模型微调
    # ================================================================
    print("\n[2/4] 确定训练模式...")

    # 定义本次训练的任务名称（用于保存运行结果）
    run_name = "fire_detect_max"
    total_epochs = 80  # 总训练轮数
    # 断点续跑所需的检查点路径（last.pt 保存训练过程中最新状态）
    last_pt    = os.path.join(BASE_DIR, "runs", "detect", run_name, "weights", "last.pt")
    best_pt_ck = os.path.join(BASE_DIR, "runs", "detect", run_name, "weights", "best.pt")
    # 最终模型输出路径
    model_out  = os.path.join(BASE_DIR, "models", "fire_yolov11.pt")
    resume = False   # 是否续跑标志
    model  = None    # 待训练的模型对象
    lr0 = 0.01       # 默认初始学习率

    # 情况 A: 检测到断点检查点，询问是否继续
    if os.path.exists(last_pt):
        try:
            # 加载检查点文件（包含模型权重和训练元信息）
            ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
            # 获取已完成的 epoch 数（通常 key 为 "epoch"）
            done = ckpt.get("epoch", -1) + 1
        except Exception:
            done = "?"  # 若加载失败，标记为未知
        # 如果已完成轮数已达到总轮数，则无需再训，直接复制最优模型
        if isinstance(done, int) and done >= total_epochs:
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            if os.path.exists(best_pt_ck):
                shutil.copy(best_pt_ck, model_out)
            print(f"\n   ✅ {done}轮已训练完成, 无需再训 → {model_out}")
            return
        # 否则提示用户是否继续
        print(f"\n   🔄 发现断点: 已完成 {done}/{total_epochs} 轮")
        print(f"   继续? (Enter=是, n=重头训): ", end="")
        if input().strip().lower() not in ("n", "no"):
            # 若用户选择继续，则加载 last.pt 作为模型，开启续跑模式
            model = YOLO(last_pt)
            resume = True
            lr0 = 0.001  # 续跑时采用较低学习率，避免破坏已学特征
            print(f"   ✅ 从第{done}轮续跑, lr={lr0}")

    # 情况 B: 无断点但存在已有模型 → 使用已有权重初始化并低学习率微调
    if model is None and prev_model:
        print(f"\n   🚀 用 {os.path.basename(prev_model)} 的权重初始化, 低学习率微调")
        print(f"   目标: {total_epochs} 轮, lr=0.001")
        model = YOLO(prev_model)  # 加载已有模型权重
        resume = False
        lr0 = 0.001

    # 情况 C: 无断点也无已有模型 → 全新训练（使用预定义的 YOLO 官方权重）
    if model is None:
        print(f"\n   🆕 全新训练 {cfg['model_name']}  {total_epochs} 轮")
        model = YOLO(cfg["model_name"])  # 从官方预训练模型开始

    # ================================================================
    # 第3步: 执行训练
    # ================================================================
    print(f"\n[3/4] 训练中...")

    # 调用 YOLO 的 train 方法，传入所有训练参数
    results = model.train(
        data=yaml_path,                # 数据集配置文件路径
        epochs=total_epochs,           # 总训练轮数
        imgsz=cfg["imgsz"],            # 输入图像分辨率
        batch=cfg["batch"],            # 批次大小
        device=0,                      # 指定 GPU 设备 0（若为 CPU 可设为 "cpu"）
        workers=4,                     # 数据加载子进程数
        patience=40,                   # 早停耐心值：验证指标 40 轮不提升则停止训练
        lr0=lr0,                       # 初始学习率
        lrf=lr0 * 0.1,                 # 最终学习率 = 初始学习率 × 0.1（学习率衰减终点）
        optimizer="AdamW",             # 使用 AdamW 优化器（对权重衰减更友好）
        warmup_epochs=5,               # 学习率预热轮数
        cos_lr=True,                   # 启用余弦退火学习率调度
        close_mosaic=20,               # 最后 20 轮关闭 Mosaic 增强以提高精度
        augment=True,                  # 启用数据增强
        hsv_h=0.02,                    # HSV 色相增强强度
        hsv_s=0.8,                     # HSV 饱和度增强强度
        hsv_v=0.5,                     # HSV 明度增强强度
        degrees=15.0,                  # 随机旋转角度范围（度）
        translate=0.15,                # 随机平移比例
        scale=0.6,                     # 随机缩放比例
        fliplr=0.5,                    # 水平翻转概率
        mosaic=1.0,                    # Mosaic 增强概率
        mixup=0.2,                     # MixUp 增强概率
        name=run_name,                 # 运行名称，用于保存结果到 runs/detect/run_name
        exist_ok=True,                 # 若同名目录存在则覆盖，不抛出异常
        plots=True,                    # 生成训练曲线等图表
        resume=resume,                 # 是否断点续跑
    )

    # 复制最佳模型到输出目录
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    shutil.copy(best_pt, model_out)
    print(f"   ✅ 模型: {model_out}")

    # ================================================================
    # 第4步: 验证模型性能并保存评估指标
    # ================================================================
    print(f"\n[4/4] 验证指标...")

    # 加载训练好的最佳模型
    m = YOLO(model_out)
    # 在验证集上进行评估
    met = m.val(data=yaml_path, split="val")
    # 提取核心评估指标
    map50   = float(met.box.map50)    # IoU@0.5 的平均精度 (mAP@0.5)
    map_all = float(met.box.map)      # IoU@0.5:0.95 的平均精度 (mAP@0.5:0.95)
    prec    = float(met.box.mp)       # 精确率 (Precision)
    rec     = float(met.box.mr)       # 召回率 (Recall)

    print(f"   mAP50      : {map50:.4f}")
    print(f"   mAP50-95   : {map_all:.4f}")
    print(f"   Precision  : {prec:.4f}")
    print(f"   Recall     : {rec:.4f}")

    # 将评估结果保存为 JSON，便于后续分析和对比
    json.dump({
        "model": cfg["model_name"].replace(".pt", ""),  # 模型名称（去掉扩展名）
        "imgsz": cfg["imgsz"],
        "epochs": total_epochs,
        "train_imgs": train_count,
        "val_imgs": val_count,
        "metrics": {
            "mAP50": map50,
            "mAP50-95": map_all,
            "Precision": prec,
            "Recall": rec
        },
        "time": datetime.now().isoformat(),  # 评估时间戳
    }, open(os.path.join(BASE_DIR, "models", "fire_yolov11_eval.json"), "w"), indent=2)

    # 打印训练完成汇总信息，提示用户后续操作
    print(f"""
    ╔══════════════════════════════════════════╗
    ║              🎉 训练完成!                ║
    ║                                        ║
    ║  模型:  models/fire_yolov11.pt         ║
    ║  测试:  python3 test_model.py          ║
    ║  运行:  python3 main.py                ║
    ╚══════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    train()