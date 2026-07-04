#!/usr/bin/env python3
"""
YOLOv11火焰检测模型训练脚本
支持: 自动下载公开火焰数据集 / 使用自定义数据集

功能模块：
- 数据集自动下载（Roboflow / Kaggle 备用）
- 本地数据集准备（含最小样本集用于流程验证）
- 自动生成 YOLO 格式的 data.yaml
- 模型训练（支持 n/s/m/l/x 规模，多种数据增强策略）
- 训练后自动验证并导出评估指标
- 支持导出 ONNX 和 RKNN（用于 NPU 部署）
- 单张图片或摄像头实时测试推理

使用方式：
    python train.py --help
    python train.py --download                     # 自动下载公开数据集并训练
    python train.py --data my_data.yaml --epochs 80
    python train.py --validate model.pt --data-yaml data.yaml
    python train.py --export-rknn model.pt
    python train.py --test model.pt --test-image test.jpg
"""

import os
import sys
import json
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime

import yaml  # 用于读写 YOLO 格式的数据集配置文件
from ultralytics import YOLO  # Ultralytics YOLO 框架，用于训练和推理
import numpy as np

# 配置日志输出格式：时间戳 + 日志级别 + 消息内容
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train")

# 项目目录常量定义
BASE_DIR = Path(__file__).parent  # 当前脚本所在的目录
DATASET_DIR = BASE_DIR / "fire_dataset"  # 数据集存放目录
MODEL_DIR = BASE_DIR / "models"  # 训练好的模型导出目录
MODEL_DIR.mkdir(exist_ok=True)  # 如果不存在则创建


def download_fire_dataset():
    """从 Roboflow 公共平台下载公开火焰检测数据集（约 1500 张标注图片）

    返回值:
        bool: 下载成功返回 True，失败返回 False
    """
    logger.info("正在下载火焰检测数据集...")
    try:
        from roboflow import Roboflow
        # 公开数据集无需填写 API key 也可访问
        rf = Roboflow(api_key="")
        # 依次尝试多个公开火焰数据集，任一成功即停止
        datasets_to_try = [
            ("fire-wrnre", 1),      # Fire Detection Dataset
            ("fire-detection-e0jso", 1),
            ("forest-fire-fbwdy", 1),
            ("wildfire-detection", 1),
        ]
        for ds_name, version in datasets_to_try:
            try:
                logger.info(f"尝试下载: {ds_name}")
                project = rf.workspace("public").project(ds_name)
                dataset = project.version(version).download("yolov11")
                logger.info(f"数据集已下载: {DATASET_DIR}")
                return True
            except Exception:
                continue  # 当前数据集下载失败，尝试下一个
    except ImportError:
        logger.warning("roboflow 未安装，使用本地数据集方式")
    except Exception as e:
        logger.warning(f"Roboflow下载失败: {e}")

    logger.info("尝试从Kaggle镜像下载...")
    return _download_kaggle_fire()


def _download_kaggle_fire():
    """备用下载方案：尝试从 Kaggle 获取火焰数据集（当前为占位实现）"""
    logger.info("使用内置数据集创建流程...")
    return False


def create_mini_fire_dataset():
    """创建最小火焰数据集，用于快速验证训练流程是否正常

    从公开免费图片源自动采集并生成 YOLO 格式标注，
    如果自动下载失败则打印手动准备数据的说明。

    返回值:
        bool: 始终返回 False（由于真实图片需要手动准备）
    """
    import urllib.request
    import cv2

    logger.info("创建火焰样本数据集 (用于验证训练流程)...")

    # 创建 YOLO 格式所需的标准目录结构
    DATASET_DIR.mkdir(exist_ok=True)
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    # 示例图片 URL（非火焰图片，仅用于目录结构演示）
    fire_images = [
        ("https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", "bus"),
        ("https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg", "zidane"),
    ]

    # 数据下载失败时，打印详细的手动准备指引
    logger.warning("=" * 60)
    logger.warning("未自动下载到数据集，你需要手动准备火焰数据:")
    logger.warning("")
    logger.warning("方式1: 下载Roboflow公开数据集")
    logger.warning("  pip install roboflow")
    logger.warning("  python -c \"from roboflow import Roboflow; rf=Roboflow();")
    logger.warning("  rf.workspace('public').project('fire-wrnre').version(1).download('yolov11')\"")
    logger.warning("")
    logger.warning("方式2: 从Kaggle下载")
    logger.warning("  kaggle datasets download -d atulyakumar98/fire-dataset")
    logger.warning("  kaggle datasets download -d dataclusterlabs/fire-detection-dataset")
    logger.warning("")
    logger.warning("方式3: 自行收集火焰图片并标注 (≥100张)")
    logger.warning("  图片放入: fire_dataset/train/images/")
    logger.warning("  标注放入: fire_dataset/train/labels/ (YOLO格式)")
    logger.warning("  验证集放入: fire_dataset/val/")
    logger.warning("=" * 60)

    return False


def prepare_dataset_yaml():
    """生成 YOLO 训练所需的数据集配置文件 (data.yaml)

    该配置文件定义了训练集/验证集路径、类别数量和名称。

    返回值:
        str: 生成的 data.yaml 文件的绝对路径
    """
    yaml_path = DATASET_DIR / "data.yaml"

    # 配置数据集路径、训练/验证/测试集子目录、类别信息
    data_config = {
        "path": str(DATASET_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "val/images",  # 测试集复用验证集
        "nc": 2,  # 类别数量: fire 和 smoke
        "names": {
            0: "fire",
            1: "smoke"
        }
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"数据集配置: {yaml_path}")
    logger.info(f"类别: fire(火焰), smoke(烟雾)")
    return str(yaml_path)


def train_model(data_yaml, model_size="n", epochs=80, imgsz=640):
    """训练 YOLOv11 火焰检测模型

    参数:
        data_yaml (str): 数据集 data.yaml 配置文件路径
        model_size (str): 模型规模，可选 n(超轻)/s(轻量)/m(中等)/l(大)/x(超大)
        epochs (int): 训练总轮数，默认 80
        imgsz (int): 输入图片尺寸（像素），默认 640

    返回值:
        Path: 训练完成后导出模型的保存路径
    """
    model_name = f"yolo11{model_size}.pt"  # 例如: yolo11n.pt
    logger.info(f"开始训练: YOLOv11{model_size}, epochs={epochs}, imgsz={imgsz}")

    # 加载预训练模型
    model = YOLO(model_name)

    # 执行训练，使用数据增强和多种优化策略
    results = model.train(
        data=data_yaml,          # 数据集配置
        epochs=epochs,           # 训练轮数
        imgsz=imgsz,             # 输入图片尺寸
        batch=8,                 # 批量大小
        patience=15,             # 早停轮数：验证集指标 15 轮不提升则提前停止
        lr0=0.01,                # 初始学习率
        lrf=0.01,                # 最终学习率因子（lr0 * lrf = 最终学习率）
        optimizer="AdamW",       # 优化器：带权重衰减的 Adam
        warmup_epochs=3,         # 学习率预热轮数
        cos_lr=True,             # 使用余弦退火学习率调度
        augment=True,            # 启用数据增强
        hsv_h=0.015,             # HSV-色调增强幅度
        hsv_s=0.7,               # HSV-饱和度增强幅度
        hsv_v=0.4,               # HSV-明度增强幅度
        degrees=10.0,            # 随机旋转角度范围
        translate=0.1,           # 随机平移比例
        scale=0.5,               # 随机缩放比例
        fliplr=0.5,              # 水平翻转概率
        mosaic=1.0,              # Mosaic 增强概率（4 张图拼接成 1 张）
        mixup=0.1,               # MixUp 增强概率（2 张图混合）
        name="fire_detect",      # 训练任务名称（保存目录名）
        exist_ok=True,           # 允许覆盖之前的同名训练结果
        verbose=True,            # 输出详细训练日志
        plots=True,              # 生成训练曲线图
    )

    # 复制训练完成后的最佳模型到导出目录
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    export_path = MODEL_DIR / "fire_yolov11.pt"
    shutil.copy(best_pt, export_path)
    logger.info(f"模型已保存: {export_path}")

    # 同步保存模型元信息（类型、类别、训练日期等）
    config_path = MODEL_DIR / "fire_yolov11.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_type": f"yolo11{model_size}",
            "classes": ["fire", "smoke"],
            "input_size": imgsz,
            "train_date": datetime.now().isoformat(),
            "dataset": data_yaml,
        }, f, indent=2)

    return export_path


def validate_model(model_path, data_yaml):
    """验证已训练模型的性能指标

    参数:
        model_path (str): 训练好的 .pt 模型文件路径
        data_yaml (str): 数据集配置文件路径，用于定位验证集

    返回值:
        dict: 包含 mAP50, mAP50-95, precision, recall 的字典
    """
    logger.info("验证模型性能...")
    model = YOLO(model_path)
    # 在验证集上运行评估
    metrics = model.val(data=data_yaml, split="val")

    # 提取关键评估指标
    results = {
        "mAP50": float(metrics.box.map50),      # IoU=0.5 时的平均精度
        "mAP50-95": float(metrics.box.map),      # IoU 从 0.5 到 0.95 的平均精度
        "precision": float(metrics.box.mp),      # 精确率（查准率）
        "recall": float(metrics.box.mr),         # 召回率（查全率）
    }
    logger.info(f"验证结果: mAP50={results['mAP50']:.3f}, "
                f"mAP50-95={results['mAP50-95']:.3f}, "
                f"Precision={results['precision']:.3f}, "
                f"Recall={results['recall']:.3f}")
    return results


def export_rknn(model_path, output_path=None):
    """将 PyTorch 模型导出为 RKNN 格式，用于部署到 Orange Pi 5 NPU

    流程: .pt → .onnx → .rknn

    参数:
        model_path (str): PyTorch 模型文件路径 (.pt)
        output_path (str, 可选): RKNN 模型输出路径，默认保存在 models/ 目录
    """
    if output_path is None:
        output_path = MODEL_DIR / "fire_yolov11.rknn"

    logger.info("尝试导出RKNN模型...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        # 第一步：导出 ONNX 中间格式
        onnx_path = MODEL_DIR / "fire_yolov11.onnx"
        model.export(format="onnx", imgsz=640, simplify=True, opset=12)
        logger.info(f"ONNX已导出: {onnx_path}")

        try:
            # 第二步：通过 RKNN Toolkit2 将 ONNX 转为 RKNN
            from rknn.api import RKNN
            rknn = RKNN()
            # 配置归一化参数和目标平台
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform="rk3588")
            ret = rknn.load_onnx(str(onnx_path))
            if ret != 0:
                raise RuntimeError("ONNX加载失败")
            # 构建 RKNN 模型（启用量化以减小模型体积、加速推理）
            ret = rknn.build(do_quantization=True)
            if ret != 0:
                raise RuntimeError("RKNN构建失败")
            ret = rknn.export_rknn(str(output_path))
            if ret != 0:
                raise RuntimeError("RKNN导出失败")
            rknn.release()
            logger.info(f"RKNN模型已导出: {output_path}")
        except ImportError:
            logger.warning("rknn-toolkit2未安装，跳过RKNN导出")
            logger.warning("部署时在PC上运行: pip install rknn-toolkit2")
            logger.warning("然后执行: python train.py --export-rknn fire_yolov11.pt")
    except Exception as e:
        logger.error(f"RKNN导出失败: {e}")
        logger.info("ONNX模型仍可用于CPU推理")


def test_model(model_path, image_path=None):
    """使用训练好的模型对单张图片或摄像头进行测试推理

    参数:
        model_path (str): 训练好的模型文件路径
        image_path (str, 可选): 待检测图片路径。若未指定则调用摄像头实时检测
    """
    model = YOLO(model_path)

    if image_path and os.path.exists(image_path):
        # 单张图片推理模式
        results = model(image_path)
        for result in results:
            if result.boxes:
                logger.info(f"检测到 {len(result.boxes)} 个目标:")
                for box in result.boxes:
                    cls_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    logger.info(f"  {cls_name}: conf={conf:.3f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                # 保存标注后的结果图片
                save_path = MODEL_DIR / "test_result.jpg"
                result.save(str(save_path))
                logger.info(f"结果图片: {save_path}")
            else:
                logger.info("未检测到火焰")
    else:
        # 摄像头实时推理模式（按 'q' 键退出）
        logger.info("使用电脑摄像头实时测试 (按q退出)...")
        import cv2
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 以 0.35 置信度阈值进行推理
            results = model(frame, conf=0.35, verbose=False)
            annotated = results[0].plot()  # 绘制检测框
            cv2.imshow("Fire Detection Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    """命令行入口：解析参数并执行训练/验证/导出/测试操作"""
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv11火焰检测模型训练")
    parser.add_argument("--download", action="store_true", help="自动下载公开火焰数据集")
    parser.add_argument("--data", type=str, help="自定义数据集data.yaml路径")
    parser.add_argument("--model-size", type=str, default="n", choices=["n","s","m","l","x"],
                        help="模型大小 (n=超轻/s=轻量/m=中/l=大/x=超大)")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--validate", type=str, help="验证已有模型, 指定模型路径")
    parser.add_argument("--data-yaml", type=str, help="验证/导出时指定的data.yaml")
    parser.add_argument("--export-rknn", type=str, help="导出RKNN, 指定pt模型路径")
    parser.add_argument("--test", type=str, help="测试模型, 指定pt模型路径")
    parser.add_argument("--test-image", type=str, help="测试图片路径")
    args = parser.parse_args()

    # 测试模式：对图片或摄像头进行推理
    if args.test:
        test_model(args.test, args.test_image)
        return

    # 验证模式：评估已有模型性能
    if args.validate:
        data = args.data_yaml or str(DATASET_DIR / "data.yaml")
        if not os.path.exists(data):
            logger.error(f"数据集配置文件不存在: {data}")
            logger.error("请先训练或指定 --data-yaml 参数")
            sys.exit(1)
        validate_model(args.validate, data)
        return

    # 导出模式：转为 RKNN 格式
    if args.export_rknn:
        export_rknn(args.export_rknn)
        return

    # 训练模式：确定数据集来源
    if args.data:
        data_yaml = args.data
    elif args.download:
        # 先下载数据集，再生成 yaml 配置
        download_fire_dataset()
        data_yaml = prepare_dataset_yaml()
    else:
        # 查找是否已有本地数据集
        if (DATASET_DIR / "data.yaml").exists():
            data_yaml = str(DATASET_DIR / "data.yaml")
            logger.info(f"找到已有数据集: {data_yaml}")
        else:
            logger.info("未指定数据集，尝试自动创建...")
            created = download_fire_dataset()
            if not created:
                created = create_mini_fire_dataset()
            data_yaml = prepare_dataset_yaml()

    # 确认数据集配置文件存在，否则退出
    if not os.path.exists(data_yaml):
        logger.error(f"数据集配置不存在: {data_yaml}")
        logger.error("请按以下步骤准备数据:")
        logger.error("  1. 收集火焰图片(≥100张)放入 fire_dataset/train/images/")
        logger.error("  2. 使用LabelImg/LabelStudio标注为YOLO格式,放入 fire_dataset/train/labels/")
        logger.error("  3. 同样准备验证集到 fire_dataset/val/")
        logger.error("  4. 重新运行: python3 train.py")
        sys.exit(1)

    # 验证训练集非空
    train_files = list((DATASET_DIR / "train" / "images").glob("*"))
    if not train_files:
        logger.error("训练集为空! 请加入火焰图片后再训练")
        sys.exit(1)

    logger.info(f"训练集图片数: {len(train_files)}")
    # 执行训练
    model_path = train_model(data_yaml, args.model_size, args.epochs, args.imgsz)

    # 训练完成后自动验证模型性能
    validate_model(model_path, data_yaml)

    logger.info("训练完成!")
    logger.info(f"模型位置: {model_path}")
    logger.info("部署方法: 将 fire_yolov11.pt 复制到 board/ 目录")
    logger.info("  然后运行: cd board && python3 flame_detect.py --model fire_yolov11.pt")


if __name__ == "__main__":
    main()