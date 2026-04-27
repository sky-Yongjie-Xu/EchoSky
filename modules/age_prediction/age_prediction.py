# -*- coding: utf-8 -*-
import os
import torch
import click
from pytorch_lightning import Trainer
from torchvision.models.video import r2plus1d_18

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.age_prediction.utils import DataLoader, EchoDataset, RegressionModelWrapper


# ==============================================
# 核心推理 pipeline（完全保留你原始逻辑）
# ==============================================
def run_pipeline(
    target,
    manifest_path,
    path_column,
    weights_path,
    save_path,
    num_workers=4,
    batch_size=64,
    gpu_devices=1,
    n_frames=16
):
    print("Initializing dataset and DataLoader...")
    test_ds = EchoDataset(
        path_column=path_column,
        manifest_path=manifest_path,
        targets=[target],
        split="test",
        n_frames=n_frames,
        augmentations=None,
    )

    test_dl = DataLoader(
        test_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    print("Initializing model and loading weights...")
    backbone = r2plus1d_18(num_classes=1)
    model = RegressionModelWrapper(
        backbone,
        output_names=[target],
    )

    weights = torch.load(weights_path, map_location="cpu")
    state_dict = weights.get("state_dict", weights) if isinstance(weights, dict) else weights
    print(model.load_state_dict(state_dict, strict=True))

    print("Running inference...")
    trainer = Trainer(accelerator="gpu", devices=gpu_devices)
    results = trainer.predict(model, dataloaders=test_dl)

    print(f"Saving predictions to {save_path}...")
    model.collate_and_save_predictions(
        results,
        save_path=save_path,
        dataset_manifest=test_ds.manifest,
        fallback_merge_on=path_column,
    )
    print("Inference completed successfully.")


# ==============================================
# 框架标准命令（click）
# ==============================================
@click.command("age_prediction")
@click.option("--target", type=str, required=True, help="预测目标列（如 age）")
@click.option("--manifest_path", type=str, required=True, help="manifest 文件路径")
@click.option("--path_column", type=str, required=True, help="存储视频路径的列名")
@click.option("--weights_path", type=str, required=True, help="模型权重路径")
@click.option("--save_path", type=str, required=True, help="结果保存路径 csv")
@click.option("--num_workers", type=int, default=4, help="DataLoader 线程数")
@click.option("--batch_size", type=int, default=64, help="批次大小")
@click.option("--gpu_devices", type=int, default=1, help="使用 GPU 数量")
@click.option("--n_frames", type=int, default=16, help="采样帧数")
def run(
    target,
    manifest_path,
    path_column,
    weights_path,
    save_path,
    num_workers,
    batch_size,
    gpu_devices,
    n_frames
):
    run_pipeline(
        target=target,
        manifest_path=manifest_path,
        path_column=path_column,
        weights_path=weights_path,
        save_path=save_path,
        num_workers=num_workers,
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        n_frames=n_frames
    )


# ==============================================
# 引擎注册（你框架固定格式）
# ==============================================
def register():
    return {
        "name": "age_prediction",
        "entry": run,
        "description": "超声年龄预测推理（视频）"
    }


if __name__ == "__main__":
    run()