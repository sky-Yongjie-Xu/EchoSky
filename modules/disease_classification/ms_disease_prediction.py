# -*- coding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pydicom
from torch.utils.data import DataLoader
import click
import sys

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.disease_classification.utils import (
    ImageDataset,
    load_model,
    load_image_model,
    mask_outside_ultrasound,
    write_to_avi,
    read_video,
    crop_and_scale,
    EchoDataset,
    process_dicom_image_with_deidentification
)
from tqdm import tqdm

# ====================== 全局配置 ======================
NEURON_NAMES = ["no", "mild", "moderate", "severe"]
NEURON_NAMES_RHEUMATIC = ['non_rheumatic', 'rheumatic']
VIDEO_MODELS = ['PLAX', 'A4C', 'PLAX_D', 'A4C_D']
IMAGE_MODELS = ['MV_CW']
RHEUMATIC_THRESHOLD = 0.06534243532233998

# ====================== 核心推理函数 ======================
def run_inference(view, manifest_path, batch_size, device, weights_dir):
    """Run inference for a given view"""
    # Setup dataset and dataloader
    if view in VIDEO_MODELS:
        test_ds = EchoDataset(split="test", data_path='.', manifest_path=manifest_path, verbose=False)
        model_loader = load_model
    else:
        test_ds = ImageDataset(split="test", data_path='.', manifest_path=manifest_path, verbose=False)
        model_loader = load_image_model
    
    test_dl = DataLoader(test_ds, batch_size=batch_size, drop_last=False, shuffle=False)
    
    # Load models
    ms_model = model_loader(device, weights_path=weights_dir/f"{view}.pt", num_classes=len(NEURON_NAMES))
    
    if view in VIDEO_MODELS:
        second_model = model_loader(device, weights_path=weights_dir/f"{view}_rheumatic.pt", 
                                    num_classes=len(NEURON_NAMES_RHEUMATIC))
        second_names = NEURON_NAMES_RHEUMATIC
        use_softmax_second = True
    else:
        second_model = model_loader(device, weights_path=weights_dir/f"{view}_meanPG.pt", num_classes=1)
        second_names = ['meanpg']
        use_softmax_second = False
    
    # Run inference
    filenames, studyids = [], []
    predictions, predictions_second = [], []
    
    with torch.no_grad():
        for batch in test_dl:
            batch_tensor = batch['primary_input'].to(device)
            
            raw_output = ms_model(batch_tensor)
            predictions.append(torch.softmax(raw_output, dim=1).cpu())
            
            raw_output_second = second_model(batch_tensor)
            if use_softmax_second:
                predictions_second.append(torch.softmax(raw_output_second, dim=1).cpu())
            else:
                predictions_second.append(raw_output_second.cpu())
            
            filenames.extend(batch['filename'])
            studyids.extend([Path(f).parents[1].name for f in batch['filename']])
    
    predictions = torch.cat(predictions, dim=0).T
    predictions_second = torch.cat(predictions_second, dim=0).T
    
    pred_dict = {"filename": filenames, "studyid": studyids, "model": [view] * len(filenames)}
    for key, value in zip(NEURON_NAMES, predictions):
        pred_dict[key] = value
    for key, value in zip(second_names, predictions_second):
        pred_dict[key] = value
    
    return pd.DataFrame(pred_dict)

# ====================== 完整Pipeline ======================
def run_full_pipeline(data_dir, weights_dir, batch_size, device):
    # os.makedirs("predictions", exist_ok=True)
    # os.makedirs("manifest", exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print('🚀 开始超声DICOM全自动推理...')

    # 遍历所有视图
    for view in VIDEO_MODELS + IMAGE_MODELS:
        print(f"\n{'='*60}")
        print(f"处理视图: {view}")
        print(f"{'='*60}")

        if len(list(data_dir.glob(f"*/{view}/*"))) == 0:
            print(f"未找到 {view} 视图DICOM，跳过...")
            studies = [d.name for d in data_dir.iterdir() if d.is_dir()]
            pred_dict = {
                'model': [view] * len(studies),
                'studyid': studies,
                'filename': ['NA'] * len(studies)
            }
            for n in NEURON_NAMES:
                pred_dict[n] = [np.nan] * len(studies)
            if view in VIDEO_MODELS:
                for n in NEURON_NAMES_RHEUMATIC:
                    pred_dict[n] = [np.nan] * len(studies)
            else:
                pred_dict['meanpg'] = [np.nan] * len(studies)
            pd.DataFrame(pred_dict).to_csv(f"modules/disease_classification/outputs/predictions_{view}.csv", index=None)
            continue

        print(f"找到 {len(list(data_dir.glob(f'*/{view}/*')))} 个文件")

        # 获取DICOM路径
        data_path = list(data_dir.glob(f"*/{view}"))
        dcm_paths = [str(file) for f in data_path for file in f.glob('*')]
        studyids = [file.parent.parent.name for f in data_path for file in f.glob('*')]
        base_paths = [str(Path('images') / file.relative_to(data_dir)) for f in data_path for file in f.glob('*')]

        remove_paths = []

        # 视频视图处理
        if view in VIDEO_MODELS:
            file_paths = [str(Path(p).with_suffix('.avi')) if p.endswith('.dcm') else p + '.avi' for p in base_paths]
            for dcm, avi in zip(dcm_paths, file_paths):
                ds = pydicom.dcmread(str(dcm))
                masked_pixel_array = mask_outside_ultrasound(ds.pixel_array)
                if masked_pixel_array.shape[0] < 32:
                    print(f"警告：帧数不足，跳过 {dcm}")
                    remove_paths.append(avi)
                    continue
                Path(avi).parent.mkdir(parents=True, exist_ok=True)
                write_to_avi(masked_pixel_array, avi, fps=30)

        # 图像视图处理
        else:
            file_paths = [str(Path(p).with_suffix('.jpg')) if p.endswith('.dcm') else p + '.jpg' for p in base_paths]
            for dcm, img_path in zip(dcm_paths, file_paths):
                ds = pydicom.dcmread(str(dcm))
                seq = getattr(ds, 'SequenceOfUltrasoundRegions', None)
                ref_pixel_y0 = None

                if seq is not None:
                    if len(seq) > 1:
                        ref_pixel_y0 = getattr(seq[1], 'ReferencePixelY0', None)
                    if ref_pixel_y0 is None:
                        for region in seq:
                            val = getattr(region, 'ReferencePixelY0', None)
                            if val is not None:
                                ref_pixel_y0 = val
                                break

                if ref_pixel_y0 is None:
                    print(f"警告：无ReferencePixelY0，跳过 {dcm}")
                    remove_paths.append(img_path)
                    continue

                if int(ref_pixel_y0) <= 120:
                    print(f"警告：ReferencePixelY0过小，跳过 {dcm}")
                    remove_paths.append(img_path)
                    continue

                transducer_data = getattr(ds, 'TransducerData', None)
                if transducer_data is not None:
                    if isinstance(transducer_data, bytes):
                        transducer_str = transducer_data.decode('utf-8', errors='ignore')
                    else:
                        transducer_str = str(transducer_data)
                    if transducer_str.strip() == 'D2cwc,24,':
                        print(f"警告：不符合TransducerData规则，跳过 {dcm}")
                        remove_paths.append(img_path)
                        continue

                success = process_dicom_image_with_deidentification(dcm, img_path, quality=95)
                if not success:
                    remove_paths.append(img_path)

        # 生成清单
        manifest_df = pd.DataFrame({
            "filename": file_paths,
            "dcm_path": dcm_paths,
            'studyid': studyids,
            "split": ["test"] * len(dcm_paths)
        })
        manifest_df = manifest_df[~manifest_df['filename'].isin(remove_paths)]
        manifest_path = f"./manifest/{view}.csv"
        manifest_df.to_csv(manifest_path, index=None)

        # 推理
        predictions_df = run_inference(view, manifest_path, batch_size, device, weights_dir)
        predictions_df.to_csv(f"modules/disease_classification/outputs/predictions_{view}.csv", index=None)

    # ======================== 集成结果 ========================
    print("\n📊 正在集成所有视图结果...")

    dfs_video = [pd.read_csv(f"./modules/disease_classification/outputs/predictions_{m}.csv").drop_duplicates() for m in VIDEO_MODELS]
    df_video = pd.concat(dfs_video, ignore_index=True)
    df_image = pd.read_csv(f"./modules/disease_classification/outputs/predictions_MV_CW.csv").drop_duplicates()

    df_ms = pd.concat([
        df_video[['filename', 'studyid', 'model'] + NEURON_NAMES],
        df_image[['filename', 'studyid', 'model'] + NEURON_NAMES]
    ], ignore_index=True)

    agg_dict = {name: 'mean' for name in NEURON_NAMES}
    agg_dict['filename'] = list
    wide_ms = df_ms.groupby(["studyid", "model"]).agg(agg_dict).unstack("model")
    wide_ms.columns = [f"{mdl}_{col}" for col, mdl in wide_ms.columns]
    feature_cols = [col for col in wide_ms.columns if not col.endswith('_filename')]

    df_meanpg = df_image[['studyid', 'meanpg']].groupby('studyid').agg({'meanpg': 'max'}).reset_index()
    merged_for_pred = wide_ms.reset_index().merge(df_meanpg, on='studyid', how='left')
    X_ms = merged_for_pred[feature_cols + ['meanpg']].to_numpy()

    with open(f'{weights_dir}/MS_HGB.pkl', 'rb') as f:
        hgb_ms = pickle.load(f)
    wide_ms[NEURON_NAMES] = hgb_ms.predict_proba(X_ms)
    wide_ms['ms_pred_class'] = wide_ms[NEURON_NAMES].idxmax(axis=1)

    df_rheumatic = df_video[['studyid', 'model', 'rheumatic']].groupby(['studyid', 'model'])['rheumatic'].mean()
    wide_rheumatic = df_rheumatic.unstack('model')
    X_rheumatic = wide_rheumatic.loc[:, VIDEO_MODELS].to_numpy()

    with open(f'{weights_dir}/Rheumatic_HGB.pkl', 'rb') as f:
        hgb_rheumatic = pickle.load(f)
    wide_rheumatic['rheumatic_prob'] = hgb_rheumatic.predict_proba(X_rheumatic)[:, 1]
    wide_rheumatic['rheumatic_pred_class'] = np.where(
        wide_rheumatic['rheumatic_prob'] >= RHEUMATIC_THRESHOLD, 'rheumatic', 'non_rheumatic'
    )

    filename_cols = [col for col in wide_ms.columns if col.endswith('_filename')]
    ensemble_df = wide_ms.reset_index()[['studyid'] + NEURON_NAMES + ['ms_pred_class'] + filename_cols].merge(
        wide_rheumatic.reset_index()[['studyid', 'rheumatic_pred_class', 'rheumatic_prob']],
        on='studyid', how='left'
    ).rename(columns={
        'no': 'no_prob',
        'mild': 'mild_prob',
        'moderate': 'moderate_prob',
        'severe': 'severe_prob'
    })

    cols = ['studyid', 'ms_pred_class', 'rheumatic_pred_class', 
            'no_prob', 'mild_prob', 'moderate_prob', 'severe_prob', 'rheumatic_prob'] + filename_cols
    ensemble_df[cols].to_csv('./modules/disease_classification/outputs/ensemble_predictions.csv', index=False)

    print("✅ 推理完成！结果已保存至 modules/disease_classification/outputs/ensemble_predictions.csv")

# ====================== Click 命令行（框架标准） ======================
@click.command("ms_disease_prediction")
@click.option("--data_dir", type=str, required=True, help="DICOM数据集根目录")
@click.option("--weights_dir", type=str, default="./weights", help="模型权重目录")
@click.option("--batch_size", type=int, default=4, help="批次大小")
@click.option("--device", type=str, default="cuda:0", help="运行设备 cuda:0 / cpu")
def run(data_dir, weights_dir, batch_size, device):
    data_dir = Path(data_dir)
    weights_dir = Path(weights_dir)
    run_full_pipeline(data_dir, weights_dir, batch_size, device)

# ====================== 引擎注册（EchoSky框架必备） ======================
def register():
    return {
        "name": "ms_disease_prediction",
        "entry": run,
        "description": "DICOM全自动超声推理（二尖瓣+风湿性+平均压差集成模型）"
    }

if __name__ == "__main__":
    run()