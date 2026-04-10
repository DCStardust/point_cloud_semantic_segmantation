import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from functools import partial

from model_scfnet import SCFNet
from SemKITTICurb_scf_block_pretrain_baseline_Dataloader import (
    SemanticKITTICurbDataset,
    RAW2TRAIN,
    NUM_CLASSES,
    scf_collate_fn,
    build_scf_inputs_from_blocks
)

# 基础工具函数
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class SimpleLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# 数据增强模块（点云适用，预留）
class PointCloudAugment:
    def __init__(self, rotate_deg=0.0, translate=0.0):
        self.rotate_deg = rotate_deg
        self.translate = translate

    def __call__(self, points):
        # points: (N, 3) numpy array or tensor
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            to_tensor = True
        else:
            points = np.asarray(points).copy()
            to_tensor = False

        # 绕 z 轴小幅旋转
        if self.rotate_deg > 0:
            angle = np.deg2rad(np.random.uniform(-self.rotate_deg, self.rotate_deg))
            cosval, sinval = np.cos(angle), np.sin(angle)
            rot_mat = np.array([
                [cosval, -sinval, 0.0],
                [sinval,  cosval, 0.0],
                [0.0,     0.0,    1.0]
            ], dtype=np.float32)
            points[:, :3] = points[:, :3] @ rot_mat.T

        # 小幅平移
        if self.translate > 0:
            shift = np.random.uniform(-self.translate, self.translate, size=(1, 3)).astype(np.float32)
            shift[0, 2] = 0.0
            points[:, :3] = points[:, :3] + shift

        if to_tensor:
            return torch.from_numpy(points).float()
        return points


# 模型选择与加载
def get_model_class(model_name: str):
    '''
    若有改进模型，Import后在此处增加条目即可
    '''
    model_dict = {
        "scfnet": SCFNet,
    }

    model_name = model_name.lower()
    if model_name not in model_dict:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_dict.keys())}")

    return model_dict[model_name]


def build_experiment_dirs(log_dir: str):
    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir / "log" / log_dir
    ckpt_dir = exp_dir / "checkpoints"

    ensure_dir(exp_dir)
    ensure_dir(ckpt_dir)


    return exp_dir, ckpt_dir


def move_batch_to_device(batch, device):
    '''
    SCF 的 batch 不是单个 tensor
    而是一个带 features / labels / xyz / neigh_idx / sub_idx / interp_idx 的字典
    适配改造：该 helper 专门负责把嵌套 batch dict 搬到 GPU
    '''
    moved = {}
    for k, v in batch.items():
        if isinstance(v, list):
            moved[k] = [x.to(device, non_blocking=True) for x in v]
        elif torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# Dataloader构造，显式传递RAW2TRAIN
def build_dataloaders(args):
    '''
    适配改造：SCF 这里必须在 loader 阶段就把每个 block 组装成层级输入
    '''
    train_transform = None  # 先不开
    if args.use_augmentation:
        train_transform = PointCloudAugment(
            rotate_deg=args.aug_rotate_deg,
            translate=args.aug_translate
        )

    val_transform = None

    if len(args.scf_sub_sampling_ratio) != len(args.scf_d_out):
        raise ValueError(
            f"len(args.scf_sub_sampling_ratio) must equal len(args.scf_d_out), "
            f"got {len(args.scf_sub_sampling_ratio)} vs {len(args.scf_d_out)}"
        )

    scf_collate = partial(
        scf_collate_fn,
        num_layers=len(args.scf_d_out),
        k_n=args.scf_k_n,
        sub_sampling_ratio=tuple(args.scf_sub_sampling_ratio),
    )

    # 数据加载
    train_dataset = SemanticKITTICurbDataset(
        root=args.data_root,
        split="train",
        num_point=args.num_points,
        ignore_label=args.ignore_label,
        label_map=RAW2TRAIN,
        transform=train_transform,
        grid_size=args.grid_size,
        rebuild_cache=args.rebuild_cache,
        block_size=args.block_size,
    )

    val_dataset = SemanticKITTICurbDataset(
        root=args.data_root,
        split="val",
        num_point=args.num_points,
        ignore_label=args.ignore_label,
        label_map=RAW2TRAIN,
        transform=val_transform,
        grid_size=args.grid_size,
        rebuild_cache=False,
        block_size=args.block_size,
        eval_mode=True,
        eval_stride=args.eval_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=scf_collate,
    )

    val_loader = DataLoader(  # 实际不再用于正式 val 评估
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=scf_collate,
    )

    return train_dataset, val_dataset, train_loader, val_loader

# 构造模型输入、loss、optimizer、scheduler，按pytorch骨架
def build_model_and_training_components(args, train_dataset, device):
    '''
    适配scf改造：把 d_out 和 dropout 也显式挂到参数里
    '''
    d_in = 3
    num_classes = NUM_CLASSES

    model_cls = get_model_class(args.model)
    model = model_cls(
        d_in=d_in,
        num_classes=num_classes,
        num_neighbors=args.num_neighbors,
        decimation=args.decimation,
        d_out=tuple(args.scf_d_out),
        dropout=args.scf_dropout,
        device=device,
    ).to(device)

    class_weights = None
    if train_dataset.labelweights is not None:
        class_weights = torch.tensor(
            train_dataset.labelweights,
            dtype=torch.float32,
            device=device
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=args.ignore_label
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 旧配置保留备查
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # 旧配置保留备查
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     gamma=args.scheduler_gamma
    # )

    return model, criterion, optimizer, scheduler


# 评价指标计算
def update_confusion_matrix(conf_mat, preds, labels, num_classes, ignore_label):
    """
    preds: (B, N)
    labels: (B, N)
    """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = labels != ignore_label
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    if labels.numel() == 0:
        return conf_mat

    inds = labels * num_classes + preds
    bins = torch.bincount(inds, minlength=num_classes * num_classes)
    bins = bins.reshape(num_classes, num_classes).cpu().numpy()

    conf_mat += bins
    return conf_mat


def compute_metrics_from_confusion(conf_mat):
    """
    conf_mat: (C, C), row=gt, col=pred
    """
    conf_mat = conf_mat.astype(np.float64)

    tp = np.diag(conf_mat)
    gt = conf_mat.sum(axis=1)
    pred = conf_mat.sum(axis=0)
    union = gt + pred - tp

    class_acc = np.divide(tp, gt, out=np.full_like(tp, np.nan), where=gt > 0)
    class_iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)

    oa = tp.sum() / conf_mat.sum() if conf_mat.sum() > 0 else 0.0
    macc = np.nanmean(class_acc)
    miou = np.nanmean(class_iou)

    return {
        "oa": float(oa),
        "macc": float(macc),
        "miou": float(miou),
        "class_acc": class_acc,
        "class_iou": class_iou,
    }

# 单轮训练定义，是否反传由 optimizer is not None 决定
# val阶段不使用
def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    ignore_label,
    optimizer=None,
    epoch=None,
    epochs=None,
    use_amp=False,
    scaler=None,
    grad_clip_norm=0.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    phase = "Train" if is_train else "Val"
    pbar = tqdm(
        loader,
        desc=f"{phase} Epoch [{epoch:03d}/{epochs:03d}]" if epoch is not None and epochs is not None else phase,
        leave=True,  # 进度条是否保留
        ncols=120,
    )
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in pbar:
            # 适配SCF改造：scf_collate_fn 返回的是 batch dict，而不是 (points, labels) 二元组。
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"].long()  # (B, N)

            if is_train:
                optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    scores = model(batch)  # (B, C, N)

                    if scores.dim() != 3:
                        raise RuntimeError(f"Unexpected score shape: {scores.shape}")

                    if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                        scores = scores.transpose(1, 2).contiguous()

                    loss = criterion(scores, labels)
            else:
                scores = model(batch)

                if scores.dim() != 3:
                    raise RuntimeError(f"Unexpected score shape: {scores.shape}")

                if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                    scores = scores.transpose(1, 2).contiguous()

                loss = criterion(scores, labels)

            # 旧配置
            # scores = model(points)   # 期望输出 (B, C, N) 或等价可转置形式
            #
            # # 为稳妥处理一下可能的输出维度
            # if scores.dim() != 3:
            #     raise RuntimeError(f"Unexpected score shape: {scores.shape}")
            #
            # # 统一到 CrossEntropyLoss 需要的 (B, C, N)
            # if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
            #     scores = scores.transpose(1, 2).contiguous()
            #
            # loss = criterion(scores, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            # if is_train:
            #     loss.backward()
            #     optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(scores, dim=1)   # (B, N)
            conf_mat = update_confusion_matrix(
                conf_mat=conf_mat,
                preds=preds.detach().cpu(),
                labels=labels.detach().cpu(),
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_batches, 1)
    metrics = compute_metrics_from_confusion(conf_mat)
    return avg_loss, metrics

# val阶段验证前向传播使用
def run_eval_full_coverage(
    model,
    dataset,
    criterion,
    device,
    num_classes,
    ignore_label,
    batch_size,
    scf_num_layers,
    scf_k_n,
    scf_sub_sampling_ratio,
    epoch=None,
    epochs=None,
    use_amp=False,
):
    """
    scene-level 全覆盖评估：
    - 对每个 scene 生成固定覆盖 blocks
    - block 预测 logits 回填到 scene 点
    - 对同一点的多次覆盖结果做平均
    - 最终按 scene 点级标签计算指标
    """
    model.eval()

    total_loss = 0.0
    total_blocks = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(
        range(len(dataset)),
        desc=f"ValFull Epoch [{epoch:03d}/{epochs:03d}]" if epoch is not None and epochs is not None else "ValFull",
        leave=True,
        ncols=120,
    )

    with torch.no_grad():
        for scene_idx in pbar:
            scene_labels, scene_sub_idx, blocks = dataset.build_eval_blocks_for_scene(scene_idx)

            if len(blocks) == 0 or scene_labels.size == 0:
                continue

            vote_logits = np.zeros((scene_labels.shape[0], num_classes), dtype=np.float32)
            vote_counts = np.zeros((scene_labels.shape[0],), dtype=np.int32)

            for start in range(0, len(blocks), batch_size):
                # 适配改造：SCF 这里要把 batch_blocks 先转成层级输入，再喂给模型
                batch_blocks = blocks[start:start + batch_size]

                batch_inputs = build_scf_inputs_from_blocks(
                    batch_blocks=batch_blocks,
                    num_layers=scf_num_layers,
                    k_n=scf_k_n,
                    sub_sampling_ratio=tuple(scf_sub_sampling_ratio),
                    to_torch=True,
                )
                batch_inputs = move_batch_to_device(batch_inputs, device)
                labels = batch_inputs["labels"].long()

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        scores = model(batch_inputs)
                        if scores.dim() != 3:
                            raise RuntimeError(f"Unexpected score shape: {scores.shape}")
                        if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                            scores = scores.transpose(1, 2).contiguous()
                        loss = criterion(scores, labels)
                else:
                    scores = model(batch_inputs)
                    if scores.dim() != 3:
                        raise RuntimeError(f"Unexpected score shape: {scores.shape}")
                    if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                        scores = scores.transpose(1, 2).contiguous()
                    loss = criterion(scores, labels)

                total_loss += loss.item()
                total_blocks += 1

                scores_np = scores.detach().cpu().transpose(1, 2).contiguous().numpy()

                for bi, block in enumerate(batch_blocks):
                    local_idx = block["scene_local_idx"]
                    vote_logits[local_idx] += scores_np[bi]
                    vote_counts[local_idx] += 1

            vote_counts = np.maximum(vote_counts, 1)
            vote_logits = vote_logits / vote_counts[:, None]
            final_pred = vote_logits.argmax(axis=1).astype(np.int64)

            conf_mat = update_confusion_matrix(
                conf_mat=conf_mat,
                preds=torch.from_numpy(final_pred),
                labels=torch.from_numpy(scene_labels.astype(np.int64)),
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            pbar.set_postfix(blocks=len(blocks))

    avg_loss = total_loss / max(total_blocks, 1)
    metrics = compute_metrics_from_confusion(conf_mat)
    return avg_loss, metrics


# 日志格式化
def format_class_vector(vec, digits=4):
    return "[" + ", ".join("nan" if np.isnan(x) else f"{x:.{digits}f}" for x in vec) + "]"

def format_class_metric_with_names(class_names, vec, digits=4):
    parts = []
    for name, value in zip(class_names, vec):
        if np.isnan(value):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={value:.{digits}f}")
    return " | ".join(parts)

def log_epoch_summary(
    logger,
    epoch,
    epochs,
    lr,
    train_loss,
    val_loss,
    train_metrics,
    val_metrics,
    best_val_miou,
    best_val_curb_iou,
    epoch_time,
    epoch_start_dt,
    epoch_end_dt,
    early_stop_metric_name,
    best_early_stop_metric
):
    logger.log(
        f"Epoch ****** {epoch:03d}  ({epoch:03d}/{epochs:03d}) ******  "
    )
    logger.log(
        f"start = {epoch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}  | "
        f"end = {epoch_end_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"time = {epoch_time:.1f}s"
    )
    logger.log(
        f"lr={lr:.6e}"
    )

    logger.log(
        "Train  | "
        f"OA={train_metrics['oa']:.4f},  "
        f"mAcc={train_metrics['macc']:.4f},  "
        f"mIoU={train_metrics['miou']:.4f},  "
        f"curb_IoU={train_metrics['class_iou'][2]:.4f} "
    )
    logger.log(
        "Val    | "
        f"OA={val_metrics['oa']:.4f},  "
        f"mAcc={val_metrics['macc']:.4f},  "
        f"mIoU={val_metrics['miou']:.4f},  "
        f"best_mIoU={best_val_miou:.4f},  "
        f"curb_IoU={val_metrics['class_iou'][2]:.4f},   "
        f"best_curb_IoU={best_val_curb_iou:.4f},  "
        f"early_stop_metric({early_stop_metric_name})={best_early_stop_metric:.4f}"
    )

    class_names = ["others",       "road",        "curb",       "sidewalk"]

    logger.log(f"Class order     : {class_names}")
    logger.log(f"Train class Acc : {format_class_metric_with_names(class_names, train_metrics['class_acc'])}")
    logger.log(f"Train class IoU : {format_class_metric_with_names(class_names, train_metrics['class_iou'])}")
    logger.log(f"Train Loss =  {train_loss:.6f}")
    logger.log(f"Val   class Acc : {format_class_metric_with_names(class_names, val_metrics['class_acc'])}")
    logger.log(f"Val   class IoU : {format_class_metric_with_names(class_names, val_metrics['class_iou'])}")
    logger.log(f"Val_loss={val_loss:.6f}")

    logger.log("-" * 100)

def log_train_only_summary(
    logger,
    epoch,
    epochs,
    lr,
    train_loss,
    train_metrics,
    epoch_time,
    epoch_start_dt,
    epoch_end_dt,
):
    logger.log(
        f"Epoch ****** {epoch:03d}  ({epoch:03d}/{epochs:03d}) ******  "
    )
    logger.log(
        f"start = {epoch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}  | "
        f"end = {epoch_end_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"time = {epoch_time:.1f}s"
    )
    logger.log(f"lr={lr:.6e}")

    logger.log(
        "Train  | "
        f"OA={train_metrics['oa']:.4f},  "
        f"mAcc={train_metrics['macc']:.4f},  "
        f"mIoU={train_metrics['miou']:.4f},  "
        f"curb_IoU={train_metrics['class_iou'][2]:.4f} "
    )
    logger.log("Val    | skipped this epoch")

    class_names = ["others", "road", "curb", "sidewalk"]
    logger.log(f"Class order     : {class_names}")
    logger.log(f"Train class Acc : {format_class_metric_with_names(class_names, train_metrics['class_acc'])}")
    logger.log(f"Train class IoU : {format_class_metric_with_names(class_names, train_metrics['class_iou'])}")
    logger.log(f"Train Loss =  {train_loss:.6f}")
    logger.log("-" * 100)


# checkpoint保存（最佳）与恢复（中断续训）
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_miou, args,
                    best_val_curb_iou=-1.0,
                    best_early_stop_metric=-1.0,
                    no_improve_epochs=0,
                    ):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_miou": best_val_miou,
            "best_val_curb_iou": best_val_curb_iou,
            "best_early_stop_metric": best_early_stop_metric,
            "no_improve_epochs": no_improve_epochs,
            "args": vars(args),
        },
        path,
    )

# 中断后续训
def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_miou = ckpt.get("best_val_miou", 0.0)
    best_val_curb_iou = ckpt.get("best_val_curb_iou", -1.0)
    best_early_stop_metric = ckpt.get("best_early_stop_metric", best_val_miou)
    no_improve_epochs = ckpt.get("no_improve_epochs", 0)

    return start_epoch, best_val_miou, best_val_curb_iou, best_early_stop_metric, no_improve_epochs


# 基于预训练微调 仅加载模型权重用于初始化
def load_pretrained_weights(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


# 早停判断
def should_early_stop(
    current_metric: float,
    best_metric: float,
    no_improve_epochs: int,
    patience: int,
    min_delta: float,
):
    """
    基于“最优指标提升是否超过阈值”判断是否累计早停计数。
    返回:
        improved: bool
        new_best_metric: float
        new_no_improve_epochs: int
        stop_now: bool
    """
    improved = current_metric > (best_metric + min_delta)

    if improved:
        best_metric = current_metric
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    stop_now = no_improve_epochs >= patience
    return improved, best_metric, no_improve_epochs, stop_now


# 主训练循环：每个 epoch 后都做一次 val，并记录 best checkpoint
def train(args):
    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    )

    log_dir = args.log_dir or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir, ckpt_dir = build_experiment_dirs(log_dir)
    logger = SimpleLogger(str(exp_dir / "train.log"))

    logger.log(f"Experiment dir: {exp_dir}")
    logger.log(f"Device: {device}")
    logger.log(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # 构造dataset
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    logger.log(f"Train samples: {len(train_dataset)}")
    logger.log(f"Val samples:   {len(val_dataset)}")

    # 检查各类别权重
    class_names = ["others", "road", "curb", "sidewalk"]
    if train_dataset.labelweights is not None:
        weight_str = " | ".join(
            f"{name}={w:.6f}" for name, w in zip(class_names, train_dataset.labelweights)
        )
        logger.log(f"Class weights   : {weight_str}")

    # 构造模型
    model, criterion, optimizer, scheduler = build_model_and_training_components(
        args=args,
        train_dataset=train_dataset,
        device=device,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)  # AMP 与 梯度裁剪

    logger.log(f"Model used: {args.model}")
    logger.log(f"Model input dim: 3")
    logger.log(f"Num classes: {NUM_CLASSES}")
    logger.log(f"Ignore label: {args.ignore_label}")

    start_epoch = 1
    best_val_miou = 0.0
    best_val_curb_iou = -1.0
    best_early_stop_metric = -1.0
    no_improve_epochs = 0
    curb_class_id = 2

    # 检查是否中断续训 / 加载预训练重训
    if args.resume and args.pretrained:
        raise ValueError("Only one of --resume or --pretrained can be set.")

    if args.resume:
        start_epoch, best_val_miou, best_val_curb_iou, best_early_stop_metric, no_improve_epochs = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        logger.log(
            f"Resume early-stop monitor from checkpoint: "
            f"best_val_miou={best_val_miou:.4f}, "
            f"best_val_curb_iou={best_val_curb_iou:.4f}, "
            f"best_early_stop_metric={best_early_stop_metric:.4f}, "
            f"no_improve_epochs={no_improve_epochs}."
        )

        logger.log(f"Resumed from: {args.resume}")
        logger.log(f"Start epoch: {start_epoch}, best_val_miou={best_val_miou:.4f}")


    elif args.pretrained:
        load_pretrained_weights(args.pretrained, model, device)
        logger.log(f"Loaded pretrained weights from: {args.pretrained}")
        logger.log("Start new training from epoch 1 with optimizer/scheduler reset.")

    global_iteration = 0
    global_seen_samples = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_dt = datetime.now()
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=NUM_CLASSES,
            ignore_label=args.ignore_label,
            optimizer=optimizer,
            epoch=epoch,
            epochs=args.epochs,
            use_amp=args.use_amp,  # 混合精度
            scaler=scaler,
            grad_clip_norm=args.grad_clip_norm,  # 梯度裁剪
        )

        # 记录 训练迭代数 和 样本数，并添加相应的 日志输出
        epoch_iteration_count = len(train_loader)
        epoch_seen_samples = len(train_loader.dataset)

        global_iteration += epoch_iteration_count
        global_seen_samples += epoch_seen_samples

        logger.log(
            f"Epoch [{epoch:03d}/{args.epochs:03d}]:  "
            f"epoch_iterations={epoch_iteration_count} "
            f"epoch_seen_samples={epoch_seen_samples}   |   "
            f"global_iterations={global_iteration} "
            f"global_seen_samples={global_seen_samples}"
        )

        # 确定本轮是否进行val
        should_validate = (epoch % args.val_freq == 0) or (epoch == args.epochs)

        if should_validate:
            val_loss, val_metrics = run_eval_full_coverage(
                model=model,
                dataset=val_dataset,
                criterion=criterion,
                device=device,
                num_classes=NUM_CLASSES,
                ignore_label=args.ignore_label,
                batch_size=args.batch_size,
                scf_num_layers=len(args.scf_d_out),
                scf_k_n=args.scf_k_n,
                scf_sub_sampling_ratio=tuple(args.scf_sub_sampling_ratio),
                epoch=epoch,
                epochs=args.epochs,
                use_amp=args.use_amp,
            )

            val_curb_iou = float(val_metrics["class_iou"][curb_class_id])
            current_early_stop_metric = float(val_metrics["miou"])

            improved_early_stop, best_early_stop_metric, no_improve_epochs, stop_now = should_early_stop(
                current_metric=current_early_stop_metric,
                best_metric=best_early_stop_metric,
                no_improve_epochs=no_improve_epochs,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )

            # best ckpt
            is_best = val_metrics["miou"] > best_val_miou
            if is_best:
                best_val_miou = val_metrics["miou"]
                best_val_curb_iou = val_metrics["class_iou"][curb_class_id]
                save_checkpoint(
                    ckpt_dir / "best_model.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_miou,
                    args,
                    best_val_curb_iou=best_val_curb_iou,
                    best_early_stop_metric=best_early_stop_metric,
                    no_improve_epochs=no_improve_epochs,
                )

            # best curb model
            is_best_curb = val_curb_iou > best_val_curb_iou
            if is_best_curb:
                best_val_curb_iou = val_curb_iou
                save_checkpoint(
                    ckpt_dir / "best_curb_model.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_miou,
                    args,
                    best_val_curb_iou=best_val_curb_iou,
                    best_early_stop_metric=best_early_stop_metric,
                    no_improve_epochs=no_improve_epochs,
                )

            logger.log(
                f"EarlyStop monitor | metric={args.early_stop_metric} "
                f"current={current_early_stop_metric:.4f} "
                f"best={best_early_stop_metric:.4f} "
                f"no_improve_epochs={no_improve_epochs}/{args.early_stop_patience} "
                f"min_delta={args.early_stop_min_delta:.6f}"
            )

        else:
            val_loss = None
            val_metrics = None
            stop_now = False
            logger.log(
                f"Validation skipped at epoch {epoch:03d} "
                f"(val_freq={args.val_freq}). "
                f"Best mIoU so far = {best_val_miou:.4f}, "
                f"Best curb IoU so far = {best_val_curb_iou:.4f}"
            )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        epoch_end_dt = datetime.now()

        # latest / periodic ckpt 每轮都保存
        save_checkpoint(
            ckpt_dir / "latest_model.pth",
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_miou,
            args,
            best_val_curb_iou=best_val_curb_iou,
            best_early_stop_metric=best_early_stop_metric,
            no_improve_epochs=no_improve_epochs,
        )

        if epoch % args.save_freq == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_miou,
                args,
                best_val_curb_iou=best_val_curb_iou,
                best_early_stop_metric=best_early_stop_metric,
                no_improve_epochs=no_improve_epochs,
            )

        if should_validate:
            log_epoch_summary(
                logger=logger,
                epoch=epoch,
                epochs=args.epochs,
                lr=current_lr,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_miou=best_val_miou,
                best_val_curb_iou=best_val_curb_iou,
                epoch_time=epoch_time,
                epoch_start_dt=epoch_start_dt,
                epoch_end_dt=epoch_end_dt,
                early_stop_metric_name=args.early_stop_metric,
                best_early_stop_metric=best_early_stop_metric,
            )
        else:
            log_train_only_summary(
                logger=logger,
                epoch=epoch,
                epochs=args.epochs,
                lr=current_lr,
                train_loss=train_loss,
                train_metrics=train_metrics,
                epoch_time=epoch_time,
                epoch_start_dt=epoch_start_dt,
                epoch_end_dt=epoch_end_dt,
            )

        # 早停判断：只在真正做了 val 的 epoch 上检查
        if should_validate and epoch >= args.early_stop_warmup and stop_now:
            logger.log(
                f"Early stopping triggered at epoch {epoch:03d}. "
                f"metric={args.early_stop_metric} failed to improve by at least {args.early_stop_min_delta:.6f} "
                f"for {args.early_stop_patience} consecutive validation rounds."
            )
            break

    logger.log("Training finished.")

def parse_args():
    parser = argparse.ArgumentParser("SCF-Net SemanticKITTI-Curb Training")

    # data
    parser.add_argument("--data_root",
                        type=str,
                        default=r"C:\Lidar\Public_data\3D-Curb-Dataset-all")
    parser.add_argument("--num_points", type=int, default=12288)   # 12k
    parser.add_argument("--grid_size", type=float, default=0.06)
    parser.add_argument("--block_size", type=float, default=20)
    parser.add_argument("--eval_stride", type=float, default=10)  # 全覆盖滑窗步长,默认用20边长
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--ignore_label", type=int, default=-1)

    # augumention 预留
    parser.add_argument("--use_augmentation", action="store_true")  # 默认不开
    parser.add_argument("--aug_rotate_deg", type=float, default=1.0)
    parser.add_argument("--aug_translate", type=float, default=0.01)


    # model
    parser.add_argument("--model", type=str, default="scfnet")
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--scf_d_out", type=int, nargs="+", default=[16, 64, 128, 256])
    parser.add_argument("--scf_k_n", type=int, default=16)
    parser.add_argument("--scf_sub_sampling_ratio", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--scf_dropout", type=float, default=0.5)

    # train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    # 旧指数衰减参数先保留，不再使用
    parser.add_argument("--scheduler_gamma", type=float, default=0.95)


    # runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # early_stopping
    parser.add_argument("--val_freq", type=int, default=2, help="每隔多少个 epoch 做一次 val，最后一轮仍会强制评估")
    parser.add_argument("--early_stop_metric", type=str, default="miou")
    parser.add_argument("--early_stop_patience", type=int, default=10)  # # 按验证次数计；val_freq=2时约等于20个训练epoch
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--early_stop_warmup", type=int, default=20)


    # save / resume / pretrained
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="", help="严格续训：恢复模型+优化器+调度器+epoch等状态")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--pretrained", type=str, default="", help="仅加载模型权重，用于新实验初始化")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 在主循环开始前
    torch.cuda.empty_cache()
    gc.collect()
    # 甚至可以手动初始化一个小的 Tensor 试试驱动
    try:
        test = torch.ones(1).cuda()
        del test
    except:
        print("显卡驱动已失效，请重启系统")

    train(args)
