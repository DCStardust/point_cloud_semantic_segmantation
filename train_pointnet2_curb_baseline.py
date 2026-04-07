import os
import sys
import time
import json
import random
import argparse
import inspect
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.SemKITTICurb_pointnet2_block_pretrain_baseline_Dataloader import (
    SemanticKITTICurbDataset,
    RAW2TRAIN,
    NUM_CLASSES,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))


# =========================
# 基础工具
# =========================
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


class PointCloudAugment:
    def __init__(self, rotate_deg=0.0, translate=0.0):
        self.rotate_deg = rotate_deg
        self.translate = translate

    def __call__(self, points):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            to_tensor = True
        else:
            points = np.asarray(points).copy()
            to_tensor = False

        if self.rotate_deg > 0:
            angle = np.deg2rad(np.random.uniform(-self.rotate_deg, self.rotate_deg))
            cosval, sinval = np.cos(angle), np.sin(angle)
            rot_mat = np.array(
                [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            points[:, :3] = points[:, :3] @ rot_mat.T

        if self.translate > 0:
            shift = np.random.uniform(-self.translate, self.translate, size=(1, 3)).astype(np.float32)
            shift[0, 2] = 0.0
            points[:, :3] = points[:, :3] + shift

        if to_tensor:
            return torch.from_numpy(points).float()
        return points


class PointNet2SegLossWrapper(nn.Module):
    def __init__(self, base_criterion, class_weights=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.class_weights = class_weights

    def forward(self, pred, target, trans_feat):
        return self.base_criterion(pred, target, trans_feat, self.class_weights)


# =========================
# PointNet++ 相关
# =========================
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1 or classname.find("Conv2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


# =========================
# 数据集与目录
# =========================
def build_experiment_dirs(log_dir: str):
    exp_dir = BASE_DIR / "log" / log_dir
    ckpt_dir = exp_dir / "checkpoints"
    ensure_dir(exp_dir)
    ensure_dir(ckpt_dir)
    return exp_dir, ckpt_dir


def build_dataloaders(args):
    train_transform = None
    if args.use_augmentation:
        train_transform = PointCloudAugment(
            rotate_deg=args.aug_rotate_deg,
            translate=args.aug_translate,
        )
    val_transform = None

    # train_dataset = _build_dataset("train", args, train_transform, rebuild_cache=args.rebuild_cache)
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
    # val_dataset = _build_dataset("val", args, val_transform, rebuild_cache=False)
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
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, val_dataset, train_loader, val_loader


# =========================
# 模型/损失/优化器/调度器
# =========================
def build_model_and_training_components(args, train_dataset, device):
    model_module = importlib.import_module(args.model)
    model = model_module.get_model(NUM_CLASSES).to(device)
    model.apply(inplace_relu)

    class_weights = None
    if getattr(train_dataset, "labelweights", None) is not None:
        class_weights = torch.tensor(
            train_dataset.labelweights,
            dtype=torch.float32,
            device=device,
        )

    base_criterion = model_module.get_loss().to(device)
    criterion = PointNet2SegLossWrapper(base_criterion, class_weights=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 原始 PointNet++ 仓库常见旧配置，保留备查
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=args.weight_decay,
    # )
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    # 原始 PointNet++ 旧 step decay 方式，保留备查
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=args.step_size,
    #     gamma=args.lr_decay,
    # )

    return model, criterion, optimizer, scheduler


# =========================
# 指标
# =========================
def update_confusion_matrix(conf_mat, preds, labels, num_classes, ignore_label):
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


# 日志格式化
def format_class_metric_with_names(class_names, vec, digits=4):
    parts = []
    for name, value in zip(class_names, vec):
        if np.isnan(value):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={value:.{digits}f}")
    return " | ".join(parts)


# =========================
# train / val
# =========================
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
        leave=True,
        ncols=120,
    )
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in pbar:
            points, labels = batch  # train/val dataset 返回 (feats, sem)

            points = points.to(device, non_blocking=True).float()  # (B, N, C)
            labels = labels.to(device, non_blocking=True).long()  # (B, N)

            if is_train:
                optimizer.zero_grad()

            # PointNet++ 通道调整
            points_transposed = points.transpose(2, 1).contiguous()  # (B, C, N)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    scores, trans_feat = model(points_transposed)  # scores: (B, N, num_classes)
                    scores_flat = scores.contiguous().view(-1, num_classes)
                    labels_flat = labels.contiguous().view(-1)
                    loss = criterion(scores_flat, labels_flat, trans_feat)
            else:
                scores, trans_feat = model(points_transposed)
                scores_flat = scores.contiguous().view(-1, num_classes)
                labels_flat = labels.contiguous().view(-1)
                loss = criterion(scores_flat, labels_flat, trans_feat)

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

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(scores, dim=2)  # (B, N)
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


# =========================
# checkpoint / early stop / logging
# =========================
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_miou, args,
                    best_val_curb_iou=-1.0,
                    best_early_stop_metric=-1.0,
                    no_improve_epochs=0,
                    scaler=None,
                    ):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_miou": best_val_miou,
            "best_val_curb_iou": best_val_curb_iou,
            "best_early_stop_metric": best_early_stop_metric,
            "no_improve_epochs": no_improve_epochs,
            "args": vars(args),
        },
        path,
    )


# 中断后续训
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

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
def should_early_stop(current_metric, best_metric, no_improve_epochs, patience, min_delta):
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
        best_early_stop_metric,
):
    logger.log(f"Epoch ****** {epoch:03d}  ({epoch:03d}/{epochs:03d}) ******")
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
        f"curb_IoU={train_metrics['class_iou'][2]:.4f}"
    )
    logger.log(
        "Val    | "
        f"OA={val_metrics['oa']:.4f},  "
        f"mAcc={val_metrics['macc']:.4f},  "
        f"mIoU={val_metrics['miou']:.4f},  "
        f"curb_IoU={val_metrics['class_iou'][2]:.4f},  "
        f"best_mIoU={best_val_miou:.4f},  "
        f"best_curb_IoU={best_val_curb_iou:.4f},  "
        f"early_stop_metric({early_stop_metric_name})={best_early_stop_metric:.4f}"
    )

    class_names = ["others", "road", "curb", "sidewalk"]
    logger.log(f"Class order     : {class_names}")
    logger.log(f"Train class Acc : {format_class_metric_with_names(class_names, train_metrics['class_acc'])}")
    logger.log(f"Train class IoU : {format_class_metric_with_names(class_names, train_metrics['class_iou'])}")
    logger.log(f"Train Loss = {train_loss:.6f}")
    logger.log(f"Val   class Acc : {format_class_metric_with_names(class_names, val_metrics['class_acc'])}")
    logger.log(f"Val   class IoU : {format_class_metric_with_names(class_names, val_metrics['class_iou'])}")
    logger.log(f"Val_loss={val_loss:.6f}")
    logger.log("-" * 100)


# =========================
# 主训练循环：每个 epoch 后都做一次 val，并记录 best checkpoint
# =========================
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
    if getattr(train_dataset, "labelweights", None) is not None:
        weight_str = " | ".join(
            f"{name}={w:.6f}" for name, w in zip(class_names, train_dataset.labelweights)
        )
        logger.log(f"Train loss class weights : {weight_str}")

    # 构造模型
    model, criterion, optimizer, scheduler = build_model_and_training_components(
        args,
        train_dataset,
        device
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    logger.log(f"Model used: {args.model}")
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
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if args.use_amp else None,
            device=device,
        )
        logger.log(
            f"Resume training from: {args.resume}\n"
            f"Start epoch: {start_epoch}, "
            f"best_val_miou={best_val_miou:.4f}, "
            f"best_val_curb_iou={best_val_curb_iou:.4f}, "
            f"best_early_stop_metric={best_early_stop_metric:.4f}, "
            f"no_improve_epochs={no_improve_epochs}"
        )
    elif args.pretrained:
        load_pretrained_weights(args.pretrained, model, device=device)
        logger.log(f"Loaded pretrained weights from: {args.pretrained}")
        logger.log("Start new training from epoch 1 with optimizer/scheduler reset.")
    else:
        model.apply(weights_init)
        logger.log("No existing checkpoint. Start training from scratch with Xavier init.")

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
            use_amp=args.use_amp,
            scaler=scaler,
            grad_clip_norm=args.grad_clip_norm,
        )

        # 记录 训练迭代数 和 样本数，并添加相应的 日志输出
        epoch_iteration_count = len(train_loader)
        epoch_seen_samples = len(train_loader.dataset)
        global_iteration += epoch_iteration_count
        global_seen_samples += epoch_seen_samples
        logger.log(
            f"Epoch [{epoch:03d}/{args.epochs:03d}]: "
            f"epoch_iterations={epoch_iteration_count} "
            f"epoch_seen_samples={epoch_seen_samples}   |   "
            f"global_iterations={global_iteration} "
            f"global_seen_samples={global_seen_samples}"
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=NUM_CLASSES,
            ignore_label=args.ignore_label,
            optimizer=None,
            epoch=epoch,
            epochs=args.epochs,
            use_amp=args.use_amp,
            scaler=None,
            grad_clip_norm=0.0,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        epoch_end_dt = datetime.now()

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
                scaler=scaler if args.use_amp else None,
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
                scaler=scaler if args.use_amp else None,
            )

        # latest / periodic ckpt
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
            scaler=scaler if args.use_amp else None,
        )

        # 定期保存 epoch (default 10)
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
                scaler=scaler if args.use_amp else None,
            )

        # 查看是否早停
        logger.log(
            f"EarlyStop monitor | metric={args.early_stop_metric} "
            f"current={current_early_stop_metric:.4f} "
            f"best={best_early_stop_metric:.4f} "
            f"no_improve_epochs={no_improve_epochs}/{args.early_stop_patience} "
            f"min_delta={args.early_stop_min_delta:.6f}"
        )

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

        # 早停判断
        if epoch >= args.early_stop_warmup and stop_now:
            logger.log(
                f"Early stopping triggered at epoch {epoch:03d}. "
                f"metric={args.early_stop_metric} failed to improve by at least {args.early_stop_min_delta:.6f} "
                f"for {args.early_stop_patience} consecutive epochs."
            )
            break

    logger.log("Training finished.")


# =========================
# 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser("PointNet++ SemanticKITTI-Curb Training")

    # data
    parser.add_argument("--data_root",
                        type=str,
                        default=r"G:\ChenXinting\Public_data\3D-Curb-Dataset-all")
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument("--grid_size", type=float, default=0.06)
    parser.add_argument("--block_size", type=float, default=15)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--ignore_label", type=int, default=-1)

    # augmentation (默认关闭)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--aug_rotate_deg", type=float, default=1.0)
    parser.add_argument("--aug_translate", type=float, default=0.01)

    # model
    parser.add_argument("--model", type=str, default="pointnet2_sem_seg_msg")

    # train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)

    # 原始 PointNet++ 旧 step decay 参数，保留备查
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--lr_decay", type=float, default=0.7)

    # runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # early stopping
    parser.add_argument("--early_stop_metric", type=str, default="miou")
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--early_stop_warmup", type=int, default=20)

    # save / resume / pretrained
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="", help="严格续训：恢复模型+优化器+调度器+epoch等状态")
    parser.add_argument("--pretrained", type=str, default="", help="仅加载模型权重，用于新实验初始化")
    parser.add_argument("--save_freq", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
