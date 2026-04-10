import os
import gc
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from model_scfnet import SCFNet
from SemKITTICurb_scf_block_pretrain_baseline_Dataloader import (
    SemanticKITTICurbDataset,
    RAW2TRAIN,
    NUM_CLASSES,
    build_scf_inputs_from_blocks,
)


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


def build_experiment_dirs(log_dir: str):
    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir / "log" / log_dir
    pred_dir = exp_dir / "predictions"

    ensure_dir(exp_dir)
    if pred_dir is not None:
        ensure_dir(pred_dir)

    return exp_dir, pred_dir

# SCF 的 batch 是一个嵌套 dict，不是单个 tensor
def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, list):
            moved[k] = [x.to(device, non_blocking=True) for x in v]
        elif torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# =========================
# 模型与数据
# =========================
def build_test_dataset(args):
    dataset = SemanticKITTICurbDataset(
        root=args.data_root,
        split=args.split,
        num_point=args.num_points,
        ignore_label=args.ignore_label,
        label_map=RAW2TRAIN,
        transform=None,
        grid_size=args.grid_size,
        rebuild_cache=False,
        block_size=args.block_size,
        eval_mode=True,
        eval_stride=args.eval_stride,
    )
    return dataset


def build_model(args, device):
    model = SCFNet(
        d_in=3,
        num_classes=NUM_CLASSES,
        num_neighbors=args.num_neighbors,
        decimation=args.decimation,
        d_out=tuple(args.scf_d_out),
        dropout=args.scf_dropout,
        device=device,
    ).to(device)
    return model


def load_model_weights(checkpoint_path, model, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


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


def format_class_metric_with_names(class_names, vec, digits=4):
    parts = []
    for name, value in zip(class_names, vec):
        if np.isnan(value):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={value:.{digits}f}")
    return " | ".join(parts)


def count_parameters_m(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6


def get_vote_offsets(num_votes: int, eval_stride: float):
    """
    生成多组滑窗起点偏移。
    例如:
        num_votes=1 -> [(0, 0)]
        num_votes=4 -> [(0,0), (0,5), (5,0), (5,5)]   # 当 eval_stride=10
    """
    if num_votes <= 1:
        return [(0.0, 0.0)]

    side = int(np.ceil(np.sqrt(num_votes)))
    offsets_1d = np.linspace(
        0.0,
        float(eval_stride),
        num=side,
        endpoint=False,
        dtype=np.float32,
    )

    offsets = []
    for ox in offsets_1d:
        for oy in offsets_1d:
            offsets.append((float(ox), float(oy)))

    return offsets[:num_votes]


# =========================
# 全覆盖测试
# SCF适配：randla的“单 tensor 前向”，不适用于 SCF
# =========================
def run_test_full_coverage(
    model,
    dataset,
    device,
    num_classes,
    ignore_label,
    batch_size,
    scf_num_layers,
    scf_k_n,
    scf_sub_sampling_ratio,
    use_amp=False,
    save_pred_dir=None,
    logger=None,
    vote_offsets=None,
):
    """
    如果 vote_offsets=[(0,0)]，就是普通 basic test。
    如果 vote_offsets 包含多组 offset，则做多次 full-coverage 聚合投票。
    """
    model.eval()

    if vote_offsets is None or len(vote_offsets) == 0:
        vote_offsets = [(0.0, 0.0)]

    if save_pred_dir is not None:
        ensure_dir(save_pred_dir)

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    has_any_label = False

    total_blocks = 0
    total_scenes = 0
    total_forward_sec = 0.0
    total_forward_batches = 0

    t0 = time.time()

    pbar = tqdm(
        range(len(dataset)),
        desc=f"TestFull [{dataset.split}]",
        leave=True,
        ncols=120,
    )

    with torch.no_grad():
        for scene_idx in pbar:
            scene_labels = None
            scene_sub_idx = None
            scene_vote_logits = None
            scene_vote_counts = None
            scene_total_blocks = 0

            for vote_id, (offset_x, offset_y) in enumerate(vote_offsets):
                cur_scene_labels, cur_scene_sub_idx, blocks = dataset.build_eval_blocks_for_scene(
                    scene_idx,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

                if scene_labels is None:
                    scene_labels = cur_scene_labels
                    scene_sub_idx = cur_scene_sub_idx
                    scene_vote_logits = np.zeros((scene_labels.shape[0], num_classes), dtype=np.float32)
                    scene_vote_counts = np.zeros((scene_labels.shape[0],), dtype=np.int32)

                if len(blocks) == 0:
                    continue

                for start in range(0, len(blocks), batch_size):
                    batch_blocks = blocks[start:start + batch_size]

                    batch_inputs = build_scf_inputs_from_blocks(
                        batch_blocks=batch_blocks,
                        num_layers=scf_num_layers,
                        k_n=scf_k_n,
                        sub_sampling_ratio=tuple(scf_sub_sampling_ratio),
                        to_torch=True,
                    )
                    batch_inputs = move_batch_to_device(batch_inputs, device)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_forward0 = time.perf_counter()

                    if use_amp:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            scores = model(batch_inputs)
                            if scores.dim() != 3:
                                raise RuntimeError(f"Unexpected score shape: {scores.shape}")
                            if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                                scores = scores.transpose(1, 2).contiguous()
                    else:
                        scores = model(batch_inputs)
                        if scores.dim() != 3:
                            raise RuntimeError(f"Unexpected score shape: {scores.shape}")
                        if scores.shape[1] != num_classes and scores.shape[2] == num_classes:
                            scores = scores.transpose(1, 2).contiguous()

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    total_forward_sec += time.perf_counter() - t_forward0
                    total_forward_batches += 1

                    # (B, C, N) -> (B, N, C)
                    scores_np = scores.detach().cpu().transpose(1, 2).contiguous().numpy()

                    for bi, block in enumerate(batch_blocks):
                        local_idx = block["scene_local_idx"]
                        scene_vote_logits[local_idx] += scores_np[bi]
                        scene_vote_counts[local_idx] += 1

                    total_blocks += len(batch_blocks)
                    scene_total_blocks += len(batch_blocks)

            if scene_labels is None or scene_vote_logits is None:
                if logger is not None:
                    sample = dataset.samples[scene_idx]
                    logger.log(
                        f"[Scene {scene_idx:04d}] seq={sample['seq']} frame={sample['frame_id']} skipped because no valid blocks."
                    )
                continue

            scene_vote_counts = np.maximum(scene_vote_counts, 1)
            scene_vote_logits = scene_vote_logits / scene_vote_counts[:, None]
            final_pred_valid = scene_vote_logits.argmax(axis=1).astype(np.int64)

            if scene_labels.size > 0 and np.any(scene_labels != ignore_label):
                has_any_label = True
                conf_mat = update_confusion_matrix(
                    conf_mat=conf_mat,
                    preds=torch.from_numpy(final_pred_valid),
                    labels=torch.from_numpy(scene_labels.astype(np.int64)),
                    num_classes=num_classes,
                    ignore_label=ignore_label,
                )

            if save_pred_dir is not None:
                sample = dataset.samples[scene_idx]

                xyz_full, sem_full, sub_idx_full, proj_idx = dataset.load_scene_points_labels(
                    scene_idx, valid_only=False, return_proj=True
                )

                pred_sub_full = np.full((xyz_full.shape[0],), ignore_label, dtype=np.int64)
                pred_sub_full[scene_sub_idx] = final_pred_valid
                pred_raw = pred_sub_full[proj_idx]

                out_path = Path(save_pred_dir) / f"{sample['seq']}_{sample['frame_id']}_pred.npz"
                np.savez_compressed(
                    out_path,
                    seq=sample["seq"],
                    frame_id=sample["frame_id"],
                    pred_valid=final_pred_valid,
                    scene_sub_idx_valid=scene_sub_idx,
                    pred_subsampled_full=pred_sub_full,
                    pred_raw=pred_raw,
                    vote_offsets=np.array(vote_offsets, dtype=np.float32),
                )

            total_scenes += 1
            pbar.set_postfix(votes=len(vote_offsets), blocks=scene_total_blocks, covered_points=len(scene_labels))

    elapsed = time.time() - t0
    metrics = compute_metrics_from_confusion(conf_mat) if has_any_label else None

    summary = {
        "num_scenes": total_scenes,
        "num_blocks": total_blocks,
        "elapsed_sec": elapsed,
        "avg_forward_batch_sec": total_forward_sec / max(total_forward_batches, 1),
        "num_forward_batches": total_forward_batches,
        "has_label": has_any_label,
        "metrics": metrics,
    }
    return summary


# =========================
# 主流程
# =========================
def test(args):
    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    )

    ckpt_path = Path(args.checkpoint)
    model_stem = ckpt_path.stem
    exp_folder = ckpt_path.parent.parent.name
    exp_suffix = exp_folder.replace("scfnet_", "")
    default_log_name = f"scfnet_test_{exp_suffix}_{model_stem}"
    log_dir = args.log_dir or default_log_name
    exp_dir, pred_dir = build_experiment_dirs(log_dir)
    logger = SimpleLogger(str(exp_dir / "test.log"))

    logger.log(f"Experiment dir: {exp_dir}")
    logger.log(f"Device: {device}")
    logger.log(json.dumps(vars(args), indent=2, ensure_ascii=False))

    dataset = build_test_dataset(args)
    logger.log(f"Test split: {args.split}")
    logger.log(f"Test scenes: {len(dataset)}")

    model = build_model(args, device)
    ckpt = load_model_weights(args.checkpoint, model, device=device)
    # 参数量统计
    total_params_m, trainable_params_m = count_parameters_m(model)

    logger.log(f"Checkpoint loaded from: {args.checkpoint}")
    logger.log(f"Model used: scfnet")
    logger.log(f"Model input dim: 3")
    logger.log(f"Num classes: {NUM_CLASSES}")
    logger.log(f"Ignore label: {args.ignore_label}")
    logger.log(f"Params | total={total_params_m:.3f} M, trainable={trainable_params_m:.3f} M")

    class_names = ["others", "road", "curb", "sidewalk"]

    basic_pred_dir = None
    if args.save_predictions:
        basic_pred_dir = pred_dir / "basic"
        ensure_dir(basic_pred_dir)

    basic_summary = run_test_full_coverage(
        model=model,
        dataset=dataset,
        device=device,
        num_classes=NUM_CLASSES,
        ignore_label=args.ignore_label,
        batch_size=args.batch_size,
        scf_num_layers=len(args.scf_d_out),
        scf_k_n=args.scf_k_n,
        scf_sub_sampling_ratio=tuple(args.scf_sub_sampling_ratio),
        use_amp=args.use_amp,
        save_pred_dir=basic_pred_dir,
        logger=logger,
        vote_offsets=[(0.0, 0.0)],
    )

    logger.log(
        f"[Basic] finished. scenes={basic_summary['num_scenes']}, "
        f"blocks={basic_summary['num_blocks']}, "
        f"time={basic_summary['elapsed_sec']:.1f}s, "
        f"avg_forward_batch_sec={basic_summary['avg_forward_batch_sec']:.6f}"
    )

    if basic_summary["has_label"] and basic_summary["metrics"] is not None:
        metrics = basic_summary["metrics"]
        logger.log(
            "[Basic] Test | "
            f"OA={metrics['oa']:.4f},  "
            f"mAcc={metrics['macc']:.4f},  "
            f"mIoU={metrics['miou']:.4f},  "
            f"curb_IoU={metrics['class_iou'][2]:.4f}"
        )
        logger.log(f"[Basic] Class order     : {class_names}")
        logger.log(f"[Basic] Test  class Acc : {format_class_metric_with_names(class_names, metrics['class_acc'])}")
        logger.log(f"[Basic] Test  class IoU : {format_class_metric_with_names(class_names, metrics['class_iou'])}")
    else:
        logger.log("[Basic] No labels found for this split. Metrics were skipped.")

    if args.enable_vote and args.num_votes > 1:
        vote_offsets = get_vote_offsets(args.num_votes, args.eval_stride)
        logger.log(f"[Vote] vote_offsets={vote_offsets}")

        vote_pred_dir = None
        if args.save_predictions:
            vote_pred_dir = pred_dir / f"vote_{args.num_votes}"
            ensure_dir(vote_pred_dir)

        vote_summary = run_test_full_coverage(
            model=model,
            dataset=dataset,
            device=device,
            num_classes=NUM_CLASSES,
            ignore_label=args.ignore_label,
            batch_size=args.batch_size,
            scf_num_layers=len(args.scf_d_out),
            scf_k_n=args.scf_k_n,
            scf_sub_sampling_ratio=tuple(args.scf_sub_sampling_ratio),
            use_amp=args.use_amp,
            save_pred_dir=vote_pred_dir,
            logger=logger,
            vote_offsets=vote_offsets,
        )

        logger.log(
            f"[Vote] finished. scenes={vote_summary['num_scenes']}, "
            f"blocks={vote_summary['num_blocks']}, "
            f"time={vote_summary['elapsed_sec']:.1f}s, "
            f"avg_forward_batch_sec={vote_summary['avg_forward_batch_sec']:.6f}"
        )

        if vote_summary["has_label"] and vote_summary["metrics"] is not None:
            metrics = vote_summary["metrics"]
            logger.log(
                "[Vote] Test | "
                f"OA={metrics['oa']:.4f},  "
                f"mAcc={metrics['macc']:.4f},  "
                f"mIoU={metrics['miou']:.4f},  "
                f"curb_IoU={metrics['class_iou'][2]:.4f}"
            )
            logger.log(f"[Vote] Class order     : {class_names}")
            logger.log(f"[Vote] Test  class Acc : {format_class_metric_with_names(class_names, metrics['class_acc'])}")
            logger.log(f"[Vote] Test  class IoU : {format_class_metric_with_names(class_names, metrics['class_iou'])}")
        else:
            logger.log("[Vote] No labels found for this split. Metrics were skipped.")

    if args.save_predictions:
        logger.log(f"Predictions root saved to: {pred_dir}")


def parse_args():
    parser = argparse.ArgumentParser("SCF-Net SemanticKITTI-Curb Testing")

    parser.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Lidar\Public_data\3D-Curb-Dataset-all"
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--num_points", type=int, default=12288)
    parser.add_argument("--grid_size", type=float, default=0.06)
    parser.add_argument("--block_size", type=float, default=20.0)
    parser.add_argument("--eval_stride", type=float, default=10.0)
    parser.add_argument("--ignore_label", type=int, default=-1)

    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--scf_d_out", type=int, nargs="+", default=[16, 64, 128, 256])
    parser.add_argument("--scf_k_n", type=int, default=16)
    parser.add_argument("--scf_sub_sampling_ratio", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument("--scf_dropout", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--save_predictions", action="store_true")

    parser.add_argument("--enable_vote", action="store_true", help="是否额外启用多偏移 full-coverage vote")
    parser.add_argument("--num_votes", type=int, default=4, help="vote 的偏移覆盖次数，推荐 4")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    try:
        test_tensor = torch.ones(1).cuda()
        del test_tensor
    except Exception:
        print("显卡驱动已失效，请重启系统")

    test(args)