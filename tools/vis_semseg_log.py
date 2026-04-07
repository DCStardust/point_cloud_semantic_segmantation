import re
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_log(log_path):
    """
    解析训练日志，返回按 epoch 排序的各类指标。
    """
    # 每个 epoch 一个 dict：{ 'train_loss':..., 'train_acc':..., 'val_loss':..., 'val_miou':..., ... }
    epoch_stats = defaultdict(dict)
    current_epoch = None

    # 小工具：把 'nan' 转成 float('nan')
    def to_float(x):
        x = x.strip()
        if x.lower() == "nan":
            return float("nan")
        return float(x)

    epoch_pattern = re.compile(r"\*\*\*\* Epoch\s+(\d+)\s*\(")
    train_loss_pat = re.compile(r"Training mean loss:\s*([0-9.]+)")
    train_acc_pat = re.compile(r"Training accuracy:\s*([0-9.]+)")
    val_loss_pat = re.compile(r"eval mean loss:\s*([0-9.]+)")
    val_miou_pat = re.compile(r"eval point avg class IoU:\s*([0-9.]+)")
    val_acc_pat = re.compile(r"eval point accuracy:\s*([0-9.]+)")

    # 类别 IoU（curb / road）
    curb_iou_pat = re.compile(r"class\s+curb\s+weight:\s*[0-9.]+,\s*IoU:\s*([0-9.a-zA-Z]+)")
    road_iou_pat = re.compile(r"class\s+road\s+weight:\s*[0-9.]+,\s*IoU:\s*([0-9.a-zA-Z]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 判断是否新 epoch
            m = epoch_pattern.search(line)
            if m:
                current_epoch = int(m.group(1))
                # 确保这个 epoch 有条目
                _ = epoch_stats[current_epoch]
                continue

            if current_epoch is None:
                # 还没读到第一个 epoch 头
                continue

            # 训练指标
            m = train_loss_pat.search(line)
            if m:
                epoch_stats[current_epoch]["train_loss"] = float(m.group(1))
                continue

            m = train_acc_pat.search(line)
            if m:
                epoch_stats[current_epoch]["train_acc"] = float(m.group(1))
                continue

            # 验证整体指标
            m = val_loss_pat.search(line)
            if m:
                epoch_stats[current_epoch]["val_loss"] = float(m.group(1))
                continue

            m = val_miou_pat.search(line)
            if m:
                epoch_stats[current_epoch]["val_miou"] = float(m.group(1))
                continue

            m = val_acc_pat.search(line)
            if m:
                epoch_stats[current_epoch]["val_acc"] = float(m.group(1))
                continue

            # 类别 IoU
            m = curb_iou_pat.search(line)
            if m:
                epoch_stats[current_epoch]["curb_iou"] = to_float(m.group(1))
                continue

            m = road_iou_pat.search(line)
            if m:
                epoch_stats[current_epoch]["road_iou"] = to_float(m.group(1))
                continue

    # 按 epoch 排序整理成列表
    epochs = sorted(epoch_stats.keys())
    stats = {
        "epoch": epochs,
        "train_loss": [epoch_stats[e].get("train_loss", float("nan")) for e in epochs],
        "train_acc": [epoch_stats[e].get("train_acc", float("nan")) for e in epochs],
        "val_loss": [epoch_stats[e].get("val_loss", float("nan")) for e in epochs],
        "val_miou": [epoch_stats[e].get("val_miou", float("nan")) for e in epochs],
        "val_acc": [epoch_stats[e].get("val_acc", float("nan")) for e in epochs],
        "curb_iou": [epoch_stats[e].get("curb_iou", float("nan")) for e in epochs],
        "road_iou": [epoch_stats[e].get("road_iou", float("nan")) for e in epochs],
    }
    return stats


def plot_curves(stats, out_prefix="semkitti_curb_full"):
    epochs = stats["epoch"]

    # 1) loss 曲线
    plt.figure()
    plt.plot(epochs, stats["train_loss"], label="train_loss")
    plt.plot(epochs, stats["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss_curve.png", dpi=200)

    # 2) accuracy 曲线
    plt.figure()
    plt.plot(epochs, stats["train_acc"], label="train_acc")
    plt.plot(epochs, stats["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_acc_curve.png", dpi=200)

    # 3) IoU 曲线：整体 mIoU + road / curb
    plt.figure()
    plt.plot(epochs, stats["val_miou"], label="val_mIoU (mean class)")
    plt.plot(epochs, stats["road_iou"], label="road IoU")
    plt.plot(epochs, stats["curb_iou"], label="curb IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation IoU (mean / road / curb)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_iou_curve.png", dpi=200)

    print(f"Saved curves to: {out_prefix}_loss_curve.png / {out_prefix}_acc_curve.png / {out_prefix}_iou_curve.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="path to training log txt")
    parser.add_argument("--out_prefix", type=str, default="semkitti_curb_full", help="prefix of output png files")
    args = parser.parse_args()

    stats = parse_log(args.log_path)
    print(f"Parsed {len(stats['epoch'])} epochs from log.")
    plot_curves(stats, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()