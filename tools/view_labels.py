import os
import numpy as np
from collections import Counter, defaultdict

def read_label_semantic_ids(label_path: str) -> np.ndarray:
    """Read SemanticKITTI-style .label and return semantic ids (low 16 bits)."""
    raw = np.fromfile(label_path, dtype=np.uint32)
    sem = (raw & 0xFFFF).astype(np.int64)
    return sem

def scan_semkitti_like_dataset(root: str, seqs=None, velo_folder="velodyne", label_folder="labels",
                               max_files_per_seq=None):
    """
    root/
      00/velodyne/*.bin
      00/labels/*.label
      01/...
    """
    if seqs is None:
        # auto-discover two-digit directories like 00,01,...,10
        seqs = sorted([d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)) and len(d) == 2 and d.isdigit()])

    overall_counter = Counter()
    per_seq_unique = {}
    per_seq_counter = {}

    for seq in seqs:
        label_dir = os.path.join(root, seq, label_folder)
        if not os.path.isdir(label_dir):
            print(f"[WARN] seq {seq}: label dir not found: {label_dir}")
            continue

        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".label")])
        if max_files_per_seq is not None:
            label_files = label_files[:max_files_per_seq]

        seq_counter = Counter()
        for lf in label_files:
            lp = os.path.join(label_dir, lf)
            sem = read_label_semantic_ids(lp)
            # count ids
            seq_counter.update(sem.tolist())

        uniq = np.array(sorted(seq_counter.keys()), dtype=np.int64)
        per_seq_unique[seq] = uniq
        per_seq_counter[seq] = seq_counter
        overall_counter.update(seq_counter)

        print(f"unique semantic ids({seq}): {uniq.tolist()}")
        print(f"num unique({seq}): {len(uniq)}")
        print("-" * 60)

    overall_unique = np.array(sorted(overall_counter.keys()), dtype=np.int64)
    print("=== OVERALL ===")
    print("unique semantic ids(ALL):", overall_unique.tolist())
    print("num unique(ALL):", len(overall_unique))

    # show top ids by frequency (helps spot extremely rare classes)
    most_common = overall_counter.most_common(20)
    print("top-20 ids by point count:", most_common)

    return per_seq_unique, per_seq_counter, overall_unique, overall_counter

def scan_flat_dataset(velodyne_dir: str, label_dir: str, max_files=None):
    """
    For Lcdet-NRS-Dataset:
      transfer_velodyne/*.bin
      transfer_labels/*.label
    """
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".label")])
    if max_files is not None:
        label_files = label_files[:max_files]

    counter = Counter()
    for lf in label_files:
        lp = os.path.join(label_dir, lf)
        sem = read_label_semantic_ids(lp)
        counter.update(sem.tolist())

    uniq = np.array(sorted(counter.keys()), dtype=np.int64)
    print("unique semantic ids(flat):", uniq.tolist())
    print("num unique(flat):", len(uniq))
    print("top-20 ids by point count:", counter.most_common(20))
    return uniq, counter

def read_label(path):
    lbl = np.fromfile(path, dtype=np.uint32).reshape(-1)
    sem = (lbl & 0xFFFF).astype(np.int64)
    return sem

# 读取 velodyne 点云
def read_velodyne_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    assert scan.size % 4 == 0, f"file {bin_path} size not divisible by 4"
    points = scan.reshape((-1, 4))  # [N, 4] -> x, y, z, remission
    return points

# 查看一帧点云 + 对应 label 的基本信息
def inspect_one_frame(root, seq="00", idx=0):
    seq_dir = os.path.join(root, seq)
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    label_dir    = os.path.join(seq_dir, "labels")

    bin_path   = os.path.join(velodyne_dir, f"{idx:06d}.bin")
    label_path = os.path.join(label_dir,    f"{idx:06d}.label")

    points = read_velodyne_bin(bin_path)          # [N,4]
    labels = np.fromfile(label_path, dtype=np.uint32)
    # SemanticKITTI 里 label 的低 16 位才是语义 id
    sem_ids = labels & 0xFFFF

    print(f"=== sequence {seq}, frame {idx:06d} ===")
    print("points shape:", points.shape)          # (N,4)
    print("xyz min:", points[:, :3].min(axis=0))
    print("xyz max:", points[:, :3].max(axis=0))
    print("remission min/max:", points[:, 3].min(), points[:, 3].max())
    uniq, cnt = np.unique(sem_ids, return_counts=True)
    print("unique semantic ids in this frame:")
    for u, c in zip(uniq, cnt):
        print(f"  id {u:3d}: {c:7d} pts")

if __name__ == "__main__":
    # 1) 3D-Curb-Dataset-all
    curb_root = r"G:\ChenXinting\Public_data\3D-Curb-Dataset-all"
    scan_semkitti_like_dataset(curb_root)

    # 查看单帧点云结构
    # inspect_one_frame(curb_root, seq="00", idx=0)

    # # 2) Lcdet-NRS-Dataset
    # nrs_velo = r"G:\ChenXinting\Public_data\Lcdet-NRS-Dataset\transfer_velodyne"
    # nrs_label = r"G:\ChenXinting\Public_data\Lcdet-NRS-Dataset\transfer_labels"
    # scan_flat_dataset(nrs_velo, nrs_label)

    # # 3) 单个检查-3D-Curb-Dataset
    # root = r"G:\ChenXinting\Public_data\3D-Curb-Dataset-all"
    # seq = "05"
    # label_dir = os.path.join(root, seq, "labels")
    # files = sorted([f for f in os.listdir(label_dir) if f.endswith(".label")])[:]  # 先看前20帧
    #
    # all_ids = []
    # for f in files:
    #     sem = read_label(os.path.join(label_dir, f))
    #     all_ids.append(np.unique(sem))
    #
    # uniq = np.unique(np.concatenate(all_ids))
    # print(f"unique semantic ids {seq}:", uniq)
    # print("num unique:", len(uniq))

