import os
import numpy as np
from torch.utils.data import Dataset

# 读取数据 原始点云
def read_bin(bin_path: str) -> np.ndarray:
    # SemanticKITTI style: float32, N x 4 (x,y,z,intensity)
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 4)
    return pts

# 读取标签 语义类别id
def read_label(label_path: str) -> np.ndarray:
    # SemanticKITTI style: uint32, N
    lbl = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    sem = (lbl & 0xFFFF).astype(np.int64)   # semantic id
    return sem

# raw semantic id -> train id
IGNORE_RAW_IDS = {0, 1}  # 0 unlabeled, 1 outlier
RAW2TRAIN = {
    # others: 0
    # on-rails / other-vehicle (moving on-rails & bus & other-vehicle) / parking / other-ground / other-structure / other-object
    16: 0, 20: 0, 44: 0, 49: 0, 52: 0, 99: 0, 256: 0, 257: 0, 259: 0,
    40: 1,  # road: 1
    3: 2,  # curb: 2
    48: 3,  # sidewalk: 3
    50: 4,  # building: 4
    51: 5,  # fence: 5
    60: 6,  # lane-marking: 6
    80: 7,  # pole: 7
    81: 8,  # traffic sign: 8
    72: 9,  # terrain: 9
    70: 10, 71: 10,  # vegetation / trunk: 10

    # human: 11
    30: 11, 31: 11, 32: 11, # person / bicyclist / motorcyclist
    253: 11, 254: 11, 255: 11, # moving-human
    # vehicle: 12
    10: 12, 11: 12, 13: 12, 15: 12, 18: 12,   # car / bicycle / bus / motorcycle / truck
    252: 12, 258: 12,  # moving-vehicle
}
NUM_CLASSES = max(RAW2TRAIN.values()) + 1  # 13

class SemanticKITTICurbDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        sequences=None,
        num_point: int = 4096,
        ignore_label: int = -1,
        label_map=None,   # optional: dict[int,int], map raw semantic id -> train id
        transform=None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.num_point = num_point
        self.ignore_label = ignore_label
        self.label_map = label_map
        self.transform = transform
        self.labelweights = None

        # Default sequence split (you can adjust)
        if sequences is None:
            # Example: use 00-08 for train, 09 for val, 10 for test
            if split == "train":
                sequences = [f"{i:02d}" for i in range(0, 9)]
            elif split == "val":
                sequences = ["09"]
            elif split == 'test':
                sequences = ["10"]
        self.sequences = sequences

        self.samples = []
        for seq in self.sequences:
            velo_dir = os.path.join(root, seq, "velodyne")
            label_dir = os.path.join(root, seq, "labels")

            if not os.path.isdir(velo_dir):
                raise FileNotFoundError(f"velodyne dir not found: {velo_dir}")
            if split != "test" and not os.path.isdir(label_dir):
                raise FileNotFoundError(f"labels dir not found: {label_dir}")

            bin_files = sorted([f for f in os.listdir(velo_dir) if f.endswith(".bin")])
            for bf in bin_files:
                bin_path = os.path.join(velo_dir, bf)
                if split == "test":
                    label_path = None
                else:
                    lf = bf.replace(".bin", ".label")
                    label_path = os.path.join(label_dir, lf)
                    if not os.path.isfile(label_path):
                        raise FileNotFoundError(f"label not found for {bin_path}: {label_path}")
                self.samples.append((bin_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check root/sequences structure.")

        print(f"[SemanticKITTICurbDataset] split={split}, sequences={self.sequences}, samples={len(self.samples)}")

        # 初始化dataset时统计训练样本分布并计算权重
        self.labelweights = None
        if split == 'train':
            label_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
            for (_, label_path) in self.samples:
                raw_sem = read_label(label_path)
                train_sem = np.full_like(raw_sem, -1, dtype=np.int32)
                for raw_id, train_id in RAW2TRAIN.items():
                    train_sem[raw_sem == raw_id] = train_id
                valid = train_sem >= 0
                train_sem = train_sem[valid]
                hist, _ = np.histogram(train_sem, bins=np.arange(NUM_CLASSES + 1))
                label_hist += hist

            label_freq = label_hist.astype(np.float32) / label_hist.sum()
            self.labelweights = 1.0 / np.log(1.2 + label_freq)  # S3DIS 同款配方

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        bin_path, label_path = self.samples[idx]
        pts = read_bin(bin_path)          # (N,4)
        xyz = pts[:, 0:3].astype(np.float32)
        inten = pts[:, 3:4].astype(np.float32)  # (N,1)

        if self.split == "test":
            sem = np.full((xyz.shape[0],), self.ignore_label, dtype=np.int64)
        else:
            sem = read_label(label_path)  # shape: (N,)

        # optional label mapping
        if self.label_map is not None and self.split != "test":
            mapped = np.full_like(sem, fill_value=self.ignore_label, dtype=np.int64) # 默认所有 raw label 都是 ignore
            for raw_id, train_id in self.label_map.items():
                mapped[sem == raw_id] = train_id  # 根据RAW2TRAIN，把原始id映射训练id
            # 丢掉 ignore_raw_ids
            for rid in IGNORE_RAW_IDS:
                mapped[sem == rid] = self.ignore_label  # 额外不在RAW2TRAIN的保险覆盖一遍
            valid_mask = mapped != self.ignore_label  # 布尔掩码，保留不是-1的，对应在RAW2TRAIN且不在IGNORE_RAW_IDS的
            # 在坐标、强度、标签三个数组上同时应用这个 mask
            xyz = xyz[valid_mask]
            inten = inten[valid_mask]
            sem = mapped[valid_mask]

        # 采样
        n = xyz.shape[0]
        if n >= self.num_point:
            choice = np.random.choice(n, self.num_point, replace=False)
        else:
            choice = np.random.choice(n, self.num_point, replace=True)

        xyz = xyz[choice, :]
        inten = inten[choice, :]
        sem = sem[choice]

        # 归一化
        # intensity normalization (simple, safe)
        # If your intensity is already [0,1], this does nothing harmful.
        inten = np.clip(inten, 0.0, 1.0)

        # local centering 局部坐标中心化-->平移
        center = xyz.mean(axis=0, keepdims=True)
        xyz_centered = xyz - center

        # global normalization within the scan (min-max) 全局坐标归一化
        xyz_min = xyz.min(axis=0, keepdims=True)
        xyz_max = xyz.max(axis=0, keepdims=True)
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)

        # final features: (num_point, 7) = xyz_centered(3) + intensity(1) + xyz_norm(3)
        # shape (N,7): 相对坐标 3 + remission 1 + 归一化坐标 3
        # feats = np.concatenate([xyz_centered, inten, xyz_norm], axis=1).astype(np.float32)

        # 对齐 9 dimension
        cur_feats = np.zeros((self.num_point, 9), dtype=np.float32)
        cur_feats[:, 0:3] = xyz_centered   # local xyz
        cur_feats[:, 3] = inten[:, 0]      # intensity 灰度
        cur_feats[:, 4:6] = 0.0            # 补位
        cur_feats[:, 6:9] = xyz_norm       # global normalized xyz

        if self.transform is not None:
            cur_feats, sem = self.transform(cur_feats, sem)

        return cur_feats, sem
