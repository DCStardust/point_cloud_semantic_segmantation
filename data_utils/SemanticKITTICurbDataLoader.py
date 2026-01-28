import os
import numpy as np
from torch.utils.data import Dataset

def read_bin(bin_path: str) -> np.ndarray:
    # SemanticKITTI style: float32, N x 4 (x,y,z,intensity)
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 4)
    return pts

def read_label(label_path: str) -> np.ndarray:
    # SemanticKITTI style: uint32, N
    lbl = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    sem = (lbl & 0xFFFF).astype(np.int64)   # semantic id
    return sem

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

        # Default sequence split (you can adjust)
        if sequences is None:
            # Example: use 00-08 for train, 09 for val, 10 for test
            if split == "train":
                sequences = [f"{i:02d}" for i in range(0, 9)]
            elif split == "val":
                sequences = ["09"]
            else:
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
            sem = read_label(label_path)  # (N,)

        # optional label mapping
        if self.label_map is not None and self.split != "test":
            mapped = np.full_like(sem, self.ignore_label)
            for k, v in self.label_map.items():
                mapped[sem == k] = v
            sem = mapped

        n = xyz.shape[0]
        if n >= self.num_point:
            choice = np.random.choice(n, self.num_point, replace=False)
        else:
            choice = np.random.choice(n, self.num_point, replace=True)

        xyz = xyz[choice, :]
        inten = inten[choice, :]
        sem = sem[choice]

        # intensity normalization (simple, safe)
        # If your intensity is already [0,1], this does nothing harmful.
        inten = np.clip(inten, 0.0, 1.0)

        # local centering 局部坐标中心化-->平移
        center = xyz.mean(axis=0, keepdims=True)
        xyz_centered = xyz - center

        # global normalization within the scan (min-max)
        xyz_min = xyz.min(axis=0, keepdims=True)
        xyz_max = xyz.max(axis=0, keepdims=True)
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)

        # final features: (num_point, 7) = xyz_centered(3) + intensity(1) + xyz_norm(3)
        feats = np.concatenate([xyz_centered, inten, xyz_norm], axis=1).astype(np.float32)

        if self.transform is not None:
            feats, sem = self.transform(feats, sem)

        return feats, sem
