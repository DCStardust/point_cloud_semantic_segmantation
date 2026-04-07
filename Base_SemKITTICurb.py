import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree


class BaseSemanticKITTIDataset(Dataset):
    """
    统一基类：负责从离线预处理后的 .npy 和 .pkl 文件加载数据。
    """

    def __init__(self, root, split, grid_size=0.06, sequences=None, label_map=None, ignore_label=-1):
        super().__init__()
        self.root = root  # 原始 sequences 目录
        self.split = split
        self.grid_size = grid_size
        self.label_map = label_map
        self.ignore_label = ignore_label

        # 1. 自动计算缓存目录 (例如: sequences_0.06)
        root_norm = os.path.normpath(self.root)
        parent_dir = os.path.dirname(root_norm)
        root_name = os.path.basename(root_norm)
        self.cache_root = os.path.join(parent_dir, f"{root_name}_{grid_size:.2f}")

        # 2. 确定 sequences
        if sequences is None:
            if split == "train":
                sequences = [f"{i:02d}" for i in range(8)] + ["09"]
            elif split == "val":
                sequences = ["08"]
            else:
                sequences = ["10"]
        self.sequences = sequences

        # 3. 扫描所有缓存文件
        self.samples = []
        for seq in self.sequences:
            pts_dir = os.path.join(self.cache_root, seq, "points")
            if not os.path.exists(pts_dir):
                continue
            fnames = sorted([f for f in os.listdir(pts_dir) if f.endswith('.npy')])
            for fn in fnames:
                fid = os.path.splitext(fn)[0]
                self.samples.append({
                    "seq": seq,
                    "fid": fid,
                    "pts_path": os.path.join(self.cache_root, seq, "points", fn),
                    "lbl_path": os.path.join(self.cache_root, seq, "labels", fn),
                    "tree_path": os.path.join(self.cache_root, seq, "KDTree", f"{fid}.pkl")
                })

        self._tree_cache = {}  # 进程内缓存 KDTree 对象

    def _load_tree(self, path):
        if path not in self._tree_cache:
            with open(path, 'rb') as f:
                self._tree_cache[path] = pickle.load(f)
        return self._tree_cache[path]

    def get_raw_block(self, idx, num_points):
        """
        核心逻辑：随机选中心，KNN 裁块。
        """
        s = self.samples[idx]
        xyz_all = np.load(s["pts_path"]).astype(np.float32)
        lbl_all = np.load(s["lbl_path"]).astype(np.int64) if os.path.exists(s["lbl_path"]) else None
        tree = self._load_tree(s["tree_path"])

        # 随机中心点采样
        center_idx = np.random.choice(len(xyz_all))
        center_xyz = xyz_all[center_idx: center_idx + 1]

        # KNN 搜索获取固定点数
        if len(xyz_all) >= num_points:
            _, queried_idx = tree.query(center_xyz, k=num_points)
            queried_idx = queried_idx[0]
        else:
            # 点数不足则重复采样
            queried_idx = np.arange(len(xyz_all))
            pad_idx = np.random.choice(len(xyz_all), num_points - len(xyz_all), replace=True)
            queried_idx = np.concatenate([queried_idx, pad_idx])

        np.random.shuffle(queried_idx)
        return xyz_all[queried_idx], lbl_all[queried_idx] if lbl_all is not None else None, queried_idx

    def __len__(self):
        return len(self.samples)