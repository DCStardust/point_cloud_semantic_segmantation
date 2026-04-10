
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree


def read_bin(bin_path: str) -> np.ndarray:
    """
    SemanticKITTI style point cloud reader.
    Returns:
        pts: (N, 4) float32, columns = [x, y, z, remission]
    """
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 4)
    return pts


def read_label(label_path: str) -> np.ndarray:
    """
    SemanticKITTI style label reader.
    Returns:
        sem: (N,) int64, semantic id only
    """
    lbl = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    sem = (lbl & 0xFFFF).astype(np.int64)
    return sem


IGNORE_RAW_IDS = {0, 1}  # unlabeled / outlier
RAW2TRAIN = {
    # road
    40: 1,
    60: 1,  # lane-marking -> road

    # curb
    3: 2,

    # sidewalk
    48: 3,

    # everything else -> others
    10: 0, 11: 0, 13: 0, 15: 0, 16: 0, 18: 0, 20: 0,
    30: 0, 31: 0, 32: 0,
    44: 0, 49: 0, 50: 0, 51: 0, 52: 0,
    70: 0, 71: 0, 72: 0,
    80: 0, 81: 0, 99: 0,
    252: 0, 253: 0, 254: 0, 255: 0, 256: 0, 257: 0, 258: 0, 259: 0,
}

NUM_CLASSES = max(RAW2TRAIN.values()) + 1  # 4


class SemanticKITTICurbDataset(Dataset):
    """
    Unified external sampling version for RandLA-Net / SCF-Net style experiments.

    设计选择固定：
   1. root仍然指向原始SemanticKITTI序列目录。
   2.在原始根目录旁边自动创建缓存(如果缓存目录经检查无cache内容)，例如sequences_0.06。
   3.离线预处理仅保留网格子采样xyz+映射标签+KDTree + proj idx。
   4.在线采样使用：
        缓存的子云-> XY块裁剪->通过随机采样/重复填充固定的num_point
   5.网络输入仅使用局部中心XYZ（无缓解/无原始XYZ/无额外通道）。

    Returned values keep the same style as the previous version:
        train / val:  (cur_feats, sem)
        test:         (cur_feats, sem, queried_idx, sample_idx)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sequences=None,
        num_point: int = 45056,
        ignore_label: int = -1,
        label_map=None,
        transform=None,
        grid_size: float = 0.06,  # 统一离线预处理
        rebuild_cache: bool = False,   # 检查是否存在，不存在重创建
        block_size: float = 15.0,   # 随机块划分边长
        use_xy_block: bool = True,   #  train 阶段 是否使用随机块采样
        eval_mode: bool = False,  # val/test 阶段 是否使用全覆盖评估模式
        eval_stride: float = None   # 全覆盖滑窗的步长
    ):
        super().__init__()
        assert split in ["train", "val", "test"]

        self.root = root
        self.split = split
        self.num_point = int(num_point)
        self.ignore_label = int(ignore_label)
        self.label_map = RAW2TRAIN if label_map is None else label_map
        self.transform = transform

        self.grid_size = float(grid_size)
        self.rebuild_cache = bool(rebuild_cache)

        self.block_size = float(block_size)
        self.use_xy_block = bool(use_xy_block)
        self.eval_mode = bool(eval_mode)
        self.eval_stride = self.block_size / 2.0 if eval_stride is None else float(eval_stride)

        self.cache_root = self._get_cache_root()
        self._tree_cache = {}

        self.labelweights = None

        # ========== 固定数据集划分 ==============
        if sequences is None:
            if split == "train":
                sequences = [f"{i:02d}" for i in range(0, 8)] + ["09"]
            elif split == "val":
                sequences = ["08"]
            else:
                sequences = ["10"]
        self.sequences = sequences

        self.raw_samples = []
        self.samples = []

        # 样本收集与初始化：适配RandLA-Net改造
        # 先收集self.raw_samples,再自动检查/生成sequence_0.06,最后从缓存目录重建self.samples
        for seq in self.sequences:
            velo_dir = os.path.join(root, seq, "velodyne")
            label_dir = os.path.join(root, seq, "labels")

            if not os.path.isdir(velo_dir):
                raise FileNotFoundError(f"velodyne dir not found: {velo_dir}")

            has_labels = os.path.isdir(label_dir)
            if split != "test" and not has_labels:
                raise FileNotFoundError(f"labels dir not found: {label_dir}")

            bin_files = sorted([f for f in os.listdir(velo_dir) if f.endswith(".bin")])

            for bf in bin_files:
                frame_id = os.path.splitext(bf)[0]
                bin_path = os.path.join(velo_dir, bf)

                label_path = None
                if has_labels:
                    candidate = os.path.join(label_dir, bf.replace(".bin", ".label"))
                    if os.path.isfile(candidate):
                        label_path = candidate

                if split in ["train", "val"] and label_path is None:
                    raise FileNotFoundError(f"label not found for {bin_path}")

                self.raw_samples.append(
                    {
                        "seq": seq,
                        "frame_id": frame_id,
                        "bin_path": bin_path,
                        "label_path": label_path,
                    }
                )

        if len(self.raw_samples) == 0:
            raise RuntimeError("No samples found. Check root/sequences structure.")

        self._ensure_preprocessed_cache()
        self.samples = self._build_cached_samples()

        print(
            f"[SemanticKITTICurbDataset] split={self.split}, sequences={self.sequences}, "
            f"cache_root={self.cache_root}, samples={len(self.samples)}, "
            f"num_point={self.num_point}, block_size={self.block_size}"
        )

        if self.split == "train":
            self.labelweights = self._compute_labelweights()

    def __len__(self):
        return len(self.samples)

    def _get_cache_root(self):
        """
        在原始 sequences 根目录同级创建 sequences_0.06 这样的缓存根目录。
        例如:
            root = /path/to/sequences
            cache_root = /path/to/sequences_0.06
        """
        root_norm = os.path.normpath(self.root)
        parent_dir = os.path.dirname(root_norm)
        root_name = os.path.basename(root_norm)
        grid_tag = f"{self.grid_size:.2f}"
        return os.path.join(parent_dir, f"{root_name}_{grid_tag}")

    def _make_cache_paths(self, seq, frame_id):
        seq_root = os.path.join(self.cache_root, seq)
        return {
            "seq_root": seq_root,
            "points_path": os.path.join(seq_root, "points", f"{frame_id}.npy"),
            "label_path": os.path.join(seq_root, "labels", f"{frame_id}.npy"),
            "kdtree_path": os.path.join(seq_root, "KDTree", f"{frame_id}.pkl"),
            "proj_path": os.path.join(seq_root, "proj", f"{frame_id}.npy"),
        }

    def _build_cached_samples(self):
        samples = []
        for raw_sample in self.raw_samples:
            cache_paths = self._make_cache_paths(raw_sample["seq"], raw_sample["frame_id"])
            samples.append(
                {
                    "seq": raw_sample["seq"],
                    "frame_id": raw_sample["frame_id"],
                    "points_path": cache_paths["points_path"],
                    "label_path": cache_paths["label_path"] if os.path.isfile(cache_paths["label_path"]) else None,
                    "kdtree_path": cache_paths["kdtree_path"],
                    "proj_path": cache_paths["proj_path"],
                }
            )
        return samples

    def _map_labels_only(self, sem: np.ndarray) -> np.ndarray:
        """
        适配RandLA-Net的数据加载采样
        只做标签映射，不在 patch 采样前删除任何点。(采样前不进行ignore过滤)
        这样 KNN 邻域仍建立在原始几何分布上，ignore 类只在训练/评测阶段屏蔽。
        """
        if sem is None:
            return None

        sem = sem.astype(np.int64)
        mapped = np.full(sem.shape, fill_value=self.ignore_label, dtype=np.int64)

        for raw_id, train_id in self.label_map.items():
            mapped[sem == raw_id] = train_id

        for raw_id in IGNORE_RAW_IDS:
            mapped[sem == raw_id] = self.ignore_label

        return mapped

    def _grid_subsample(self, xyz: np.ndarray, sem: np.ndarray = None):
        """
        纯 numpy 版本体素下采样。
        xyz: (N, 3)
        sem: (N,) or None
        返回:
            sub_xyz: (M, 3)
            sub_sem: (M,) or None
        """
        if xyz.shape[0] == 0:
            raise RuntimeError("Empty point cloud encountered during cache preparation.")

        voxel_keys = np.floor(xyz / self.grid_size).astype(np.int64)
        _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
        num_voxels = int(inverse.max()) + 1

        sub_xyz = np.zeros((num_voxels, 3), dtype=np.float32)
        counts = np.bincount(inverse, minlength=num_voxels).astype(np.float32)
        np.add.at(sub_xyz, inverse, xyz)
        sub_xyz /= counts[:, None]

        if sem is None:
            return sub_xyz, None

        order = np.argsort(inverse)
        inverse_sorted = inverse[order]
        sem_sorted = sem[order]

        split_idx = np.flatnonzero(np.diff(inverse_sorted)) + 1
        grouped_labels = np.split(sem_sorted, split_idx)

        sub_sem = np.empty(num_voxels, dtype=np.int64)
        for voxel_id, labels_in_voxel in enumerate(grouped_labels):
            uniq, cnt = np.unique(labels_in_voxel, return_counts=True)
            sub_sem[voxel_id] = uniq[np.argmax(cnt)]

        return sub_xyz, sub_sem

    def _preprocess_one_sample(self, raw_sample):
        """
        从原始 bin/label 生成:
        - 子采样点 points/*.npy
        - 子采样标签 labels/*.npy (若有标签)
        - KDTree/*.pkl
        - proj/*.npy  (原始点 -> 子采样点最近邻索引)
        """
        seq = raw_sample["seq"]
        frame_id = raw_sample["frame_id"]

        cache_paths = self._make_cache_paths(seq, frame_id)

        os.makedirs(os.path.dirname(cache_paths["points_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(cache_paths["label_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(cache_paths["kdtree_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(cache_paths["proj_path"]), exist_ok=True)

        pts = read_bin(raw_sample["bin_path"])
        xyz = pts[:, :3].astype(np.float32)

        raw_sem = read_label(raw_sample["label_path"]) if raw_sample["label_path"] is not None else None
        mapped_sem = self._map_labels_only(raw_sem) if raw_sem is not None else None

        sub_xyz, sub_sem = self._grid_subsample(xyz, mapped_sem)

        search_tree = KDTree(sub_xyz, leaf_size=50)
        proj_idx = search_tree.query(xyz, k=1, return_distance=False).squeeze(1).astype(np.int32)

        np.save(cache_paths["points_path"], sub_xyz)
        if sub_sem is not None:
            np.save(cache_paths["label_path"], sub_sem)

        with open(cache_paths["kdtree_path"], "wb") as f:
            pickle.dump(search_tree, f)

        np.save(cache_paths["proj_path"], proj_idx)

    def _ensure_preprocessed_cache(self):
        """
        第一次构造 dataset 时自动检查缓存是否存在，不存在就生成。
        """
        for raw_sample in self.raw_samples:
            cache_paths = self._make_cache_paths(raw_sample["seq"], raw_sample["frame_id"])

            required_paths = [
                cache_paths["points_path"],
                cache_paths["kdtree_path"],
                cache_paths["proj_path"],
            ]
            if raw_sample["label_path"] is not None:
                required_paths.append(cache_paths["label_path"])

            cache_ok = (not self.rebuild_cache) and all(os.path.isfile(p) for p in required_paths)
            if not cache_ok:
                self._preprocess_one_sample(raw_sample)

    def _load_search_tree(self, kdtree_path):
        """
        简单的进程内 KDTree 缓存，避免同一 worker 反复从磁盘反序列化。
        """
        if kdtree_path not in self._tree_cache:
            with open(kdtree_path, "rb") as f:
                self._tree_cache[kdtree_path] = pickle.load(f)
        return self._tree_cache[kdtree_path]

    def _compute_labelweights(self):
        label_hist = np.zeros(NUM_CLASSES, dtype=np.int64)

        for sample in self.samples:
            if sample["label_path"] is None:
                continue
            train_sem = np.load(sample["label_path"]).astype(np.int64)
            valid = train_sem >= 0
            train_sem = train_sem[valid]
            if train_sem.size == 0:
                continue
            hist, _ = np.histogram(train_sem, bins=np.arange(NUM_CLASSES + 1))
            label_hist += hist

        if label_hist.sum() == 0:
            return np.ones(NUM_CLASSES, dtype=np.float32)

        label_freq = label_hist.astype(np.float32) / label_hist.sum()
        labelweights = 1.0 / np.log(1.2 + label_freq)  # 加权交叉熵
        return labelweights.astype(np.float32)

    def _choose_center_idx(self, xyz: np.ndarray, sem: np.ndarray = None):
        n = xyz.shape[0]
        if n == 0:
            raise RuntimeError("Empty cached point cloud encountered.")

        if sem is not None:
            valid_idx = np.where(sem >= 0)[0]
            if valid_idx.size > 0:
                return int(np.random.choice(valid_idx))

        return int(np.random.randint(0, n))

    def _crop_block_indices(self, xyz: np.ndarray, center_idx: int):
        if not self.use_xy_block:
            return np.arange(xyz.shape[0], dtype=np.int64)

        center = xyz[center_idx]
        half = self.block_size / 2.0

        mask = (
            (xyz[:, 0] >= center[0] - half) & (xyz[:, 0] <= center[0] + half) &
            (xyz[:, 1] >= center[1] - half) & (xyz[:, 1] <= center[1] + half)
        )
        idx = np.where(mask)[0].astype(np.int64)

        if idx.size == 0:
            idx = np.array([center_idx], dtype=np.int64)
        return idx

    def _fix_num_points(self, candidate_idx: np.ndarray):
        n = candidate_idx.shape[0]

        if n == self.num_point:
            queried_idx = candidate_idx.copy()
        elif n > self.num_point:
            queried_idx = np.random.choice(candidate_idx, self.num_point, replace=False)
        else:
            if n == 0:
                raise RuntimeError("No candidate points found for block sampling.")
            pad_idx = np.random.choice(candidate_idx, self.num_point - n, replace=True)
            queried_idx = np.concatenate([candidate_idx, pad_idx], axis=0)

        np.random.shuffle(queried_idx)
        return queried_idx.astype(np.int64)

    def _center_xyz(self, xyz: np.ndarray):
        center = xyz.mean(axis=0, keepdims=True)
        return (xyz - center).astype(np.float32)

    def load_scene_points_labels(self, idx: int, valid_only: bool = True, return_proj: bool = False):
        """
        读取一个 scene（即一帧缓存后的子采样点云）及其标签，并保留子采样点原始索引。
        返回:
            xyz_all:      (M, 3) float32
            sem_all:      (M,)   int64
            sub_idx_all:  (M,)   int64   # 在原始子采样 scene 中的索引
            proj_idx:     (N_raw,) int32 # 可选，原始点 -> 子采样点 最近邻映射
        """
        sample = self.samples[idx]

        xyz_all = np.load(sample["points_path"]).astype(np.float32)
        if sample["label_path"] is not None:
            sem_all = np.load(sample["label_path"]).astype(np.int64)
        else:
            sem_all = np.full((xyz_all.shape[0],), self.ignore_label, dtype=np.int64)

        sub_idx_all = np.arange(xyz_all.shape[0], dtype=np.int64)

        if valid_only:
            valid_mask = sem_all >= 0
            xyz_all = xyz_all[valid_mask]
            sem_all = sem_all[valid_mask]
            sub_idx_all = sub_idx_all[valid_mask]

        if return_proj:
            proj_idx = np.load(sample["proj_path"]).astype(np.int32)
            return xyz_all, sem_all, sub_idx_all, proj_idx

        return xyz_all, sem_all, sub_idx_all

    def _get_window_starts(self, coord_min: float, coord_max: float, offset: float = 0.0):
        """
        生成一维滑窗起点，支持 vote 时的起点偏移，并保证首尾被覆盖。
        """
        if coord_max - coord_min <= self.block_size:
            return np.array([coord_min], dtype=np.float32)

        if self.eval_stride <= 0:
            raise ValueError("eval_stride must be > 0 for full-coverage evaluation.")

        offset = float(offset) % self.eval_stride

        starts = np.arange(
            coord_min + offset,
            coord_max - self.block_size + 1e-6,
            self.eval_stride,
            dtype=np.float32,
        )

        # 始终补上首尾窗口，保证 coverage
        starts = np.concatenate(
            [
                starts,
                np.array([coord_min, coord_max - self.block_size], dtype=np.float32)
            ],
            axis=0
        )

        starts = np.unique(np.round(starts, 6)).astype(np.float32)
        starts.sort()
        return starts

    def _fix_num_points_eval(self, candidate_idx: np.ndarray, seed: int):
        """
        eval 阶段固定随机种子，保证同一个 block 的采样可复现。
        """
        n = candidate_idx.shape[0]
        if n == 0:
            raise RuntimeError("No candidate points found for eval block sampling.")

        rng = np.random.RandomState(seed)

        if n == self.num_point:
            queried_idx = candidate_idx.copy()
        elif n > self.num_point:
            queried_idx = rng.choice(candidate_idx, self.num_point, replace=False)
        else:
            pad_idx = rng.choice(candidate_idx, self.num_point - n, replace=True)
            queried_idx = np.concatenate([candidate_idx, pad_idx], axis=0)

        rng.shuffle(queried_idx)
        return queried_idx.astype(np.int64)

    def build_eval_blocks_for_scene(self, idx: int, offset_x: float = 0.0, offset_y: float = 0.0):
        """
        针对一个 scene 构建全覆盖评估 blocks。
        返回:
            scene_labels:   (M,)   过滤 ignore 后的整帧标签
            scene_sub_idx:  (M,)   对应原始子采样 scene 中的索引
            blocks: list[dict]
                each dict contains:
                    "points":         (num_point, 3)
                    "labels":         (num_point,)
                    "scene_local_idx":(num_point,)  # 在当前有效 scene 中的索引
                    "scene_sub_idx":  (num_point,)  # 在原始子采样 scene 中的索引
        """
        xyz_all, sem_all, sub_idx_all = self.load_scene_points_labels(idx, valid_only=True, return_proj=False)

        if xyz_all.shape[0] == 0:
            return sem_all, sub_idx_all, []

        x_min, x_max = float(xyz_all[:, 0].min()), float(xyz_all[:, 0].max())
        y_min, y_max = float(xyz_all[:, 1].min()), float(xyz_all[:, 1].max())

        x_starts = self._get_window_starts(x_min, x_max, offset=offset_x)
        y_starts = self._get_window_starts(y_min, y_max, offset=offset_y)

        blocks = []
        block_counter = 0

        for xs in x_starts:
            for ys in y_starts:
                xe = xs + self.block_size
                ye = ys + self.block_size

                mask = (
                        (xyz_all[:, 0] >= xs) & (xyz_all[:, 0] <= xe) &
                        (xyz_all[:, 1] >= ys) & (xyz_all[:, 1] <= ye)
                )
                candidate_idx = np.where(mask)[0].astype(np.int64)

                if candidate_idx.size == 0:
                    continue

                queried_local_idx = self._fix_num_points_eval(
                    candidate_idx,
                    seed=idx * 100000 + block_counter
                )
                block_counter += 1

                xyz = xyz_all[queried_local_idx, :]
                sem = sem_all[queried_local_idx]
                cur_feats = self._center_xyz(xyz)

                blocks.append(
                    {
                        "points": cur_feats.astype(np.float32),
                        "labels": sem.astype(np.int64),
                        "scene_local_idx": queried_local_idx.astype(np.int64),
                        "scene_sub_idx": sub_idx_all[queried_local_idx].astype(np.int64),
                    }
                )

        return sem_all.astype(np.int64), sub_idx_all.astype(np.int64), blocks

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        xyz_all = np.load(sample["points_path"]).astype(np.float32)
        if sample["label_path"] is not None:
            sem_all = np.load(sample["label_path"]).astype(np.int64)
        else:
            sem_all = np.full((xyz_all.shape[0],), self.ignore_label, dtype=np.int64)

        # ========= 提前过滤 ===========
        valid_mask = sem_all >= 0  # 只看目标类别中的点
        xyz_all = xyz_all[valid_mask]
        sem_all = sem_all[valid_mask]

        # ========== 采样 ===========
        '''
        离线 grid 后的子点云里，按统一 block size 做 XY 方块裁剪，再固定点数采样/重复补齐
        '''
        # 随机选有效中心点
        center_idx = self._choose_center_idx(xyz_all, sem_all)
        # 以其为中心裁一个XY block
        block_idx = self._crop_block_indices(xyz_all, center_idx)
        # 从block覆盖范围内随机采样固定点数（多则随即丢弃、少则重复采样）
        queried_idx = self._fix_num_points(block_idx)

        xyz = xyz_all[queried_idx, :]
        sem = sem_all[queried_idx]

        # 局部中心化
        cur_feats = self._center_xyz(xyz)

        if self.transform is not None:
            cur_feats, sem = self.transform(cur_feats, sem)

        if self.split == "test":
            return cur_feats, sem, queried_idx.astype(np.int64), np.array([idx], dtype=np.int64)

        return cur_feats, sem


# =========================
# SCF-Net hierarchy builders 适配
# =========================
def _pad_knn_idx(idx: np.ndarray, target_k: int) -> np.ndarray:
    """
    当 support 点数 < k 时，用最后一个近邻重复补齐，保证输出形状稳定。
    idx: (N, k_cur)
    """
    if idx.shape[1] == target_k:
        return idx.astype(np.int64)

    if idx.shape[1] > target_k:
        return idx[:, :target_k].astype(np.int64)

    pad = np.repeat(idx[:, -1:], target_k - idx.shape[1], axis=1)
    idx = np.concatenate([idx, pad], axis=1)
    return idx.astype(np.int64)

def knn_search_numpy(support_pts: np.ndarray, query_pts: np.ndarray, k: int) -> np.ndarray:
    """
    用 sklearn KDTree 做 KNN 查询。
    support_pts: (N_support, 3)
    query_pts:   (N_query, 3)
    return:      (N_query, k)
    """
    if support_pts.shape[0] == 0:
        raise RuntimeError("support_pts is empty in knn_search_numpy.")
    if query_pts.shape[0] == 0:
        raise RuntimeError("query_pts is empty in knn_search_numpy.")

    k_eff = min(int(k), int(support_pts.shape[0]))
    tree = KDTree(support_pts, leaf_size=50)
    idx = tree.query(query_pts, k=k_eff, return_distance=False)
    idx = _pad_knn_idx(idx, int(k))

    return idx.astype(np.int64)

def build_scf_hierarchy_single(
        points: np.ndarray,
        labels: np.ndarray = None,
        num_layers: int = 4,
        k_n: int = 16,
        sub_sampling_ratio=(4, 4, 4, 4),
):
    """
    为单个 block 构建 SCF-Net 所需的多层层级结构。

    输入:
        points: (N, 3)  已经中心化后的 block 点
        labels: (N,) or None

    返回:
        {
            "features":   (N, 3),
            "labels":     (N,) or None,
            "xyz":        list[(N_i, 3)], len=num_layers
            "neigh_idx":  list[(N_i, k_n)], len=num_layers
            "sub_idx":    list[(N_{i+1}, k_n)], len=num_layers
            "interp_idx": list[(N_i, 1)], len=num_layers
        }
    """
    if len(sub_sampling_ratio) != num_layers:
        raise ValueError(
            f"len(sub_sampling_ratio) must equal num_layers, "
            f"got {len(sub_sampling_ratio)} vs {num_layers}"
        )

    cur_xyz = np.asarray(points, dtype=np.float32)
    cur_labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    xyz_list = []
    neigh_idx_list = []
    sub_idx_list = []
    interp_idx_list = []

    for i in range(num_layers):
        if cur_xyz.shape[0] == 0:
            raise RuntimeError(f"Empty points encountered at SCF layer {i}.")

        # 当前层自身 KNN
        neigh_idx = knn_search_numpy(cur_xyz, cur_xyz, k_n)  # (N_i, k)

        # 下采样后的点数
        ratio = int(sub_sampling_ratio[i])
        next_n = max(1, cur_xyz.shape[0] // ratio)

        # 直接取前 next_n 个点作为下一层输入
        # 这里之所以可行，是因为 __getitem__ / eval block 已经完成了随机采样与打乱；
        # 对当前统一基线复现，保持这一简单策略即可。
        sub_xyz = cur_xyz[:next_n, :]

        # pooling indices: 下一层每个点，在当前层找 k 个近邻做 random_sample/max-pool
        sub_idx = knn_search_numpy(cur_xyz, sub_xyz, k_n)  # (N_{i+1}, k)

        # interpolation indices: 当前层每个点，在下一层找 1NN 做上采样
        interp_idx = knn_search_numpy(sub_xyz, cur_xyz, 1)  # (N_i, 1)

        xyz_list.append(cur_xyz.astype(np.float32))
        neigh_idx_list.append(neigh_idx.astype(np.int64))
        sub_idx_list.append(sub_idx.astype(np.int64))
        interp_idx_list.append(interp_idx.astype(np.int64))

        cur_xyz = sub_xyz

    return {
        "features": np.asarray(points, dtype=np.float32),
        "labels": None if cur_labels is None else cur_labels.astype(np.int64),
        "xyz": xyz_list,
        "neigh_idx": neigh_idx_list,
        "sub_idx": sub_idx_list,
        "interp_idx": interp_idx_list,
    }

def build_scf_hierarchy_batch(
        points_batch: np.ndarray,
        labels_batch: np.ndarray = None,
        num_layers: int = 4,
        k_n: int = 16,
        sub_sampling_ratio=(4, 4, 4, 4),
        to_torch: bool = True,
):
    """
    为一个 batch 的 block 构建 SCF-Net 层级输入。

    输入:
        points_batch: (B, N, 3)
        labels_batch: (B, N) or None

    返回:
        dict, 其中：
            features:   (B, N, 3)
            labels:     (B, N)
            xyz:        list[(B, N_i, 3)]
            neigh_idx:  list[(B, N_i, k)]
            sub_idx:    list[(B, N_{i+1}, k)]
            interp_idx: list[(B, N_i, 1)]
    """
    points_batch = np.asarray(points_batch, dtype=np.float32)
    if labels_batch is not None:
        labels_batch = np.asarray(labels_batch, dtype=np.int64)

    per_sample = []
    batch_size = points_batch.shape[0]

    for b in range(batch_size):
        sample_labels = None if labels_batch is None else labels_batch[b]
        sample_dict = build_scf_hierarchy_single(
            points=points_batch[b],
            labels=sample_labels,
            num_layers=num_layers,
            k_n=k_n,
            sub_sampling_ratio=sub_sampling_ratio,
        )
        per_sample.append(sample_dict)

    batch_dict = {
        "features": np.stack([d["features"] for d in per_sample], axis=0).astype(np.float32),
        "labels": None if labels_batch is None else np.stack([d["labels"] for d in per_sample], axis=0).astype(
            np.int64),
        "xyz": [],
        "neigh_idx": [],
        "sub_idx": [],
        "interp_idx": [],
    }

    for i in range(num_layers):
        batch_dict["xyz"].append(
            np.stack([d["xyz"][i] for d in per_sample], axis=0).astype(np.float32)
        )
        batch_dict["neigh_idx"].append(
            np.stack([d["neigh_idx"][i] for d in per_sample], axis=0).astype(np.int64)
        )
        batch_dict["sub_idx"].append(
            np.stack([d["sub_idx"][i] for d in per_sample], axis=0).astype(np.int64)
        )
        batch_dict["interp_idx"].append(
            np.stack([d["interp_idx"][i] for d in per_sample], axis=0).astype(np.int64)
        )

    if not to_torch:
        return batch_dict

    torch_batch = {
        "features": torch.from_numpy(batch_dict["features"]).float(),
        "labels": None if batch_dict["labels"] is None else torch.from_numpy(batch_dict["labels"]).long(),
        "xyz": [torch.from_numpy(x).float() for x in batch_dict["xyz"]],
        "neigh_idx": [torch.from_numpy(x).long() for x in batch_dict["neigh_idx"]],
        "sub_idx": [torch.from_numpy(x).long() for x in batch_dict["sub_idx"]],
        "interp_idx": [torch.from_numpy(x).long() for x in batch_dict["interp_idx"]],
    }
    return torch_batch

def scf_collate_fn(
        batch,
        num_layers: int = 4,
        k_n: int = 16,
        sub_sampling_ratio=(4, 4, 4, 4),
):
    """
    给 DataLoader(train/val) 用的 collate_fn。
    适配当前 dataset.__getitem__ 的输出:
        train / val: (cur_feats, sem)
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch received in scf_collate_fn.")

    # train / val only
    points_list = []
    labels_list = []

    for item in batch:
        if len(item) != 2:
            raise ValueError(
                "scf_collate_fn expects dataset items in format (points, labels). "
                f"Got item with length {len(item)}."
            )
        pts, lbl = item
        points_list.append(np.asarray(pts, dtype=np.float32))
        labels_list.append(np.asarray(lbl, dtype=np.int64))

    points_batch = np.stack(points_list, axis=0)  # (B, N, 3)
    labels_batch = np.stack(labels_list, axis=0)  # (B, N)

    return build_scf_hierarchy_batch(
        points_batch=points_batch,
        labels_batch=labels_batch,
        num_layers=num_layers,
        k_n=k_n,
        sub_sampling_ratio=sub_sampling_ratio,
        to_torch=True,
    )

def build_scf_inputs_from_blocks(
        batch_blocks,
        num_layers: int = 4,
        k_n: int = 16,
        sub_sampling_ratio=(4, 4, 4, 4),
        to_torch: bool = True,
):
    """
    给 test/val full-coverage 时的 batch_blocks 用。
    batch_blocks: list[dict], each contains "points" and "labels"
    """
    if len(batch_blocks) == 0:
        raise RuntimeError("Empty batch_blocks received in build_scf_inputs_from_blocks.")

    points_batch = np.stack([b["points"] for b in batch_blocks], axis=0).astype(np.float32)
    labels_batch = np.stack([b["labels"] for b in batch_blocks], axis=0).astype(np.int64)

    return build_scf_hierarchy_batch(
        points_batch=points_batch,
        labels_batch=labels_batch,
        num_layers=num_layers,
        k_n=k_n,
        sub_sampling_ratio=sub_sampling_ratio,
        to_torch=to_torch,
    )