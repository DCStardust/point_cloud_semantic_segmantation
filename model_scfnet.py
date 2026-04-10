import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """
    Linear -> BN -> LeakyReLU applied on the last channel dimension.
    Supports inputs of shape (..., C).
    """

    def __init__(self, in_channels, out_channels, bn=True, activation=True, negative_slope=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=not bn)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if activation else None

    def forward(self, x):
        orig_shape = x.shape
        x = self.linear(x)
        if self.bn is not None:
            x = x.reshape(-1, x.shape[-1])
            x = self.bn(x)
            x = x.reshape(*orig_shape[:-1], -1)
        if self.activation is not None:
            x = self.activation(x)
        return x


def gather_neighbour(pc, neighbor_idx):
    """
    Gather neighbor features/coordinates.

    Args:
        pc:           (B, N, C)
        neighbor_idx: (B, M, K), indices in [0, N)

    Returns:
        gathered:     (B, M, K, C)
    """
    if neighbor_idx.dtype != torch.long:
        neighbor_idx = neighbor_idx.long()

    b, n, c = pc.shape
    _, m, k = neighbor_idx.shape

    idx_base = torch.arange(b, device=pc.device).view(b, 1, 1) * n
    flat_idx = (neighbor_idx + idx_base).reshape(-1)

    flat_pc = pc.reshape(b * n, c)
    gathered = flat_pc[flat_idx]
    gathered = gathered.reshape(b, m, k, c)
    return gathered


def random_sample(feature, pool_idx):
    """
    Max-pool features according to pool indices.

    Args:
        feature:  (B, N, C)
        pool_idx: (B, N', K)

    Returns:
        pooled:   (B, N', C)
    """
    pooled = gather_neighbour(feature, pool_idx)
    pooled = pooled.max(dim=2).values
    return pooled


def nearest_interpolation(feature, interp_idx):
    """
    Nearest interpolation by index.

    Args:
        feature:    (B, N, C)
        interp_idx: (B, N_up, 1) or (B, N_up)

    Returns:
        interp:     (B, N_up, C)
    """
    if interp_idx.dim() == 3:
        interp_idx = interp_idx[..., 0]
    if interp_idx.dtype != torch.long:
        interp_idx = interp_idx.long()

    b, n, c = feature.shape
    _, n_up = interp_idx.shape

    idx_base = torch.arange(b, device=feature.device).view(b, 1) * n
    flat_idx = (interp_idx + idx_base).reshape(-1)

    flat_feature = feature.reshape(b * n, c)
    interpolated = flat_feature[flat_idx].reshape(b, n_up, c)
    return interpolated


class DualDistanceAttentivePooling(nn.Module):
    """
    DDAP in the original TF implementation.
    Input feature_set channel = feature + local_rep concatenated channel.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att_fc = nn.Linear(in_channels + 2, in_channels, bias=False)
        self.out_mlp = SharedMLP(in_channels, out_channels, bn=True, activation=True)

    def forward(self, feature_set, f_dis, g_dis):
        """
        Args:
            feature_set: (B, N, K, C)
            f_dis:       (B, N, K, 1)
            g_dis:       (B, N, K, 1)
        Returns:
            (B, N, out_channels)
        """
        concat = torch.cat([g_dis, 0.1 * f_dis, feature_set], dim=-1)
        att_scores = torch.softmax(self.att_fc(concat), dim=2)
        f_lc = (feature_set * att_scores).sum(dim=2)
        f_lc = self.out_mlp(f_lc)
        return f_lc


class LocalContextLearning(nn.Module):
    """
    (LPR + DDAP) * 2
    """

    def __init__(self, in_channels, d_out, eps=1e-8):
        super().__init__()
        self.eps = eps

        # local_rep raw dim = 2(angle) + 1(relative_dis) + 3(center xyz) + 3(neighbor xyz) = 9
        self.local_rep_mlp1 = SharedMLP(9, in_channels, bn=True, activation=True)
        self.ddap1 = DualDistanceAttentivePooling(in_channels * 2, d_out // 2)

        self.local_rep_mlp2 = SharedMLP(in_channels, d_out // 2, bn=True, activation=True)
        self.ddap2 = DualDistanceAttentivePooling(d_out, d_out)

    def relative_pos_transforming(self, xyz, neighbor_xyz):
        xyz_tile = xyz.unsqueeze(2).expand_as(neighbor_xyz)
        relative_xyz = xyz_tile - neighbor_xyz

        relative_alpha = torch.atan2(relative_xyz[..., 1], relative_xyz[..., 0]).unsqueeze(-1)
        relative_xydis = torch.sqrt(torch.sum(relative_xyz[..., :2] ** 2, dim=-1).clamp_min(self.eps))
        relative_beta = torch.atan2(relative_xyz[..., 2], relative_xydis).unsqueeze(-1)
        relative_dis = torch.sqrt(torch.sum(relative_xyz ** 2, dim=-1, keepdim=True).clamp_min(self.eps))

        relative_info = torch.cat([relative_dis, xyz_tile, neighbor_xyz], dim=-1)
        exp_dis = torch.exp(-relative_dis)

        local_volume = torch.pow(relative_dis.squeeze(-1).amax(dim=-1).clamp_min(self.eps), 3.0)
        return relative_info, relative_alpha, relative_beta, exp_dis, local_volume

    def local_polar_representation(self, xyz, neigh_idx):
        neighbor_xyz = gather_neighbour(xyz, neigh_idx)

        relative_info, relative_alpha, relative_beta, geometric_dis, local_volume = \
            self.relative_pos_transforming(xyz, neighbor_xyz)

        neighbor_mean = neighbor_xyz.mean(dim=2)
        direction = xyz - neighbor_mean
        direction_tile = direction.unsqueeze(2).expand_as(neighbor_xyz)

        direction_alpha = torch.atan2(direction_tile[..., 1], direction_tile[..., 0]).unsqueeze(-1)
        direction_xydis = torch.sqrt(torch.sum(direction_tile[..., :2] ** 2, dim=-1).clamp_min(self.eps))
        direction_beta = torch.atan2(direction_tile[..., 2], direction_xydis).unsqueeze(-1)

        angle_alpha = relative_alpha - direction_alpha
        angle_beta = relative_beta - direction_beta
        angle_updated = torch.cat([angle_alpha, angle_beta], dim=-1)

        local_rep = torch.cat([angle_updated, relative_info], dim=-1)

        global_dis = torch.sqrt(torch.sum(xyz ** 2, dim=-1, keepdim=True).clamp_min(self.eps))
        # Keep the original TF code semantics: point-wise ||xyz||^3 in the denominator.
        global_volume = torch.pow(global_dis.squeeze(-1).clamp_min(self.eps), 3.0)
        lg_volume_ratio = (local_volume / global_volume).unsqueeze(-1)

        return local_rep, geometric_dis, lg_volume_ratio

    @staticmethod
    def cal_feature_dis(feature, f_neighbours):
        feature_tile = feature.unsqueeze(2).expand_as(f_neighbours)
        feature_dist = torch.mean(torch.abs(feature_tile - f_neighbours), dim=-1, keepdim=True)
        feature_dist = torch.exp(-feature_dist)
        return feature_dist

    def forward(self, xyz, feature, neigh_idx):
        """
        Args:
            xyz:       (B, N, 3)
            feature:   (B, N, C_in)
            neigh_idx: (B, N, K)
        Returns:
            f_lc:            (B, N, d_out)
            lg_volume_ratio: (B, N, 1)
        """
        local_rep, g_dis, lg_volume_ratio = self.local_polar_representation(xyz, neigh_idx)

        local_rep_1 = self.local_rep_mlp1(local_rep)
        f_neigh_1 = gather_neighbour(feature, neigh_idx)
        f_concat_1 = torch.cat([f_neigh_1, local_rep_1], dim=-1)
        f_dis_1 = self.cal_feature_dis(feature, f_neigh_1)
        f_lc_1 = self.ddap1(f_concat_1, f_dis_1, g_dis)

        local_rep_2 = self.local_rep_mlp2(local_rep_1)
        f_neigh_2 = gather_neighbour(f_lc_1, neigh_idx)
        f_concat_2 = torch.cat([f_neigh_2, local_rep_2], dim=-1)
        f_dis_2 = self.cal_feature_dis(f_lc_1, f_neigh_2)
        f_lc_2 = self.ddap2(f_concat_2, f_dis_2, g_dis)

        return f_lc_2, lg_volume_ratio


class SCFBlock(nn.Module):
    """
    One SCF module from the original implementation.
    Input  : (B, N, C_in)
    Output : (B, N, 4 * d_out)
    """

    def __init__(self, in_channels, d_out):
        super().__init__()
        self.mlp1 = SharedMLP(in_channels, d_out // 2, bn=True, activation=True)
        self.local_context = LocalContextLearning(d_out // 2, d_out)
        self.mlp2 = SharedMLP(d_out, d_out * 2, bn=True, activation=False)
        self.shortcut = SharedMLP(in_channels, d_out * 2, bn=True, activation=False)
        self.global_context = SharedMLP(4, d_out * 2, bn=True, activation=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_lc, lg_volume_ratio = self.local_context(xyz, f_pc, neigh_idx)
        f_lc = self.mlp2(f_lc)
        shortcut = self.shortcut(feature)

        f_gc_in = torch.cat([xyz, lg_volume_ratio], dim=-1)
        f_gc = self.global_context(f_gc_in)

        out = torch.cat([f_lc + shortcut, f_gc], dim=-1)
        return self.act(out)


class SCFNet(nn.Module):
    """
    PyTorch rewrite of the official TensorFlow SCF-Net backbone/head.

    Expected input batch dict:
        {
            "features":   FloatTensor (B, N0, d_in), optional; if absent use xyz[0]
            "xyz":        list[FloatTensor], each (B, Ni, 3)
            "neigh_idx":  list[LongTensor],  each (B, Ni, K)
            "sub_idx":    list[LongTensor],  each (B, N{i+1}, K_pool)
            "interp_idx": list[LongTensor],  each (B, Ni, 1)
        }

    Output:
        logits: (B, num_classes, N0)
    """

    def __init__(
        self,
        d_in=3,
        num_classes=13,
        num_neighbors=16,
        decimation=4,
        d_out=(16, 64, 128, 256),
        dropout=0.5,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_out = list(d_out)
        self.num_layers = len(self.d_out)
        self.device_ref = device

        self.fc0 = SharedMLP(d_in, 8, bn=True, activation=True)

        self.encoder_dims = [4 * x for x in self.d_out]
        encoder_in_dims = [8] + self.encoder_dims[:-1]
        self.encoder_blocks = nn.ModuleList([
            SCFBlock(in_ch, d)
            for in_ch, d in zip(encoder_in_dims, self.d_out)
        ])

        self.bottleneck = SharedMLP(self.encoder_dims[-1], self.encoder_dims[-1], bn=True, activation=True)

        decoder_skip_dims = list(reversed([self.encoder_dims[0]] + self.encoder_dims[:-1]))
        decoder_in_dims = []
        current_dim = self.encoder_dims[-1]
        for skip_dim in decoder_skip_dims:
            decoder_in_dims.append(current_dim + skip_dim)
            current_dim = skip_dim

        self.decoder_blocks = nn.ModuleList([
            SharedMLP(in_dim, out_dim, bn=True, activation=True)
            for in_dim, out_dim in zip(decoder_in_dims, decoder_skip_dims)
        ])

        self.head_fc1 = SharedMLP(decoder_skip_dims[-1], 64, bn=True, activation=True)
        self.head_fc2 = SharedMLP(64, 32, bn=True, activation=True)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = SharedMLP(32, num_classes, bn=False, activation=False)

    def forward(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(
                "SCFNet forward expects a dict with keys "
                "features/xyz/neigh_idx/sub_idx/interp_idx."
            )

        xyz = batch["xyz"]
        neigh_idx = batch["neigh_idx"]
        sub_idx = batch["sub_idx"]
        interp_idx = batch["interp_idx"]

        if "features" in batch and batch["features"] is not None:
            feature = batch["features"]
        else:
            feature = xyz[0]

        if feature.shape[-1] != self.d_in:
            raise ValueError(
                f"Expected input feature dim = {self.d_in}, but got {feature.shape[-1]}."
            )

        feature = self.fc0(feature)

        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.encoder_blocks[i](feature, xyz[i], neigh_idx[i])
            f_sampled_i = random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i

            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        feature = self.bottleneck(f_encoder_list[-1])

        skip_features = list(reversed(f_encoder_list[:-1]))
        for j in range(self.num_layers):
            f_interp_i = nearest_interpolation(feature, interp_idx[-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([skip_features[j], f_interp_i], dim=-1))
            feature = f_decoder_i

        feature = self.head_fc1(feature)
        feature = self.head_fc2(feature)
        feature = self.dropout(feature)
        logits = self.classifier(feature)
        logits = logits.transpose(1, 2).contiguous()
        return logits