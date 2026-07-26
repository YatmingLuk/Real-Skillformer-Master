"""
train_VAE.py — Trajectory VAE 预训练

============================================================================
设计目标:
  TrajectoryVAE 将 Wayformer 的 ego-local 未来轨迹 [T, 2] 编码到 8 维
  latent space, 并从中解码还原。VAE 训练独立于 Wayformer, 但数据提取
  复用 Wayformer 的 NuPlanDataset 管线, 确保坐标系、场景划分完全一致。

============================================================================
坐标系 (与 Wayformer 完全一致):
  数据来源: Wayformer.data_processor.collect_future()
  坐标系:   initial ego frame, rot = -heading + π/2
            +x = ego 初始前进方向旋转后的分量
            +y = ego 左侧方向旋转后的分量
  数据格式: labels = [pred_horizon=30, 3]  →  [x, y, heading]
  取前 2 维作为 VAE 训练数据: [30, 2]

============================================================================
与 Wayformer 的集成接口 (未来工作):

  1. Wayformer regressor 输出 8 维 VAE latent (替代原来的 T*5=150 维)
     # 原来: self.regressor = nn.Linear(256, 150)
     # 改为: self.regressor = nn.Linear(256, 8)

  2. 加载预训练 VAE decoder, 将 8 维 latent → [T, 2] 轨迹
     vae = TrajectoryVAE(latent_dim=8, traj_len=30)
     vae.load_state_dict(torch.load('pth/trajectory_vae_8d.pth'))
     vae.eval()

  3. 轨迹参与 Wayformer 的 train/eval:
     trajs = vae.decode(latent_weights)      # [B*k, T, 2]
     loss = F.smooth_l1_loss(trajs, labels)   # labels also [..., 2]

============================================================================
数据提取方式 (★ 复用 Wayformer 管线, 而非独立实现):
  1. 使用 Wayformer.wayformer_config 的 config 和 load_config
  2. 使用 Wayformer.wf_dataset 的 NuplanDataset 加载已缓存的场景数据
  3. 从每个 mapping 中提取 labels[:, :2] (与训练标签同源)
  4. 保存为 .npy 供 VAE 训练

运行方式:
  # 完整流程: 从 Wayformer 缓存提取 + 训练 (推荐)
  python train_VAE.py --mode extract_and_train

  # 仅训练 (使用已有 .npy)
  python train_VAE.py --mode train --npy_path traj_data/ego_trajs.npy

  # 仅提取
  python train_VAE.py --mode extract
============================================================================
"""

import os
import sys
import argparse
import math
import pickle
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple

# ===========================================================================
# 路径设置
# ===========================================================================
_cur = os.path.dirname(os.path.abspath(__file__))
if _cur not in sys.path:
    sys.path.insert(0, _cur)

# 压缩/解压 — 与 wf_dataset 一致
COMPRESS = {"zlib": (zlib.compress, zlib.decompress)}
compress, decompress = COMPRESS["zlib"]


# ===========================================================================
# 1. 数据提取 — 复用 Wayformer 的 NuPlanDataset 管线
# ===========================================================================

def extract_ego_trajectories_from_wayformer(
    config_path: str = '/home/test/real-Skillformer-master/config/wayformer.1.json',# /home/test/real-Skillformer-master
    output_npy: str = './traj_data/ego_trajs.npy',
    modes: Tuple[str, ...] = ('train', 'val'),
    val_ratio: float = 0.2,
) -> str:
    """
    ★ 核心: 使用 Wayformer 的 NuPlanDataset 提取 ego-local 未来轨迹.

    这种方式与 Wayformer 训练完全一致:
      - 相同的 config (路径、超参)
      - 相同的 ScenarioFilter (场景划分逻辑)
      - 相同的 data_processor (坐标变换)
      - 相同的缓存机制 (ex_list_nuplan_{mode}.pkl)

    提取逻辑:
      对 train/val 数据集中的每个 mapping:
        1. mapping['labels'] 已是 ego-local frame 下的 [30, 3] = [x, y, heading]
        2. 取 labels[:, :2] 作为 VAE 训练数据
        3. 无需额外坐标变换 (Wayformer 已在 data_processor 中完成)

    Args:
        config_path: Wayformer 配置 JSON 路径
        output_npy:  输出 .npy 路径
        modes:       要提取的数据集模式 ('train', 'val')
        val_ratio:   val 集的场景比例

    Returns:
        output_npy 路径
    """
    from Wayformer.wayformer_config import config as WfConfig, load_config
    from Wayformer.wf_dataset import NuplanDataset

    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)

    print(f"[Extract] 加载 Wayformer 配置: {config_path}")
    cfg: WfConfig = load_config(config_path)
    print(f"[Extract] DATA_PATH={cfg.DATA_PATH}")
    print(f"[Extract] MAP_PATH={cfg.MAP_PATH}")
    print(f"[Extract] temp_file_dir={cfg.temp_file_dir}")
    print(f"[Extract] pred_horizon={cfg.pred_horizon}")

    all_trajs = []

    for mode in modes:
        print(f"\n[Extract] 加载 NuplanDataset(mode='{mode}', val_ratio={val_ratio})...")
        try:
            dataset = NuplanDataset(
                config=cfg,
                mode=mode,
                reuse_cache=True,
                num_workers=4,
                val_ratio=val_ratio,
            )
        except Exception as e:
            print(f"[Extract] 加载 {mode} 数据集失败: {e}")
            print(f"[Extract] 尝试重新构建缓存 (reuse_cache=False)...")
            dataset = NuplanDataset(
                config=cfg,
                mode=mode,
                reuse_cache=False,
                num_workers=4,
                val_ratio=val_ratio,
            )

        print(f"[Extract] {mode} 数据集大小: {len(dataset)}")

        # 遍历数据集, 提取 ego-local 未来轨迹
        for idx in range(len(dataset)):
            try:
                mapping = dataset[idx]
                labels = mapping['labels']                # [30, 3] = [x, y, heading]
                traj = labels[:, :2].astype(np.float32)   # [30, 2] = [x, y]
                all_trajs.append(traj)
            except Exception as e:
                print(f"[Extract] WARNING: 跳过样本 {idx} ({mode}): {e}")
                continue

            if (idx + 1) % 500 == 0:
                print(f"  [Extract {mode}] {idx + 1}/{len(dataset)} ...")

        print(f"[Extract] {mode} 提取完成, 累计轨迹数: {len(all_trajs)}")

    if not all_trajs:
        raise RuntimeError(
            "[Extract] 未提取到任何轨迹! "
            "请确认 NuPlanDataset 缓存存在或 NuPlan 数据路径正确."
        )

    # 堆叠并保存
    traj_array = np.stack(all_trajs, axis=0).astype(np.float32)
    np.save(output_npy, traj_array)
    print(f"\n[Extract] ========================================")
    print(f"[Extract] 已保存 {traj_array.shape[0]} 条 ego-local 轨迹到:")
    print(f"[Extract]   {output_npy}")
    print(f"[Extract] shape={traj_array.shape}, dtype={traj_array.dtype}")
    print(f"[Extract] x range: [{traj_array[..., 0].min():.1f}, {traj_array[..., 0].max():.1f}]")
    print(f"[Extract] y range: [{traj_array[..., 1].min():.1f}, {traj_array[..., 1].max():.1f}]")
    print(f"[Extract] ========================================")

    return output_npy


# ===========================================================================
# 2. 数据集封装 (独立于 Wayformer)
# ===========================================================================

class TrajectoryDataset(Dataset):
    """
    轨迹数据集: 从 .npy 文件加载, 返回 [T, 2].

    完全独立, 不依赖 Wayformer 任何模块.
    """
    def __init__(self, npy_path: str):
        data = np.load(npy_path)[..., :2].astype(np.float32)
        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def compute_global_stats(dataset: TrajectoryDataset) -> dict:
    """
    计算速度与横向偏移统计量, 用于物理锚定 loss.

    统计量用于将 VAE latent 的前 2 维锚定到:
      - dim 0: 平均速度 (speed)
      - dim 1: 横向总偏移 (lateral displacement)
    """
    trajs = dataset.data
    diffs = torch.diff(trajs, dim=1)                         # [N, T-1, 2]
    speeds = torch.norm(diffs, dim=-1).mean(dim=1)           # [N] 平均速度
    lat_disps = trajs[:, -1, 1] - trajs[:, 0, 1]            # [N] 横向总偏移
    return {
        'v_mean': speeds.mean().item(),
        'v_std': speeds.std().item(),
        'lat_mean': lat_disps.mean().item(),
        'lat_std': lat_disps.std().item(),
    }


def extract_physical_features(traj: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    提取归一化的物理特征 [速度, 横向偏移] → [B, 2].

    这些特征用作 Anchor Loss 的 ground truth.
    """
    diffs = torch.diff(traj, dim=1)
    speeds = torch.norm(diffs, dim=-1).mean(dim=1)
    lat_disps = traj[:, -1, 1] - traj[:, 0, 1]
    v_norm = torch.clamp((speeds - stats['v_mean']) / (stats['v_std'] + 1e-6), -3, 3)
    lat_norm = torch.clamp((lat_disps - stats['lat_mean']) / (stats['lat_std'] + 1e-6), -3, 3)
    return torch.stack([v_norm, lat_norm], dim=1)


# ===========================================================================
# 3. TrajectoryVAE 模型 (完全独立)
# ===========================================================================

class TrajectoryVAE(nn.Module):
    """
    Trajectory VAE: 整条轨迹 [T, 2] 编码到 latent_dim 维, 再解码还原.

    Encoder: Flatten(T × 2) → MLP → mu + logvar (各 latent_dim 维)
    Decoder: latent → MLP → unflatten → [T, 2]

    全连接设计不预设时序结构, 完全从数据中学习轨迹流形.

    ============================================================================
    ★★★ 与 Wayformer 的集成接口 ★★★

    集成方式 (在 Wayformer 中):
      # 加载预训练 VAE
      vae = TrajectoryVAE(latent_dim=8, traj_len=30)
      vae.load_state_dict(torch.load('pth/trajectory_vae_8d.pth'))
      vae.eval()
      for p in vae.parameters():
          p.requires_grad = False

      # Wayformer regressor 输出 8 维 latent
      latent_weights = self.regressor(embeddings)  # [B, k, 8]
      B, k, _ = latent_weights.shape
      z = latent_weights.reshape(B * k, 8)
      trajs = vae.decode(z)                        # [B*k, 30, 2]
      trajs = trajs.reshape(B, k, 30, 2)
    ============================================================================

    Args:
        latent_dim:  潜在空间维度 (★ 与 Wayformer regressor 输出维度一致)
        traj_len:    轨迹帧数   (★ 与 Wayformer.pred_horizon 一致)
        feature_dim: 每帧特征数, 默认 2 (x, y)
        hidden_dim:  MLP 隐藏层维度
    """
    def __init__(self, latent_dim: int = 8, traj_len: int = 30, feature_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.traj_len = traj_len
        self.feature_dim = feature_dim
        self.input_dim = traj_len * feature_dim

        # --- Encoder: [B, T*C=60] → [B, latent_dim*2=16] ---
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # --- Decoder: [B, latent_dim=8] → [B, T*C=60] → [B, T, C] ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码轨迹到 latent space.

        Args:
            x: [B, T, C] 轨迹

        Returns:
            mu:    [B, latent_dim]
            logvar: [B, latent_dim]
        """
        x_flat = x.reshape(x.size(0), -1)
        mu_logvar = self.encoder(x_flat)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化: z = mu + ε · σ, 其中 ε ~ N(0, I).

        使采样过程可微分, 梯度通过 mu 和 logvar 回传.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        ★ 核心接口: 从 latent vector 解码轨迹.

        这是 Wayformer 集成的主要入口点.
        Wayformer regressor → 8 维 latent → decode() → [T, 2] 轨迹.

        Args:
            z: [*, latent_dim] latent vectors

        Returns:
            trajectory: [*, T, C] 重建轨迹
        """
        x_recon_flat = self.decoder(z)
        return x_recon_flat.reshape(z.size(0), self.traj_len, self.feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整 VAE 前向: encode → reparameterize → decode.

        Args:
            x: [B, T, C] 输入轨迹

        Returns:
            x_recon: [B, T, C] 重建轨迹
            mu:      [B, latent_dim]
            logvar:  [B, latent_dim]
            z:       [B, latent_dim] latent code
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# ===========================================================================
# 4. Jacobian Loss — 维持潜在空间结构
# ===========================================================================

def compute_jacobian_loss(model: TrajectoryVAE, z: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
    """
    Jacobian 正则化损失, 约束 latent 自由维度 (dim 2~7) 的行为:

      l_ortho:  正交性 — 不同 latent 维度应独立变化
      l_balance: 平衡性 — 各维度的敏感度应均匀, 防止某一维主导
      l_dead:   防死 — 惩罚敏感度过低的维度

    前 2 维 (锚定到速度/横向偏移) 不参与正交性约束,
    因为它们的含义已由 Anchor Loss 确定.

    Args:
        model:       TrajectoryVAE 实例
        z:           [B, latent_dim] latent code
        num_samples: 随机采样点数

    Returns:
        scalar loss
    """
    B, latent_dim = z.shape
    # 对 z 求导, 但同时保留计算图用于 loss 回传
    z = z.clone().detach().requires_grad_(True)

    x_recon = model.decode(z)                               # [B, T, C]
    x_flat = x_recon.reshape(B, -1)                         # [B, T*C]
    T_C = x_flat.shape[1]

    # 随机采样 10 个输出位置, 计算 Jacobian
    sample_indices = torch.randint(0, T_C, (num_samples,), device=z.device)
    x_sampled = x_flat[:, sample_indices]                    # [B, num_samples]

    jacobians = []
    for i in range(num_samples):
        grads = torch.autograd.grad(
            outputs=x_sampled[:, i].sum(),
            inputs=z,
            create_graph=True,
            retain_graph=True,
        )[0]                                                 # [B, latent_dim]
        jacobians.append(grads.unsqueeze(1))

    J = torch.cat(jacobians, dim=1)                         # [B, num_samples, latent_dim]
    JTJ = torch.bmm(J.transpose(1, 2), J)                   # [B, latent_dim, latent_dim]

    # 正交性: 非对角线元素 → 0, 但排除前 2 维 (物理锚定)
    mask = 1.0 - torch.eye(latent_dim, device=z.device)
    mask[:, :2] = 0.0
    mask[:2, :] = 0.0
    l_ortho = (JTJ * mask).pow(2).mean()

    # 平衡性: 各自由维度敏感度应接近均值
    sensitivities = torch.norm(J, p=2, dim=1).mean(dim=0)   # [latent_dim]
    free_sens = sensitivities[2:]
    target_sens = free_sens.mean().detach() + 1e-3
    l_balance = torch.mean((free_sens - target_sens) ** 2)

    # 防死: 敏感度过低 → 指数惩罚
    l_dead = torch.exp(-free_sens * 5.0).mean()

    return l_ortho + 0.5 * (l_balance + l_dead)


# ===========================================================================
# 5. 训练函数
# ===========================================================================

def train_trajectory_vae(
    npy_path: str = './traj_data/ego_trajs.npy',
    latent_dim: int = 8,
    traj_len: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 400,
    device: Optional[str] = None,
    use_amp: bool = False,
    save_dir: str = './pth',
    save_name: str = 'trajectory_vae_{latent_dim}d.pth',
) -> TrajectoryVAE:
    """
    训练 TrajectoryVAE.

    损失组成 (按权重):
      Reconstruction  (×10):  MSE(x_recon, x)        — 轨迹重建保真度
      Anchor          (×3):   MSE(mu[:2], phi_gt)     — 前 2 维锚定到物理量
      Jacobian        (×0.2): ortho + balance + dead   — 后 6 维结构化
      KL              (×0.01): KL(mu, logvar)          — 正则化

    Args:
        npy_path:   ego-local 轨迹 .npy 文件 [N, T, 2]
        latent_dim: 潜在空间维度 (★ 8)
        traj_len:   轨迹帧数     (★ 30, = Wayformer.pred_horizon)
        batch_size: 批次大小
        lr:         学习率
        epochs:     训练轮数
        device:     "cuda" / "cpu"
        use_amp:    是否使用混合精度
        save_dir:   权重保存目录
        save_name:  权重文件名模板

    Returns:
        训练好的 TrajectoryVAE 模型
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[VAE Train] device={device}")
    print(f"[VAE Train] npy={npy_path}")
    print(f"[VAE Train] latent_dim={latent_dim}, traj_len={traj_len}")
    print(f"[VAE Train] epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"{'='*60}")

    # --- 加载数据 ---
    dataset = TrajectoryDataset(npy_path)
    stats = compute_global_stats(dataset)
    print(f"[VAE Train] 数据集大小: {len(dataset)}")
    print(f"[VAE Train] speed:    mean={stats['v_mean']:.2f}, std={stats['v_std']:.2f}")
    print(f"[VAE Train] lat_disp:  mean={stats['lat_mean']:.2f}, std={stats['lat_std']:.2f}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # --- 创建模型 ---
    model = TrajectoryVAE(
        latent_dim=latent_dim,
        traj_len=traj_len,
        feature_dim=2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- 训练循环 ---
    model.train()
    loss_log = []
    best_recon = float('inf')
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_anchor = 0.0
        epoch_jac = 0.0
        n_batches = 0

        for trajs in dataloader:
            trajs = trajs.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                phi_gt = extract_physical_features(trajs, stats).to(device)
                x_recon, mu, logvar, z = model(trajs)

                # ---- 四项损失 ----
                l_recon = nn.MSELoss()(x_recon, trajs)
                l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / trajs.size(0)
                l_anchor = nn.MSELoss()(mu[:, :2], phi_gt)
                l_jac = compute_jacobian_loss(model, z)

                loss = 10.0 * l_recon + 3.0 * l_anchor + 0.2 * l_jac + 0.01 * l_kl

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_recon += l_recon.item()
            epoch_kl += l_kl.item()
            epoch_anchor += l_anchor.item()
            epoch_jac += l_jac.item()
            n_batches += 1

        epoch_r = epoch_recon / n_batches
        epoch_k = epoch_kl / n_batches
        epoch_a = epoch_anchor / n_batches
        epoch_j = epoch_jac / n_batches

        # ★ 追踪最佳模型 (最低 reconstruction loss)
        if epoch_r < best_recon:
            best_recon = epoch_r
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            flag = " ★" if epoch == best_epoch else ""
            print(
                f"Epoch {epoch:03d}{flag} | "
                f"Recon: {epoch_r:.4f} | "
                f"KL: {epoch_k:.4f} | "
                f"Anchor: {epoch_a:.4f} | "
                f"Jac: {epoch_j:.4f}"
            )

        # ★ 记录到 CSV
        loss_log.append({
            'epoch': epoch,
            'recon': epoch_r,
            'kl': epoch_k,
            'anchor': epoch_a,
            'jac': epoch_j,
        })

    # --- 保存权重 ---
    os.makedirs(save_dir, exist_ok=True)
    save_name_filled = save_name.format(latent_dim=latent_dim)

    # 最后 epoch 的权重
    last_path = os.path.join(save_dir, save_name_filled)
    torch.save(model.state_dict(), last_path)
    print(f"[VAE Train] 最后 epoch ({epochs-1}) 权重已保存至: {last_path}")

    # 最佳 epoch 的权重
    if best_state is not None:
        best_name = save_name_filled.replace('.pth', '_best.pth')
        best_path = os.path.join(save_dir, best_name)
        torch.save(best_state, best_path)
        print(f"[VAE Train] 最佳 epoch ({best_epoch}) 权重已保存至: {best_path}  (Recon={best_recon:.4f})")

    # ★ 保存 loss 日志
    if loss_log:
        import csv
        log_path = os.path.join(save_dir, save_name_filled.replace('.pth', '_loss.csv'))
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'recon', 'kl', 'anchor', 'jac'])
            writer.writeheader()
            writer.writerows(loss_log)
        print(f"[VAE Train] Loss 日志已保存至: {log_path}")

    return model


# ===========================================================================
# 6. VAEInterface — 面向 Wayformer 的高级接口
# ===========================================================================

class VAEInterface:
    """
    ★ Wayformer 集成接口.

    封装 VAE 加载、推理、轨迹生成. Wayformer 只需实例化此类.

    使用方式 (在 Wayformer 中):

        # 初始化 — 加载预训练 VAE
        self.vae = VAEInterface(
            weight_path='pth/trajectory_vae_8d.pth',
            latent_dim=8,
            traj_len=30,
            device='cuda',
        )

        # Wayformer forward — 将 regressor 输出的 8 维 latent 解码为轨迹
        # latent_weights: [B, k, 8] ← 来自 self.regressor(embeddings)
        trajs = self.vae.decode_batch(latent_weights)   # → [B, k, 30, 2]

        # 训练时: loss = F.smooth_l1_loss(trajs, labels[..., :2])
        # 评估时: minADE / minFDE / missRate 均基于 trajs 计算
    """

    def __init__(
        self,
        weight_path: str,
        latent_dim: int = 8,
        traj_len: int = 30,
        device: str = 'cuda',
    ):
        self.latent_dim = latent_dim
        self.traj_len = traj_len
        self.device = device

        self.model = TrajectoryVAE(
            latent_dim=latent_dim,
            traj_len=traj_len,
            feature_dim=2,
        ).to(device)

        state_dict = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(state_dict)

        # ★ 冻结 VAE 参数 — 只用于推理
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        print(f"[VAEInterface] 已加载 VAE 权重: {weight_path}")
        print(f"[VAEInterface] latent_dim={latent_dim}, traj_len={traj_len}")
        print(f"[VAEInterface] 参数已冻结, 仅用于推理")

    @torch.no_grad()
    def decode_single(self, z: torch.Tensor) -> torch.Tensor:
        """
        单个 latent vector → 轨迹.

        Args:
            z: [latent_dim] 或 [1, latent_dim]

        Returns:
            [traj_len, 2] numpy 轨迹
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device)
        traj = self.model.decode(z)
        return traj.squeeze(0).cpu()

    @torch.no_grad()
    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        """
        ★ 核心: 批量 latent vectors → 批量轨迹.

        Wayformer 集成时最常用的方法.

        Args:
            z: [B, k, latent_dim] 或 [B, latent_dim]

        Returns:
            [B, k, traj_len, 2] 或 [B, traj_len, 2]
        """
        z = z.to(self.device)

        if z.dim() == 3:
            B, k, D = z.shape
            z_flat = z.reshape(B * k, D)
            trajs = self.model.decode(z_flat)             # [B*k, T, 2]
            return trajs.reshape(B, k, self.traj_len, 2)   # [B, k, T, 2]
        elif z.dim() == 2:
            return self.model.decode(z)                    # [B, T, 2]
        else:
            raise ValueError(f"Unsupported z shape: {z.shape}")

    @torch.no_grad()
    def encode(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        轨迹 → latent (用于分析/可视化).

        Args:
            traj: [B, T, 2]

        Returns:
            mu:    [B, latent_dim]
            logvar: [B, latent_dim]
        """
        traj = traj.to(self.device)
        return self.model.encode(traj)


# ===========================================================================
# 7. 主函数
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TrajectoryVAE 预训练 — 数据提取复用 Wayformer 管线, 训练独立"
    )

    # --- 模式 ---
    parser.add_argument(
        "--mode", type=str, default="extract_and_train",
        choices=["train", "extract", "extract_and_train"],
        help="extract_and_train: 提取+训练 | extract: 仅提取 | train: 仅训练",
    )

    # --- Wayformer 配置 (extract 模式使用) ---
    parser.add_argument(
        "--config_path", type=str,
        default="/home/test/real-Skillformer-master/config/wayformer.1.json",
        help="Wayformer 配置 JSON 路径 (用于 NuPlanDataset 初始化)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="val 集的场景比例 (与 train_wf.py 保持一致)",
    )

    # --- 数据路径 ---
    parser.add_argument(
        "--npy_path", type=str,
        default="./traj_data/ego_trajs.npy",
        help="训练用 .npy 路径",
    )
    parser.add_argument(
        "--output_npy", type=str,
        default="./traj_data/ego_trajs.npy",
        help="输出 .npy 路径 (mode=extract 时使用)",
    )

    # --- VAE 模型参数 ---
    parser.add_argument("--latent_dim", type=int, default=8,
                        help="★ 潜在空间维度 (= Wayformer regressor 输出维度)")
    parser.add_argument("--traj_len", type=int, default=30,
                        help="★ 轨迹帧数 (= Wayformer.pred_horizon)")

    # --- 训练参数 ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true")

    # --- 保存参数 ---
    parser.add_argument("--save_dir", type=str, default="./pth")
    parser.add_argument("--save_name", type=str,
                        default="trajectory_vae_{latent_dim}d.pth")

    args = parser.parse_args()

    npy_to_train = args.npy_path
    output_npy = args.output_npy

    # ================================================================
    # Extract
    # ================================================================
    if args.mode in ("extract", "extract_and_train"):
        if args.mode == "extract_and_train" and os.path.exists(output_npy):
            print(f"[Skip Extract] {output_npy} 已存在, 跳过提取.")
            print(f"[Skip Extract] 删除该文件或使用 --mode extract 强制重新提取.")
            npy_to_train = output_npy
        else:
            try:
                npy_to_train = extract_ego_trajectories_from_wayformer(
                    config_path=args.config_path,
                    output_npy=output_npy,
                    modes=('train', 'val'),
                    val_ratio=args.val_ratio,
                )
            except Exception as e:
                print(f"[ERROR] 提取失败: {e}")
                import traceback
                traceback.print_exc()
                return

        if args.mode == "extract":
            print(f"[Done] 轨迹已提取到: {npy_to_train}")
            return

    # ================================================================
    # Train
    # ================================================================
    if args.mode in ("train", "extract_and_train"):
        if not os.path.exists(npy_to_train):
            print(f"[ERROR] .npy 文件不存在: {npy_to_train}")
            print("请先运行提取:")
            print(f"  python train_VAE.py --mode extract")
            return

        train_trajectory_vae(
            npy_path=npy_to_train,
            latent_dim=args.latent_dim,
            traj_len=args.traj_len,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device,
            use_amp=args.use_amp,
            save_dir=args.save_dir,
            save_name=args.save_name,
        )


if __name__ == "__main__":
    main()
