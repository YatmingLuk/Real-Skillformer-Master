import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from typing import List, Tuple

from .utils import EarlyFusion,Decoder
from .wayformer_config import get_from_mapping,config,Metrics

class Wayformer(nn.Module):

    def __init__(self,config:config) -> None:
        super(Wayformer,self).__init__()
        self.config = config
        self.encoder = EarlyFusion(config)   #history/interact/road -> Memory
        self.decoder = Decoder(config)

        # z normalization for skill loss
        if getattr(config, 'use_z_norm', False):
            stats = torch.load(config.z_stats_path, map_location='cpu')
            self.register_buffer('z_mean', stats['mean'])
            self.register_buffer('z_std', stats['std'])
        else:
            latent_dim = getattr(config, 'vae_latent_dim', 8)
            self.register_buffer('z_mean', torch.zeros(latent_dim))
            self.register_buffer('z_std', torch.ones(latent_dim))

    def regress_skill(self, mappings: List, device) -> Tensor:
        """Regress raw (unnormalized) scene skill latents, [B, latent_dim]."""
        agents = get_from_mapping(mappings, 'agents')
        matrix = get_from_mapping(mappings, 'matrix')
        memory, memory_mask = self.encoder(agents, matrix, device)
        return self.decoder(memory, memory_mask)

    @torch.no_grad()
    def encode_gt_skill(self, mappings: List, device) -> Tensor:
        """Encode ground-truth xy trajectories as VAE posterior means."""
        labels = np.asarray(get_from_mapping(mappings, 'labels'))
        labels = torch.as_tensor(labels, device=device, dtype=torch.float32)
        mu_gt, _ = self.decoder.vae.encode(labels[..., :2])
        return mu_gt
    
    def forward(self, mappings: List, device) -> Tensor:
        """Train one scene-conditioned latent using only skill supervision."""
        z_pred = self.regress_skill(mappings, device)
        mu_gt = self.encode_gt_skill(mappings, device)

        if self.config.use_z_norm:
            z_std = torch.clamp(self.z_std, min=1e-4)
            z_pred = (z_pred - self.z_mean) / z_std
            mu_gt = (mu_gt - self.z_mean) / z_std

        skill_loss = F.mse_loss(z_pred, mu_gt)
        return skill_loss
    
    @torch.no_grad()
    def predict_skill(self, mappings: List, device) -> Tensor:
        """Predict raw (unnormalized) skill latents, [B, latent_dim]."""
        return self.regress_skill(mappings, device)

    @torch.no_grad()
    def predict(self,mappings: List, device) -> Tensor:
        """Predict one ego-local trajectory per scene, [B, T, 2]."""
        z_pred = self.predict_skill(mappings, device)
        return self.decoder.decode(z_pred)

    
class WayformerPL(pl.LightningModule):

    def __init__(self,config:config) -> None:
        super(WayformerPL,self).__init__()
        self.config = config
        self.model = Wayformer(config)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if param.dim() > 1:
                        if 'weight' in name:
                            torch.nn.init.xavier_normal_(param, gain=np.sqrt(2))
                        elif 'bias' in name:
                            torch.nn.init.constant_(param, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
            elif hasattr(m, 'weight') and m.weight is not None:
                if m.weight.dim() > 1:
                    torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                    if hasattr(m, 'bias') and m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)

        # init_weights above also visits the frozen VAE. Restore its
        # pretrained parameters after initializing the trainable network.
        self.model.decoder.vae.load_state_dict(
            torch.load(
                self.model.decoder._vae_weight_path,
                map_location='cpu',
            )
        )

        self.save_hyperparameters()
    
    def forward(self,mappings:List) -> Tensor:

        return self.model(mappings,self.device)
    
    def training_step(self,batch,batch_idx):
        loss = self(batch)

        if isinstance(batch, list):
            batch_size = len(batch)
        elif isinstance(batch, tuple):
            batch_size = batch[0].shape[0] if hasattr(batch[0], 'shape') else len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1

        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('train_skill_loss', loss, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self,batch,batch_idx) -> Tensor:
        loss = self(batch)
        if isinstance(batch, list):
            batch_size = len(batch)
        elif isinstance(batch, tuple):
            batch_size = batch[0].shape[0] if hasattr(batch[0], 'shape') else len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1

        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('val_skill_loss', loss, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=1e-4,
            eps=1e-4  
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.config.learning_rate*2,  
            steps_per_epoch=1, 
            epochs=self.config.max_epochs,
            pct_start=0.1,  
            div_factor=25.0,  
            final_div_factor=100.0  
        )
        
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def evaluate_model(self, mappings: List, threshold: float = 2.0,
                       show_plot: bool = False) -> Metrics:

        trajs = self.model.predict(mappings, self.device)
        labels = get_from_mapping(mappings, 'labels')
        labels = torch.as_tensor(
            np.asarray(labels),
            device=self.device,
            dtype=torch.float32,
        )

        if show_plot:
            pred = trajs[0].cpu().numpy()
            gt   = labels[0].cpu().numpy()
            plt.figure(figsize=(8, 8))
            matrix = get_from_mapping(mappings, 'matrix')
            if matrix is not None:
                lanes = matrix[0] if isinstance(matrix, list) else matrix
                for lane in lanes:
                    x = lane[:, -3]
                    y = lane[:, -4]
                    tls = lane[0, -8:-4] if lane.shape[1] >= 12 else None
                    if tls is not None and (tls > 0).any():
                        if tls[0] > 0:
                         color = '#FF4444'
                        elif tls[1] > 0:
                         color = '#FFD700'
                        elif tls[2] > 0:
                         color = '#00FF44'
                        else:
                         color = '#FFA500'
                        plt.plot(x, y, color=color, linewidth=2, alpha=0.7)
                    else:
                         plt.plot(x, y, color='#6495ED', linewidth=2, alpha=0.5)
                    x_left = lane[:, -12]
                    y_left = lane[:, -13]
                    plt.plot(x_left, y_left, color='#A9A9A9', linewidth=1, alpha=0.5, linestyle='--')
                    x_right = lane[:, -14]
                    y_right = lane[:, -15]
                    plt.plot(x_right, y_right, color='#A9A9A9', linewidth=1, alpha=0.5, linestyle='--')
            plt.plot(gt[:,0], gt[:,1], 'k-', label='gt')
            plt.plot(
                pred[:, 0],
                pred[:, 1],
                color='#E30FD1',
                alpha=0.9,
                linewidth=3,
                label='prediction',
            )
            agents = get_from_mapping(mappings, 'agents')
            agents_sample = agents[0]
            num_agents = len(agents_sample)
            plt.plot(agents_sample[0,:,0], agents_sample[0,:,1], color='#d33e4c', label='ego_hist', linewidth=2, alpha=0.8)
            plt.scatter(agents_sample[0,0,0], agents_sample[0,0,1], color='#d33e4c', marker='*', s=100, label='ego_start')
            agent_type_colors = {0: '#007672', 1: '#1f77b4', 2: '#ff7f0e'}
            agent_type_labels = {0: 'vehicle', 1: 'pedestrian', 2: 'bicycle'}
            for i in range(1, num_agents):
                if not np.allclose(agents_sample[i,:,0:2], 0):
                    agent_type = int(np.argmax(agents_sample[i,0,4:7]))
                    color_map = {0: agent_type_colors[1], 1: agent_type_colors[0], 2: agent_type_colors[2]}
                    label_map = {0: agent_type_labels[1], 1: agent_type_labels[0], 2: agent_type_labels[2]}
                    color = color_map[agent_type]
                    label = label_map[agent_type] if i == 1 else None
                    plt.plot(agents_sample[i,:,0], agents_sample[i,:,1], color=color, alpha=0.5, linewidth=1)
                    plt.scatter(agents_sample[i,0,0], agents_sample[i,0,1], color=color, marker='o', s=40, label=label)
            plt.axis('equal')
            plt.legend()
            plt.show()
            labels = torch.as_tensor(labels, device=self.device)
        ADE = self.get_ADE(trajs, labels)
        FDE, missed = self.get_FDE(trajs, labels, threshold)

        return Metrics(ADE, FDE, missed)

    @torch.no_grad()
    def get_ADE(self, trajs: Tensor, labels: Tensor) -> float:
        """Average displacement error for one trajectory per scene."""
        distances = torch.linalg.vector_norm(
            trajs[..., :2] - labels[..., :2],
            dim=-1,
        )
        return distances.mean().cpu().item()
    
    @torch.no_grad()
    def get_FDE(self, trajs: Tensor, labels: Tensor,
                threshold: float) -> Tuple[float, int]:
        """Final displacement error and miss count for one trajectory."""
        final_distances = torch.linalg.vector_norm(
            trajs[:, -1, :2] - labels[:, -1, :2],
            dim=-1,
        )
        return (
            final_distances.mean().cpu().item(),
            (final_distances > threshold).sum().cpu().item(),
        )

