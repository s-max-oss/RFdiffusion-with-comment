# rfdiffusion/inference/character_constraints.py
import torch
import numpy as np
from .character_to_points import text_to_3d_points

class CharacterConstrainedSampler:
    def __init__(self, base_sampler, character_text):
        self.base_sampler = base_sampler
        self.character_text = character_text
        
        # 从配置中获取参数
        if hasattr(base_sampler, '_conf'):
            self._conf = base_sampler._conf
            scale = self._conf.inference.get('character_scale', 10.0)
            density = self._conf.inference.get('character_density', 5)
            spacing = self._conf.inference.get('character_spacing', 15.0)
            self.character_points = text_to_3d_points(character_text, spacing)
        else:
            self.character_points = text_to_3d_points(character_text)
            
        # 转换为torch tensor
        self.character_points = torch.from_numpy(self.character_points).float()
            
    def sample_init(self):
        return self.base_sampler.sample_init()
        
    def sample_step(self, t, x_t, seq_init, final_step):
        # 在采样步骤中加入字符约束
        px0, x_t, seq_t, plddt = self.base_sampler.sample_step(
            t=t, x_t=x_t, seq_init=seq_t, final_step=final_step
        )
        
        # 应用字符形状约束
        x_t = self.apply_character_constraints(x_t, t)
        
        return px0, x_t, seq_t, plddt
        
    def apply_character_constraints(self, x_t, t):
        """
        根据字符点云约束调整结构
        """
        # 只在后期步骤中应用约束
        if t < 50 and len(self.character_points) > 0:
            with torch.no_grad():
                # 简单示例：将末端原子拉向字符点云
                # 获取backbone原子坐标 (batch, length, 4, 3)
                if x_t.shape[-2] >= 4:  # 确保有backbone原子
                    # 获取CA原子坐标 (batch, length, 3)
                    ca_coords = x_t[:, :, 1, :]  # CA是第二个原子
                    
                    # 计算当前结构中心
                    structure_center = ca_coords.mean(dim=(0, 1))
                    
                    # 将字符点云移到结构中心附近
                    target_points = self.character_points.clone()
                    if len(target_points) > 0:
                        point_center = target_points.mean(dim=0)
                        target_points = target_points - point_center + structure_center
                        
                        # 简单地将一些原子推向目标点
                        # 这里只是示范，实际应该更精细地处理
                        num_atoms = ca_coords.shape[1]
                        num_targets = min(len(target_points), num_atoms)
                        
                        if num_targets > 0:
                            # 将末尾的原子推向目标点
                            for i in range(num_targets):
                                atom_idx = num_atoms - num_targets + i
                                if atom_idx < num_atoms:
                                    # 加权移动，保留一定的结构自由度
                                    weight = min(1.0, (50 - t) / 50.0) * 0.1
                                    ca_coords[:, atom_idx, :] = (
                                        (1 - weight) * ca_coords[:, atom_idx, :] + 
                                        weight * target_points[i].to(ca_coords.device)
                                    )
                                    
                                    # 同步更新backbone其他原子
                                    x_t[:, atom_idx, 1, :] = ca_coords[:, atom_idx, :]
        
        return x_t
    
    @property
    def inf_conf(self):
        return self.base_sampler.inf_conf
        
    @property
    def t_step_input(self):
        return self.base_sampler.t_step_input
        
    @property
    def binderlen(self):
        return getattr(self.base_sampler, 'binderlen', None)
        
    @property
    def chain_idx(self):
        return getattr(self.base_sampler, 'chain_idx', None)
        
    @property
    def contig_map(self):
        return getattr(self.base_sampler, 'contig_map', None)