from typing import Dict, List, Any
import torch
import numpy as np
import zarr
import os
import shutil
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
class IsaacDataset(BaseImageDataset):
    def __init__(self, 
            zarr_path,  # <--- 必须叫 zarr_path，和 yaml 对应
            shape_meta: dict = None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_cache=False
        ):
        super().__init__()
        
        self.zarr_path = zarr_path # 保存路径供验证集使用
        self.shape_meta = shape_meta
        self.use_cache = use_cache
        
        # 1. 自动识别 Zarr 结构
        print(f"[IsaacDataset] Opening Zarr: {zarr_path}")
        root = zarr.open(zarr_path, mode='r')
        
        # ★★★ 修复 KeyError 'obs'：自动寻找数据组 ★★★
        if 'data' in root:
            data_group = root['data']
        elif 'obs' in root:
            data_group = root['obs']
        else:
            # 可能是直接存的数组
            data_group = root
            
        all_keys = list(data_group.keys())
        print(f"[IsaacDataset] Found keys: {all_keys}")
        
        # 2. 动态构建读取列表
        # 必须包含 state 和 action
        read_keys = []
        if 'state' in all_keys: read_keys.append('state')
        elif 'agent_pos' in all_keys: read_keys.append('agent_pos')
        
        if 'action' in all_keys: read_keys.append('action')
        
        # 自动识别图像 key
        self.img_keys = []
        for k in all_keys:
            if 'img' in k or 'image' in k or 'rgb' in k: 
                read_keys.append(k)
                self.img_keys.append(k)
        
        print(f"[IsaacDataset] Loading Keys into Memory: {read_keys}")
        
        # 3. 读取数据到 ReplayBuffer
        # 注意：ReplayBuffer 需要指定正确的 root group，如果数据在 data 下，path要指向 data
        # 但 ReplayBuffer.copy_from_path 通常处理根目录。
        # 我们这里做一个 trick：如果 keys 在 data 组里，我们传 root 给它可能会找不到
        # 所以最好用 root['data'] (如果是 Zarr Group 对象) 或者直接传路径
        
        # 简单处理：我们假设 ReplayBuffer 能处理标准的 Zarr 结构
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=read_keys)
        
        # 4. 设置采样器
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        # ★★★ 修复验证集路径报错 ★★★
        # 传入 self.zarr_path (字符串)，而不是 self.replay_buffer.root (对象)
        val_set = IsaacDataset(
            zarr_path=self.zarr_path, 
            shape_meta=self.shape_meta,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            val_ratio=1.0, # 这里的逻辑不重要，下面会覆盖 sampler
            use_cache=self.use_cache
        )
        # 共享内存，避免重新读取
        val_set.replay_buffer = self.replay_buffer
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # 1. 准备数据字典
        data = {'action': self.replay_buffer['action']}
        
        # 2. ★★★ 关键修复：统一把 'state' 映射为 'agent_pos' ★★★
        # 无论 Zarr 里叫 state 还是 agent_pos，我们都把它塞给 'agent_pos' 这个 key
        # 这样就和 yaml 里的定义对应上了
        if 'state' in self.replay_buffer:
            data['agent_pos'] = self.replay_buffer['state']
        elif 'agent_pos' in self.replay_buffer:
            data['agent_pos'] = self.replay_buffer['agent_pos']
            
        # 3. 拟合低维数据
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 4. 注册图像 Identity (保持不变)
        for img_key in self.img_keys:
            normalizer[img_key] = SingleFieldLinearNormalizer.create_identity()
            
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample
        
        # 映射 state -> agent_pos (这是 yaml 里定义的 key)
        state_data = data['state'] if 'state' in data else data['agent_pos']
        
        torch_data = {
            'obs': {
                'agent_pos': torch.from_numpy(state_data), 
            },
            'action': torch.from_numpy(data['action'])
        }
        
        # 动态处理所有图像
        for img_key in self.img_keys:
            # (T,H,W,C) -> (T,C,H,W) -> float32 [0,1]
            torch_data['obs'][img_key] = torch.from_numpy(data[img_key]).permute(0, 3, 1, 2).float() / 255.0
        
        return torch_data