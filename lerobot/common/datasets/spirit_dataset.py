import copy
from typing import Dict, Optional

import os
import numpy as np
import torch
from tqdm import trange, tqdm
from dataclasses import dataclass, field

from lerobot.common.datasets.base_dataset import BaseDataset
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.transforms import DeltaActions


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


class SpiritDataset(BaseDataset):
    def __init__(self,
        cfg: TrainPipelineConfig,
        lerobot_dataset: LeRobotDataset,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):

        self.lerobot_dataset = lerobot_dataset
        self.transforms = []

        # add transform for SpiritDataset
        if cfg.policy.use_delta_joint_actions_aloha:
            mask = make_bool_mask(-2, 12)
            self.transforms.append(DeltaActions(mask=mask))

    
    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.lerobot_dataset.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return self.lerobot_dataset.num_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return self.lerobot_dataset.num_episodes
    
    @property
    def meta(self) -> LeRobotDatasetMetadata:
        """LeRobotDatasetMetadata"""
        return self.lerobot_dataset.meta
    
    @property
    def episode_data_index(self) -> dict[str, torch.Tensor]:
        """ {
        "from": torch.LongTensor(...),
        "to": torch.LongTensor(...),
        } """
        return self.lerobot_dataset.episode_data_index
    
    def __len__(self):
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        data = self.lerobot_dataset.__getitem__(idx)

        for transform in self.transforms:
            data = transform(data)

        return data
