import platform
import random
from functools import partial
from typing import Optional, Union

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# from .beat_dataset import BEATDataset

from .samplers import DistributedSampler

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def build_dataset(
    cfg: Union[dict, list, tuple], default_args: Optional[Union[dict, None]] = None
):
    """ "Build dataset by the given config."""
    from .dataset_wrappers import (
        ConcatDataset,
        RepeatDataset,
    )

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    else:
        # dataset = build_from_cfg(cfg, DATASETS, default_args)
        # breakpoint()
        dataset_name = cfg.pop("type")
        dataset = DATASETS.get(dataset_name)(**cfg)

    return dataset


def beatx_collate_fn(batch):
    """Collate function for BEAT dataset."""
    # breakpoint()
    # print("batch", [b["gesture_labels"] for b in batch])
    adapted_batch = {
        "motion": collate([sample["motion"] for sample in batch]),
        "motion_upper": collate([sample["motion_upper"] for sample in batch]),
        "motion_lower": collate([sample["motion_lower"] for sample in batch]),
        "motion_face": collate([sample["motion_face"] for sample in batch]),
        "motion_hands": collate([sample["motion_hands"] for sample in batch]),

        "motion_length": [sample["motion_length"] for sample in batch],
        "motion_mask": collate([sample["motion_mask"] for sample in batch]),

        "contact": collate([sample["contact"] for sample in batch]),
        "trans": collate([sample["trans"] for sample in batch]),
        "facial": collate([sample["facial"] for sample in batch]),
        "beta": collate([sample["beta"] for sample in batch]),

        "raw_audio": collate([sample["raw_audio"] for sample in batch]),
        "audio": collate([sample["audio"] for sample in batch]),

        "raw_word": [sample["raw_word"] for sample in batch],
        "word": collate([sample["word"] for sample in batch]),
        "text_features": [sample["text_feature"] for sample in batch],
        "text_segments": [sample["text_segments"] for sample in batch],

        "speaker_ids": collate([sample["speaker_id"] for sample in batch]),
        "emo": collate([sample["emo"] for sample in batch]),
        "gesture_labels": [sample["gesture_labels"] for sample in batch],
        "sem_score": collate([sample["sem_score"] for sample in batch]),
        "discourse": [sample["discourse"] for sample in batch],
        "prominence": [sample["prominence"] for sample in batch],
        
        "sample_idx": [sample["sample_idx"] for sample in batch],
        "sample_name": [sample["sample_name"] for sample in batch],
    }
    return adapted_batch


def build_dataloader(
    dataset: Dataset,
    samples_per_gpu: int,
    workers_per_gpu: int,
    num_gpus: Optional[int] = 1,
    dist: Optional[bool] = True,
    shuffle: Optional[bool] = True,
    round_up: Optional[bool] = True,
    seed: Optional[Union[int, None]] = None,
    persistent_workers: Optional[bool] = True,
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int, optional): Number of GPUs. Only used in non-distributed
            training.
        dist (bool, optional): Distributed training/test or not. Default: True.
        shuffle (bool, optional): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool, optional): Whether to round up the length of dataset by
            adding extra samples to make it evenly divisible. Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, round_up=round_up
        )
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(beatx_collate_fn),
        pin_memory=False,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        persistent_workers=persistent_workers,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """Init random seed for each worker."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
