# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Subset
import torchvision
from torchvision.datasets import ImageNet
from tqdm import tqdm
from omegaconf import DictConfig
import math

from .ffhq import ImageFolder, FFHQ
from .transforms import create_transforms

import torchvision.transforms as transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))

class VariableBatchDistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
  def __init__(self, dataset, batch_sizes, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
    """
    Args:
      dataset: MultiHeightScoreDataset instance
      batch_sizes: Dict mapping bucket names to batch sizes
      num_replicas: Number of distributed processes
      rank: Rank of current process
      shuffle: Whether to shuffle samples within buckets
      seed: Random seed for shuffling
      drop_last: Whether to drop last incomplete batch
    """
    super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

    # Store batch size per bucket
    self.batch_sizes = batch_sizes
    self.bucket_indices = dataset.bucket_indices

    # Calculate number of samples per bucket that each process will handle
    self.bucket_samples_per_replica = {}
    for bucket, indices in self.bucket_indices.items():
      num_samples = len(indices)
      batch_size = self.batch_sizes[bucket]
      
      # Ensure even distribution across replicas by rounding down to nearest batch_size multiple
      samples_per_rank = (num_samples // self.num_replicas // batch_size) * batch_size
      self.bucket_samples_per_replica[bucket] = samples_per_rank

    print("Bucket samples per replica:", self.bucket_samples_per_replica)

    # Total length is sum of samples across all buckets
    self.total_size = sum(self.bucket_samples_per_replica.values())

  def __iter__(self):
    # Deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    indices = []
    for bucket, bucket_indices in self.bucket_indices.items():
      bucket_indices = bucket_indices.copy()
      if self.shuffle:
        rand_indices = torch.randperm(len(bucket_indices), generator=g).tolist()
        bucket_indices = [bucket_indices[i] for i in rand_indices]

      batch_size = self.batch_sizes[bucket]
      samples_per_replica = self.bucket_samples_per_replica[bucket]
      
      # Ensure even distribution by using same number of samples per replica
      start_idx = self.rank * samples_per_replica
      end_idx = start_idx + samples_per_replica
      rank_indices = bucket_indices[start_idx:end_idx]

      # Create batches
      for i in range(0, len(rank_indices), batch_size):
        batch_indices = rank_indices[i:i + batch_size]
        if len(batch_indices) == batch_size:  # Only add complete batches
          indices.append(batch_indices)

    if self.shuffle:
      # Shuffle order of batches
      rand_batch_indices = torch.randperm(len(indices), generator=g).tolist()
      indices = [indices[i] for i in rand_batch_indices]

    return iter(indices)

  def __len__(self):
    # Calculate total number of batches across all buckets
    total_batches = 0
    for bucket, samples_per_replica in self.bucket_samples_per_replica.items():
      batch_size = self.batch_sizes[bucket]
      total_batches += math.ceil(samples_per_replica / batch_size)
    return total_batches



class ScoreDataset(ImageFolder):
    def __init__(self, root, split='train', **kwargs):
        train_list_file = f'{root}/train.txt'
        val_list_file = f'{root}/test.txt'
        super().__init__(root, train_list_file, val_list_file, split, **kwargs)

class MultiHeightScoreDataset(ScoreDataset):
    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, split, **kwargs)
        
        heights = []
        valid_indices = []
        for i, sample in enumerate(tqdm(self.samples, desc='Loading image heights')):
            img = self.loader(sample)
            height = img.height
            if img.width < height:  # Drop portrait images
                continue
            if 70 <= height <= 390:  # Only keep samples within valid height range
                heights.append(height)
                valid_indices.append(i)
        # Filter samples to only keep valid ones
        self.samples = [self.samples[i] for i in valid_indices]
        self.heights = heights
        
        # Create bucket indices that will be used by DistributedBucketSampler
        self.bucket_indices = {
            '70_130': [i for i, h in enumerate(heights) if 70 <= h < 130], # 64px crop
            '130_260': [i for i, h in enumerate(heights) if 130 <= h < 260], # 128px crop
            '260_360': [i for i, h in enumerate(heights) if 260 <= h < 360], # 256px crop
            '360_390': [i for i, h in enumerate(heights) if 360 <= h <= 390] # 352px crop
        }

        # Get indices for images with height >= 260px
        self.big_indices = self.bucket_indices['260_360'] + self.bucket_indices['360_390']

        # Create transforms for each bucket
        self.bucket_transforms = {
            '70_130': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '130_260': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '260_360': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            '360_390': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(352),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        }

    def get_bucket_sizes(self):
        return {k: len(v) for k, v in self.bucket_indices.items()}

    def __getitem__(self, index, with_transform=True):
        sample, _ = super().__getitem__(index, with_transform=False)
        height = self.heights[index]
        
        if with_transform:
            if 70 <= height < 130:
                sample = self.bucket_transforms['70_130'](sample)
            elif 130 <= height < 260:
                sample = self.bucket_transforms['130_260'](sample)
            elif 260 <= height < 360:
                sample = self.bucket_transforms['260_360'](sample)
            elif 360 <= height <= 390:
                sample = self.bucket_transforms['360_390'](sample)
                
        return sample, 0
    
    def get_big_items(self, idx):
        # Get the image at specified index
        sample, _ = super().__getitem__(self.big_indices[idx], with_transform=False)
        sample = self.bucket_transforms['260_360'](sample)
        
        return sample, 0

# class Grandstaff(ImageFolder):
#     train_list_file = 'data/Olimpic_grandstaff_128_gray/train.txt'
#     val_list_file = 'data/Olimpic_grandstaff_128_gray/test.txt'

#     def __init__(self, root, split='train', **kwargs):
#         super().__init__(root, Grandstaff.train_list_file, Grandstaff.val_list_file, split, **kwargs)


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'data/imagenet'

        def target_transform(_):
            return 0
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn, target_transform=target_transform)
        dataset_val = ImageNet(root, split='val', transform=transforms_val, target_transform=target_transform)
    elif config.dataset.type == 'ffhq':
        root = root if root else 'data/ffhq'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        from .lsun import LSUNClass  # requires the optional lmdb package
        root = root if root else 'data/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, transform=transforms_val)
    elif config.dataset.type == 'Olimpic_grandstaff_128_gray':
        root = root if root else 'data/Olimpic_grandstaff_128_gray'
        dataset_trn = ScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = ScoreDataset(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'LSDSQ_flattened_240_gray':
        root = root if root else 'data/LSDSQ_flattened_240_gray'
        dataset_trn = ScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = ScoreDataset(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSD_360anchored_gray', 'LSD_360anchored_gray_debug', 'LSD_360anchored_gray_yolo']:
        root = root if root else f'data/{config.dataset.type}'
        assert isinstance(config.experiment.batch_size, DictConfig)
        dataset_trn = MultiHeightScoreDataset(root, split='train', transform=transforms_trn)
        dataset_val = MultiHeightScoreDataset(root, split='val', transform=transforms_val)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val
