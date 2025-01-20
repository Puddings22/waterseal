# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

try:
    from decord import VideoReader, cpu
    decord_available = True
except ImportError:
    VideoReader = None
    decord_available = False

import tensorflow as tf
import numpy as np
import io
from PIL import Image
# 1) Import torch-tfrecord for TFRecord handling
from torch_tfrecord import TFRecordDataset, ExampleReader

from ..utils.dist import is_dist_avail_and_initialized
from .datasets import CocoImageIDWrapper, ImageFolder, VideoDataset


##############################################
#  [A] CUSTOM TFRECORD DATASET & DATALOADER
##############################################

class YouTube8MDataset(torch.utils.data.Dataset):
    """
    Example dataset class for handling TFRecords that contain 'video' and 'label' features.
    If your TFRecords store data differently, adjust 'decode_video' accordingly.
    """
    def __init__(self, tfrecord_files, transform=None):
        """
        Args:
            tfrecord_files (list): List of paths to TFRecord files.
            transform (callable, optional): Optional transform to apply to each sample (frames).
        """
        super().__init__()
        self.transform = transform

        # Create a TFRecordDataset from the file paths
        self.dataset = TFRecordDataset(
            tfrecord_files,
            index_path=None,  # or path to an index if you have one
            shuffle=False,
            transform=ExampleReader(
                features={
                    "video": tf.io.FixedLenFeature([], tf.string),
                    "label": tf.io.FixedLenFeature([], tf.int64),
                    # Add other features here if your TFRecord has them
                }
            )
        )
        # Shuffle buffer for the dataset
        self.dataset = self.dataset.shuffle(buffer_size=1000)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the parsed example from TFRecordDataset
        example = self.dataset[idx]
        video_bytes = example["video"]
        label = example["label"]

        # Decode the video bytes into frames
        video_tensor = self.decode_video(video_bytes)

        # Apply transforms if any (may need customizing if transforms
        # expect (C,H,W) instead of (num_frames,C,H,W))
        if self.transform is not None:
            # Example: you might want to transform each frame
            # If transforms expect a single image, you may need to loop
            # over frames or design a specialized transform
            # This example calls the transform on the entire (num_frames,C,H,W)
            video_tensor = self.transform(video_tensor)

        return video_tensor, label

    def decode_video(self, video_bytes):
        """
        Example logic that assumes 'video' feature is made up of multiple JPEG frames
        stored in sequence. If your TFRecord is encoded differently (e.g., raw video),
        adapt this method accordingly.
        """
        frame_list = []
        frame_data = io.BytesIO(video_bytes.numpy())

        while True:
            try:
                frame = Image.open(frame_data).convert("RGB")
                # Convert from PIL Image -> (C,H,W) float tensor
                frame_t = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
                frame_list.append(frame_t)
            except EOFError:
                # Reached the end of the byte stream
                break
            except Exception as e:
                print(f"decode_video error reading frame: {e}")
                break

        if not frame_list:
            # Return a default if no frames
            return torch.zeros(3, 256, 256)

        # Stack frames -> shape: (num_frames, C, H, W)
        video_tensor = torch.stack(frame_list, dim=0)
        return video_tensor


def get_tfrecord_dataloader(
    data_dir: str,
    transform=None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Create a PyTorch DataLoader for TFRecord files located in data_dir.
    Looks for *.tfrecord files, uses YouTube8MDataset to parse them.
    """
    # Gather all TFRecord files in data_dir
    tfrecord_files = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord")))
    if not tfrecord_files:
        warnings.warn(f"No TFRecord files found in directory: {data_dir}")

    dataset = YouTube8MDataset(tfrecord_files, transform=transform)

    # If distributed training is in use, create a distributed sampler
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    return loader


##############################################
#  [B] EXISTING CODE FOR IMAGE / MP4 LOADING
##############################################

def load_video(fname):
    """
    Load full video.
    Args:
        fname (str): The path to the video file.
        num_workers (int): The number of worker threads to use for video loading. Defaults to 8.
    Returns:
        tuple: A tuple containing the loaded video frames as a PyTorch tensor (Frames, H, W, C) and a mask tensor.
    """
    if not os.path.exists(fname):
        warnings.warn(f'video path not found {fname=}')
        return [], None
    _fsize = os.path.getsize(fname)
    if _fsize < 1 * 1024:  # avoid hanging issue
        warnings.warn(f'video too short {fname=}')
        return [], None
    if decord_available:
        vr = VideoReader(fname, num_threads=8, ctx=cpu(0))
        vid_np = vr.get_batch(range(len(vr))).asnumpy()
        vid_np = vid_np.transpose(0, 3, 1, 2) / 255.0  # normalize to 0 - 1
        vid_pt = torch.from_numpy(vid_np).float()
    else:
        vid_pt, _, _ = torchvision.io.read_video(fname, output_format="TCHW")
        vid_pt = vid_pt.float() / 255.0
    return vid_pt


def get_dataloader(
    data_dir: str,
    transform: callable = torchvision.transforms.ToTensor(),
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8
) -> DataLoader:
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    return dataloader


def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    images, masks = zip(*batch)
    images = torch.stack(images)

    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    # Pad each mask tensor to have 'max_masks' masks
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        union_mask = torch.max(mask, dim=0).values
        # Calculate the inverse of the union mask
        inverse_mask = ~union_mask

        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(
                mask, pad=(0, 0, 0, 0, 0, pad_size),
                mode='constant', value=0
            )
        else:
            padded_mask = mask

        # If you wish to append the inverse mask, uncomment:
        # padded_mask = torch.cat([padded_mask, inverse_mask.unsqueeze(0)], dim=0)

        padded_masks.append(padded_mask)

    masks = torch.stack(padded_masks)
    return images, masks


def get_dataloader_segmentation(
    data_dir: str,
    ann_file: str,
    transform: callable,
    mask_transform: callable,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    random_nb_object=True,
    multi_w=False,
    max_nb_masks=4
) -> DataLoader:
    """ Get dataloader for COCO dataset. """
    if "coco" in data_dir:
        dataset = CocoImageIDWrapper(
            root=data_dir,
            annFile=ann_file,
            transform=transform,
            mask_transform=mask_transform,
            random_nb_object=random_nb_object,
            multi_w=multi_w,
            max_nb_masks=max_nb_masks
        )
    else:
        dataset = ImageFolder(path=data_dir, transform=transform, mask_transform=mask_transform)

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    return dataloader


def get_video_dataloader(
    data_dir: str,
    transform: callable = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 8,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Get dataloader for the videos in data_dir (must contain .mp4 or other recognized video files).
    Existing approach using VideoDataset (not TFRecords).
    """
    dataset_kwargs.update({
        'folder_paths': [data_dir],
        'transform': transform
    })

    dataset = VideoDataset(num_workers=num_workers, **dataset_kwargs)
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=drop_last)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True, drop_last=drop_last)
    return dataloader


# Test the VideoLoader class (optional self-test)
if __name__ == "__main__":
    video_folder_path = "./assets/videos/"

    video_dataloader = get_video_dataloader(
        data_dir=video_folder_path,
        frames_per_clip=16,
        frame_step=4,
        num_clips=4,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        output_resolution=(250, 250),
        flatten_clips_to_frames=True,
    )
    for video_batch, masks_batch, frames_positions in video_dataloader:
        print(f"loaded a batch of {video_batch.shape} size, each frame in .mp4 context")
        print(video_batch.shape)
        print(frames_positions)
        break

    video_dataloader = get_video_dataloader(
        data_dir=video_folder_path,
        frames_per_clip=16,
        frame_step=4,
        num_clips=4,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        output_resolution=(250, 250),
        flatten_clips_to_frames=False,
    )
    for video_batch, masks_batch, frames_positions in video_dataloader:
        print(f"loaded a batch of size {video_batch.shape[0]}, each has {video_batch.shape[1]} clips")
        print(video_batch.shape)
        print(frames_positions)
        break

    print("Video dataloader test completed.")
