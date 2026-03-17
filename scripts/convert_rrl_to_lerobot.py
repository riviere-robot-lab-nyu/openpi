""" 
Script to convert RRL hdf5 data to LeRobot dataset v2.0 format.

Based on examples/aloha_real/convert_aloha_data_to_lerobot.py

Example usage: uv run scripts/convert_rrl_to_lerobot.py --raw-dir /path/to/raw/data --repo-id /path/blah/blah/blah (This may need to change for local dataset)
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
        repo_id: str, # This is following lerobot convention. when the first term is "local" -> /local/repo/ repo_id becomes path, which is what we will use
        robot_type: str = "rrl_m3", 
        mode: Literal["video", "image"] = "video",
        *,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": [

                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "left_carriage_joint",
                "lin_vel_x",
                "lin_vel_y",
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "left_carriage_joint",
                "F_x",
                "F_y",
                "T_z",
            ] 
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        },

        "observation.images.base": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s = dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def populate_dataset(
        dataset: LeRobotDataset,
        hdf5_file: str,
        task: str,
) -> LeRobotDataset:
    with h5py.File(hdf5_file, "r") as f:
        episodes = list(f["data"].keys())
        n_episodes = len(episodes)
        print(f"Processing {n_episodes} episodes")
        for ep in episodes:
            ep_data = f["data"][ep]
            state = torch.from_numpy(ep_data["observation/state"][:])
            raw_acts = ep_data["actions"][:]
            #action = torch.from_numpy(np.concatenate([raw_acts[:,3:], raw_acts[:,:3]],axis=1))
            action = torch.from_numpy(raw_acts)
            raw_base = ep_data["observation/images/base"][:]
            raw_wrist = ep_data["observation/images/wrist"][:]

            base_imgs = torch.from_numpy(raw_base).permute(0, 3, 1, 2)
            wrist_imgs = torch.from_numpy(raw_wrist).permute(0, 3, 1, 2)

            num_frames = action.shape[0]
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "observation.images.base": base_imgs[i],
                    "observation.images.wrist": wrist_imgs[i],
                    "task": task,
                }

                dataset.add_frame(frame)
            print("Finished episode: ")
            print(ep)
            dataset.save_episode()
    
    return dataset

def port_rrl(
        hdf5_file: str,
        repo_id: str,
        task: str = "Navigate to the cabinet and open the top drawer",
        *,
        mode: Literal["video", "image"] = "video",
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if not Path(hdf5_file).exists():
        raise FileNotFoundError(f"Could not find your HDF5 file at {hdf5_file}")

    dataset = create_empty_dataset(
        repo_id=repo_id,
        mode=mode,
        dataset_config=dataset_config,
    )

    dataset = populate_dataset(
        dataset,
        hdf5_file=hdf5_file,
        task=task,
    )

    #dataset.consolidate()

    print(f"Dataset successfully built locally at: {repo_id}")


if __name__ == "__main__":
    tyro.cli(port_rrl)


