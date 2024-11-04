import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shelve
from typing import List
from shelve import DbfilenameShelf
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.utils import generate_camera_view_occupancy, occupancy_grid_to_list, occupancy_indicies_to_numpy
from src.dense_lidar_generator import DenseLidarGenerator
from src.constants import (
    STATIC_OBJECT_IDS, NUM_BOX_CLOUD_POINTS, OCCUPANCY_GRID_WIDTH,
    OCCUPANCY_GRID_HEIGHT, OCCUPANCY_GRID_DEPTH, NUM_FUTURE_SAMPLES,
    FRUSTUM_DISTANCE, NUM_VIDEO_FRAMES, DEFAULT_VOXEL_SIZE, GROUND_CLEARANCE
)

parser = ArgumentParser()
parser.add_argument("--nuscenes_dataset", default="nuscenes_simplified.dataset")
parser.add_argument("--output_path", default="occupancy.dataset")
parser.add_argument("--occupancy_data_output_dir", default="voxel_grids")
parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
parser.add_argument("--num_video_frames", type=int, default=NUM_VIDEO_FRAMES)

def process_scene(
        scene_samples:List[dict], 
        dataset_file:DbfilenameShelf,
        occupancy_data_output_dir:str,
        class_to_colour:dict, 
        num_video_frames:int,
        occupancy_grid_template:o3d.geometry.VoxelGrid):
    
    lidar_generator = DenseLidarGenerator(
        scene_samples,
        NUM_FUTURE_SAMPLES,
        num_video_frames,
        class_to_colour,
        STATIC_OBJECT_IDS,
        NUM_BOX_CLOUD_POINTS,
        FRUSTUM_DISTANCE
    )

    for dense_lidar, labels, camera in tqdm(lidar_generator, desc="Frame", leave=False):
        occupancy_result = generate_camera_view_occupancy(
            dense_lidar, 
            camera.transform, 
            OCCUPANCY_GRID_WIDTH, 
            OCCUPANCY_GRID_DEPTH, 
            OCCUPANCY_GRID_HEIGHT, 
            camera,
            occupancy_grid_template
        )

        metadata = dataset_file["metadata"]
        index = metadata["length"]

        occupancy_points_list, occupancy_colours = occupancy_grid_to_list(occupancy_result.occupancy_grid)
        occupancy_numpy = occupancy_indicies_to_numpy(
            occupancy_points_list, 
            (metadata["num_width_voxels"], metadata["num_height_voxels"], metadata["num_depth_voxels"])
        )
        occupancy_data_path = os.path.join(occupancy_data_output_dir, f"{index}.npz")
        np.savez_compressed(
            occupancy_data_path, 
            occupancy_grid=occupancy_numpy,
            occupancy_indicies=np.array(occupancy_points_list, dtype=np.uint8),
            occupancy_colours=np.array(occupancy_colours, dtype=np.uint8),
            occupancy_labels=np.array(labels, dtype=np.uint8)
        )
        data = {
            "image_paths": camera.image_paths,
            "occupancy_path": occupancy_data_path
        }
        dataset_file[str(index)] = data

        
        total_voxels = metadata["num_width_voxels"] + metadata["num_height_voxels"] + metadata["num_depth_voxels"]
        metadata["length"] = index + 1
        metadata["num_occupied"] += len(occupancy_points_list)
        metadata["num_not_occupied"] += total_voxels - len(occupancy_points_list)
        dataset_file["metadata"] = metadata

def main(args):
    nuscenes_data = shelve.open(args.nuscenes_dataset, flag="r")
    class_to_colour = nuscenes_data["class_to_colour"]
    num_scenes = nuscenes_data["num_scenes"]

    occupancy_data_output_dir = args.occupancy_data_output_dir
    os.makedirs(occupancy_data_output_dir, exist_ok=True)

    dataset = shelve.open(args.output_path, flag="n")
    dataset["metadata"] = dict(
        voxel_size=args.voxel_size,
        grid_width=OCCUPANCY_GRID_WIDTH,
        grid_height=OCCUPANCY_GRID_HEIGHT,
        grid_depth=OCCUPANCY_GRID_DEPTH,
        num_width_voxels=round(OCCUPANCY_GRID_WIDTH / args.voxel_size),
        num_height_voxels=round(OCCUPANCY_GRID_HEIGHT / args.voxel_size),
        num_depth_voxels=round(OCCUPANCY_GRID_DEPTH/ args.voxel_size),
        length=0,
        num_occupied=0,
        num_not_occupied=0,
    )

    occupancy_grid_template = o3d.geometry.VoxelGrid.create_dense([-OCCUPANCY_GRID_WIDTH/2, 0, -GROUND_CLEARANCE], [255, 255, 255], args.voxel_size, OCCUPANCY_GRID_WIDTH, OCCUPANCY_GRID_DEPTH, OCCUPANCY_GRID_HEIGHT)
    [occupancy_grid_template.remove_voxel(voxel.grid_index) for voxel in occupancy_grid_template.get_voxels()]
    
    try:
        for i in tqdm(range(num_scenes), desc="Scene"):
            scene_samples = nuscenes_data[str(i)]
            process_scene(
                scene_samples,
                dataset,
                occupancy_data_output_dir,
                class_to_colour, 
                args.num_video_frames, 
                occupancy_grid_template
            )
    except KeyboardInterrupt:
        return
    
    dataset.close()
    nuscenes_data.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
