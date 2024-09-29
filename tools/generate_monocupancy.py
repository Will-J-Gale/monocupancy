import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shelve
from typing import List
from shelve import DbfilenameShelf
from argparse import ArgumentParser

from tqdm import tqdm

from src.utils import generate_camera_view_occupancy, occupancy_grid_to_list
from src.dense_lidar_generator import DenseLidarGenerator
from src.constants import (
    STATIC_OBJECT_IDS, NUM_BOX_CLOUD_POINTS, OCCUPANCY_GRID_WIDTH,
    OCCUPANCY_GRID_HEIGHT, OCCUPANCY_GRID_DEPTH, NUM_FUTURE_SAMPLES,
    FRUSTUM_DISTANCE, NUM_VIDEO_FRAMES, DEFAULT_VOXEL_SIZE
)

parser = ArgumentParser()
parser.add_argument("--nuscenes_dataset", default="nuscenes_simplified.dataset")
parser.add_argument("--output_path", default="occupancy.dataset")
parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
parser.add_argument("--num_video_frames", type=int, default=NUM_VIDEO_FRAMES)
parser.add_argument("--output_dir", default=".")

def process_scene(
        scene_samples:List[dict], 
        dataset_file:DbfilenameShelf,
        class_to_colour:dict, 
        voxel_size:float, 
        num_video_frames:int):
    
    lidar_generator = DenseLidarGenerator(
        scene_samples,
        NUM_FUTURE_SAMPLES,
        num_video_frames,
        class_to_colour,
        STATIC_OBJECT_IDS,
        NUM_BOX_CLOUD_POINTS,
        FRUSTUM_DISTANCE
    )

    for dense_lidar, camera in tqdm(lidar_generator, desc="Frame", leave=False):
        occupancy = generate_camera_view_occupancy(
            dense_lidar, 
            camera.transform, 
            OCCUPANCY_GRID_WIDTH, 
            OCCUPANCY_GRID_DEPTH, 
            OCCUPANCY_GRID_HEIGHT, 
            voxel_size, 
            camera
        )

        occupancy_points_list, occupancy_colours = occupancy_grid_to_list(occupancy)
        data = {
            "image_paths": camera.image_paths,
            "occupancy": occupancy_points_list,
            "occupancy_colours": occupancy_colours
        }

        metadata = dataset_file["metadata"]
        index = metadata["length"]
        dataset_file[str(index)] = data

        metadata["length"] = index + 1
        dataset_file["metadata"] = metadata

def main(args):
    nuscenes_data = shelve.open(args.nuscenes_dataset, flag="r")
    class_to_colour = nuscenes_data["class_to_colour"]
    num_scenes = nuscenes_data["num_scenes"]

    dataset = shelve.open(args.output_path, flag="n")
    dataset["metadata"] = dict(
        voxel_size=args.voxel_size,
        grid_width=OCCUPANCY_GRID_WIDTH,
        grid_height=OCCUPANCY_GRID_HEIGHT,
        grid_depth=OCCUPANCY_GRID_DEPTH,
        num_width_voxels=round(OCCUPANCY_GRID_WIDTH / args.voxel_size),
        num_height_voxels=round(OCCUPANCY_GRID_HEIGHT / args.voxel_size),
        num_depth_voxels=round(OCCUPANCY_GRID_DEPTH/ args.voxel_size),
        length=0
    )
    
    try:
        for i in tqdm(range(num_scenes), desc="Scene"):
            scene_samples = nuscenes_data[str(i)]
            process_scene(
                scene_samples,
                dataset,
                class_to_colour, 
                args.voxel_size, 
                args.num_video_frames, 
            )
    except KeyboardInterrupt:
        return
    
    dataset.close()
    nuscenes_data.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
