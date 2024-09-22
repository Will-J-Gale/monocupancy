import shelve
from io import TextIOWrapper
from argparse import ArgumentParser

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm

from src.utils import Frustum, generate_camera_view_occupancy, occupancy_grid_to_list
from src.dense_lidar_generator import DenseLidarGenerator
from src.constants import (
    STATIC_OBJECT_IDS, NUM_BOX_CLOUD_POINTS, OCCUPANCY_GRID_WIDTH,
    OCCUPANCY_GRID_HEIGHT, OCCUPANCY_GRID_DEPTH, NUM_FUTURE_SAMPLES,
    FRUSTUM_DISTANCE, NUM_VIDEO_FRAMES, DEFAULT_VOXEL_SIZE
)

parser = ArgumentParser()
parser.add_argument("--dataset_root")
parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
parser.add_argument("--num_video_frames", type=int, default=NUM_VIDEO_FRAMES)
parser.add_argument("--output_dir", default=".")
parser.add_argument("--dataset_version", default="v1.0-trainval", choices=["v1.0-trainval", "v1.0-test", "v1.0-mini"])

def process_scene(
        dataset_file:TextIOWrapper,
        scene:dict, 
        nusc:NuScenes, 
        nusc_can:NuScenesCanBus, 
        class_to_colour:dict, 
        voxel_size:float, 
        num_video_frames:int):
    
    lidar_generator = DenseLidarGenerator(
        nusc,
        nusc_can,
        scene,
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
    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)
    class_to_colour = {}
    dataset = shelve.open("occupancy.dataset", flag="n")
    metadata = dict(
        voxel_size=args.voxel_size,
        grid_width=OCCUPANCY_GRID_WIDTH,
        grid_height=OCCUPANCY_GRID_HEIGHT,
        grid_depth=OCCUPANCY_GRID_DEPTH,
        num_width_voxels=round(OCCUPANCY_GRID_WIDTH / args.voxel_size),
        num_height_voxels=round(OCCUPANCY_GRID_HEIGHT / args.voxel_size),
        num_depth_voxels=round(OCCUPANCY_GRID_DEPTH/ args.voxel_size),
        length=0,
        grid_index_order= "XZY"
    )

    dataset["metadata"] = metadata

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        class_to_colour[index] = colour
    
    try:
        for scene in tqdm(nusc.scene, desc="Scene"):
            process_scene(
                dataset,
                scene, 
                nusc, 
                nusc_can, 
                class_to_colour, 
                args.voxel_size, 
                args.num_video_frames, 
            )
    except KeyboardInterrupt:
        return
    
    dataset.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
