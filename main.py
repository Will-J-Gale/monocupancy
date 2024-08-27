import pickle
from io import TextIOWrapper
from argparse import ArgumentParser

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm

from src.utils import Frustum, generate_camera_view_occupancy, occupancy_grid_to_list
from src.dense_lidar_generator import DenseLidarGenerator

parser = ArgumentParser()
parser.add_argument("--dataset_root")
parser.add_argument("--voxel_size", type=float, default=0.30)
parser.add_argument("--num_video_frames", type=int, default=4)
parser.add_argument("--output_dir", default=".")
parser.add_argument("--dataset_version", default="v1.0-trainval")

STATIC_OBJECT_IDS = [ 0, 13, 24, 25, 26, 27, 28, 29, 30 ]
NUM_BOX_CLOUD_POINTS = 4000
OCCUPANCY_GRID_WIDTH = 35
OCCUPANCY_GRID_DEPTH = 35
OCCUPANCY_GRID_HEIGHT = 10
NUM_FUTURE_SAMPLES = 15
FRUSTUM_DISTANCE = 100

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
        frustum = Frustum(camera.frustum_geometry.points)
        occupancy, _ = generate_camera_view_occupancy(
            dense_lidar, 
            camera.transform, 
            OCCUPANCY_GRID_WIDTH, 
            OCCUPANCY_GRID_DEPTH, 
            OCCUPANCY_GRID_HEIGHT, 
            voxel_size, 
            frustum
        )

        occupancy_points_list, occupancy_colours = occupancy_grid_to_list(occupancy)

        data = {
            "image_paths": camera.image_paths,
            "occupancy": occupancy_points_list,
            "occupancy_colours": occupancy_colours
        }

        pickle.dump(data, dataset_file)

def main(args):
    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)
    class_to_colour = {}
    dataset_file = open("occupancy_dataset.pickle", "wb")
    metadata = dict(
        voxel_size=args.voxel_size,
        grid_width=OCCUPANCY_GRID_WIDTH,
        grid_height=OCCUPANCY_GRID_HEIGHT,
        grid_depth=OCCUPANCY_GRID_DEPTH
    )

    print(metadata)
    pickle.dump(metadata, dataset_file)

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        class_to_colour[index] = colour
    
    try:
        for scene in tqdm(nusc.scene, desc="Scene"):
            process_scene(
                dataset_file,
                scene, 
                nusc, 
                nusc_can, 
                class_to_colour, 
                args.voxel_size, 
                args.num_video_frames, 
            )
    except KeyboardInterrupt:
        return
    
    dataset_file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
