import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.dirname(__file__))) #Add parent folder to path

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from python_input import Input

from src.visualisation import Visualizer
from src.utils import Frustum, generate_camera_view_occupancy
from src.dense_lidar_generator import DenseLidarGenerator
from src.constants import (
    STATIC_OBJECT_IDS, NUM_BOX_CLOUD_POINTS, OCCUPANCY_GRID_WIDTH,
    OCCUPANCY_GRID_HEIGHT, OCCUPANCY_GRID_DEPTH, NUM_FUTURE_SAMPLES,
    FRUSTUM_DISTANCE, NUM_VIDEO_FRAMES, DEFAULT_VOXEL_SIZE
)

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes-v1.0-mini")
parser.add_argument("--scene_index", type=int, default=0)
parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
parser.add_argument("--show_image", action='store_true')
parser.add_argument("--dataset_version", default="v1.0-mini")

def main(args):
    inp = Input()

    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)
    colourmap = {}

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour
    
    lidar_generator = DenseLidarGenerator(
        nusc,
        nusc_can,
        nusc.scene[args.scene_index],
        NUM_FUTURE_SAMPLES,
        NUM_VIDEO_FRAMES,
        colourmap,
        STATIC_OBJECT_IDS,
        NUM_BOX_CLOUD_POINTS,
        FRUSTUM_DISTANCE
    )

    index = 0
    dense_lidar, camera = lidar_generator[index]
    occupancy_grid = generate_camera_view_occupancy(
        dense_lidar, 
        camera.transform, 
        OCCUPANCY_GRID_WIDTH, 
        OCCUPANCY_GRID_DEPTH, 
        OCCUPANCY_GRID_HEIGHT, 
        args.voxel_size, 
        camera
    )

    vis = Visualizer()
    vis.add(
        [dense_lidar, camera.frustum_geometry], 
        occupancy_grid, 
    )

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                index += 1
                dense_lidar, camera = lidar_generator[index]
                occupancy_grid = generate_camera_view_occupancy(
                    dense_lidar, 
                    camera.transform, 
                    OCCUPANCY_GRID_WIDTH, 
                    OCCUPANCY_GRID_DEPTH, 
                    OCCUPANCY_GRID_HEIGHT, 
                    args.voxel_size, 
                    camera
                )
                vis.reset()
                vis.add(
                    [dense_lidar, camera.frustum_geometry],
                    occupancy_grid, 
                )

            if(inp.get_key_down("a")):
                pass
            
            vis.render()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
