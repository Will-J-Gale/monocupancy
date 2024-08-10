from argparse import ArgumentParser

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from python_input import Input

from src.visualisation import Visualizer
from src.utils import Frustum, generate_camera_view_occupancy
from src.dense_lidar_generator import DenseLidarGenerator

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes-v1.0-mini")
parser.add_argument("--scene_index", type=int, default=0)
parser.add_argument("--voxel_size", type=float, default=0.30)
parser.add_argument("--show_image", action='store_true')
parser.add_argument("--dataset_version", default="v1.0-mini")

STATIC_OBJECT_IDS = [ 0, 13, 24, 25, 26, 27, 28, 29, 30 ]
NUM_BOX_CLOUD_POINTS = 4000
OCCUPANCY_GRID_WIDTH = 35
OCCUPANCY_GRID_DEPTH = 35
OCCUPANCY_GRID_HEIGHT= 10
NUM_FUTURE_SAMPLES = 15
FRUSTUM_DISTANCE = 100

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
        colourmap,
        STATIC_OBJECT_IDS,
        NUM_BOX_CLOUD_POINTS,
        FRUSTUM_DISTANCE
    )

    index = NUM_FUTURE_SAMPLES
    dense_lidar, camera = lidar_generator.get(index)
    frustum = Frustum(camera.frustum_geometry.points)
    occupancy, occupancy_box = generate_camera_view_occupancy(
        dense_lidar, 
        camera.transform, 
        OCCUPANCY_GRID_WIDTH, 
        OCCUPANCY_GRID_DEPTH, 
        OCCUPANCY_GRID_HEIGHT, 
        args.voxel_size, frustum
    )
    
    vis = Visualizer()
    vis.add_lidar(dense_lidar, occupancy)
    vis.add_pointcloud_geometry([occupancy_box, camera.frustum_geometry])

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                index += 1
                dense_lidar, camera = lidar_generator.get(index)
                frustum = Frustum(camera.frustum_geometry.points)
                occupancy, occupancy_box = generate_camera_view_occupancy(
                    dense_lidar, 
                    camera.transform, 
                    OCCUPANCY_GRID_WIDTH, 
                    OCCUPANCY_GRID_DEPTH, 
                    OCCUPANCY_GRID_HEIGHT, 
                    args.voxel_size, 
                    frustum
                )
                vis.reset()
                vis.add_lidar(dense_lidar, occupancy)
                vis.add_pointcloud_geometry([occupancy_box, camera.frustum_geometry])

            if(inp.get_key_down("a")):
                pass
            
            vis.render()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
