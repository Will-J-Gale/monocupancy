import os
import sys
from argparse import ArgumentParser
import shelve
sys.path.append(os.path.dirname(os.path.dirname(__file__))) #Add parent folder to path

import cv2
import numpy as np
import open3d as o3d
from python_input import Input

from src.visualisation import Visualizer
from src.utils import generate_camera_view_occupancy, occupancy_grid_to_list, occupancy_indicies_to_numpy
from src.dense_lidar_generator import DenseLidarGenerator
from src.constants import (
    STATIC_OBJECT_IDS, NUM_BOX_CLOUD_POINTS, OCCUPANCY_GRID_WIDTH,
    OCCUPANCY_GRID_HEIGHT, OCCUPANCY_GRID_DEPTH, NUM_FUTURE_SAMPLES,
    FRUSTUM_DISTANCE, NUM_VIDEO_FRAMES, DEFAULT_VOXEL_SIZE, GROUND_CLEARANCE
)

def numpy_to_voxel_grid(tensor, voxel_grid):
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    for index in zip(*np.nonzero(tensor >= 0.95)):
        x, z, y = index
        voxel_grid.add_voxel(o3d.geometry.Voxel([x,y,z], (255, 255, 255)))
    
parser = ArgumentParser()
parser.add_argument("--dataset", default="./nuscenes_simplified.dataset")
parser.add_argument("--scene", type=int, default=0)
parser.add_argument("--frame", type=int, default=0)
parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
parser.add_argument("--show_image", action='store_true')

def occupancy_bounding_box(occupancy, colour=(1.0, 1.0, 0.0)):
    min_bounds = occupancy.get_min_bound()
    max_bounds = occupancy.get_max_bound()

    center = max_bounds - min_bounds
    width = abs(max_bounds[0] - min_bounds[0])
    height = abs(max_bounds[1] - min_bounds[1])
    depth = abs(max_bounds[2] - min_bounds[2])

    occupancy_box = o3d.geometry.OrientedBoundingBox(occupancy.get_center(), np.eye(3), (width, height, depth))
    occupancy_box.color = colour

    return occupancy_box

def main(args):
    inp = Input()

    nuscenes_data = shelve.open(args.dataset, flag="r")
    colourmap = nuscenes_data["class_to_colour"]

    occupancy_grid_template = o3d.geometry.VoxelGrid.create_dense([-OCCUPANCY_GRID_WIDTH/2, 0, -GROUND_CLEARANCE], [255, 255, 255], args.voxel_size, OCCUPANCY_GRID_WIDTH, OCCUPANCY_GRID_DEPTH, OCCUPANCY_GRID_HEIGHT)
    [occupancy_grid_template.remove_voxel(voxel.grid_index) for voxel in occupancy_grid_template.get_voxels()]

    lidar_generator = DenseLidarGenerator(
        nuscenes_data[str(args.scene)],
        NUM_FUTURE_SAMPLES,
        NUM_VIDEO_FRAMES,
        colourmap,
        STATIC_OBJECT_IDS,
        NUM_BOX_CLOUD_POINTS,
        FRUSTUM_DISTANCE
    )

    index = args.frame
    dense_lidar, _, camera = lidar_generator[index]
    image = cv2.imread(camera.image_paths[-1])
    occupancy_result = generate_camera_view_occupancy(
        dense_lidar, 
        camera.transform, 
        OCCUPANCY_GRID_WIDTH, 
        OCCUPANCY_GRID_DEPTH, 
        OCCUPANCY_GRID_HEIGHT, 
        camera,
        occupancy_grid_template
    )

    vis = Visualizer()
    vis.add(
        [dense_lidar, occupancy_result.occupancy_box_car, camera.frustum_geometry],
        [occupancy_result.occupancy_box_origin, occupancy_result.occupancy_grid, occupancy_result.frustum_origin]
    )

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                index += 1
                dense_lidar, _, camera = lidar_generator[index]
                occupancy_result = generate_camera_view_occupancy(
                    dense_lidar, 
                    camera.transform, 
                    OCCUPANCY_GRID_WIDTH, 
                    OCCUPANCY_GRID_DEPTH, 
                    OCCUPANCY_GRID_HEIGHT, 
                    camera,
                    occupancy_grid_template
                )

                vis.reset()
                vis.add(
                    [dense_lidar, occupancy_result.occupancy_box_car, camera.frustum_geometry],
                    [occupancy_result.occupancy_box_origin, occupancy_result.occupancy_grid, occupancy_result.frustum_origin]
                )

            if(inp.get_key_down("a")):
                pass
            
            vis.render()

            if(args.show_image):
                cv2.imshow("image", image)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
