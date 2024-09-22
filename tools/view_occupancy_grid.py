import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shelve
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from python_input import Input

from src.visualisation import Visualizer

parser = ArgumentParser()
parser.add_argument("occupancy_path")

OCCUPANCY_SIZE = 0.3
OCCUPANCY_GRID_WIDTH = 35
OCCUPANCY_GRID_DEPTH = 35
OCCUPANCY_GRID_HEIGHT = 10
UPDATE_CAMERA = True

def load_occupancy_grid(
        occupancy_data:dict, 
        vis:o3d.visualization.Visualizer(),
        voxel_size:float,
        grid_width:int,
        grid_height:int,
        grid_depth:int):
    global UPDATE_CAMERA
    voxel_grid = o3d.geometry.VoxelGrid().create_dense([0, 0, 0], [0, 0, 0], voxel_size, grid_width, grid_depth, grid_height)
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    for point, colour in zip(occupancy_data["occupancy"], occupancy_data["occupancy_colours"]):
        voxel_grid.add_voxel(o3d.geometry.Voxel(point, colour))

    vis.add_geometry(voxel_grid, UPDATE_CAMERA)
    UPDATE_CAMERA = False

def main(args):
    inp = Input()
    vis = o3d.visualization.Visualizer()
    vis.create_window( "Occupancy", 1920//2, 1080, 0, 0)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    index = 0

    dataset = shelve.open(args.occupancy_path, "r")
    metadata = dataset["metadata"]
    dataset_length = metadata["length"]

    load_occupancy_grid(
        dataset[str(index)], 
        vis,
        metadata["voxel_size"],
        metadata["grid_width"],
        metadata["grid_height"],
        metadata["grid_depth"],
    )

    while(True):
        if(not vis.poll_events() or inp.get_key_down("q")):
            break

        if(inp.any_key_pressed()):
            if(inp.get_key_down("space")):
                index = min(index + 1, dataset_length - 1)
            elif(inp.get_key_down("a")):
                index = max(index - 1, 0)
            
            vis.clear_geometries()
            load_occupancy_grid(
                dataset[str(index)], 
                vis,
                metadata["voxel_size"],
                metadata["grid_width"],
                metadata["grid_height"],
                metadata["grid_depth"],
            )
            
        vis.update_renderer()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)