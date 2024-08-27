import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from argparse import ArgumentParser

import open3d as o3d
from python_input import Input

from src.visualisation import Visualizer
from src.load_dataset import load_dataset

parser = ArgumentParser()
parser.add_argument("--occupancy_path")

OCCUPANCY_SIZE = 0.3
OCCUPANCY_GRID_WIDTH = 35
OCCUPANCY_GRID_DEPTH = 35
OCCUPANCY_GRID_HEIGHT = 10

def load_occupancy_grid(
        occupancy_data:dict, 
        vis:Visualizer,
        voxel_size:float,
        grid_width:int,
        grid_height:int,
        grid_depth:int):
    voxel_grid = o3d.geometry.VoxelGrid().create_dense([0, 0, 0], [0, 0, 0], voxel_size, grid_width, grid_depth, grid_height)
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    for point, colour in zip(occupancy_data["occupancy"], occupancy_data["occupancy_colours"]):
        voxel_grid.add_voxel(o3d.geometry.Voxel(point, colour))

    vis.add_pointcloud_geometry(voxel_grid, True)

def main(args):
    inp = Input()
    vis = Visualizer()
    index = 0

    metadata, occupancy = load_dataset(args.occupancy_path)

    load_occupancy_grid(
        occupancy[index], 
        vis,
        metadata["voxel_size"],
        metadata["grid_width"],
        metadata["grid_height"],
        metadata["grid_depth"],
    )

    while(True):
        if(not vis.poll_events()):
            break

        if(inp.any_key_pressed()):
            if(inp.get_key_down("space")):
                index = min(index + 1, len(occupancy) - 1)
            elif(inp.get_key_down("a")):
                index = max(index - 1, 0)
            
            vis.reset()
            load_occupancy_grid(
                occupancy[index], 
                vis,
                metadata["voxel_size"],
                metadata["grid_width"],
                metadata["grid_height"],
                metadata["grid_depth"],
            )
            
        vis.render()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)