import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shelve
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from python_input import Input

parser = ArgumentParser()
parser.add_argument("occupancy_path")

OCCUPANCY_SIZE = 0.3
OCCUPANCY_GRID_WIDTH = 35
OCCUPANCY_GRID_DEPTH = 35
OCCUPANCY_GRID_HEIGHT = 10
CAMERA_X = 18
UPDATE_CAMERA = True

DEFAULT_MATERIAL = rendering.MaterialRecord()
DEFAULT_MATERIAL.base_color = [1.0, 1.0, 1.0, 1]
DEFAULT_MATERIAL.shader = "defaultLit"

def create_window(name, width=1920//2, height=1080, x=0, y=0):
    window = gui.Application.instance.create_window(name, width, height, x, y)
    scene_widget = gui.SceneWidget()
    scene = rendering.Open3DScene(window.renderer)
    scene.set_background([0, 0, 0, 255])
    scene_widget.scene = scene
    window.add_child(scene_widget)
    bbox = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [20, 20, 20])
    scene_widget.setup_camera(60, bbox, [0, 0, 0])
    
    material = rendering.MaterialRecord()
    material.base_color = [1.0, 1.0, 1.0, 1]
    material.shader = "defaultLit"

    scene_widget.scene.scene.enable_sun_light(True)
    scene.camera.look_at([CAMERA_X, 25, 0], [CAMERA_X, -12, 15], [0, 1, 0])

    return scene

def load_occupancy_grid(occupancy_data:dict, voxel_grid:o3d.geometry.VoxelGrid):
    global UPDATE_CAMERA
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    occupancy_grid_file = np.load(occupancy_data["occupancy_path"])
    occopancy_indicies = occupancy_grid_file["occupancy_indicies"]
    occupancy_colours = occupancy_grid_file["occupancy_colours"]

    for point, colour in zip(occopancy_indicies, occupancy_colours):
        colour = np.array(colour) / 255
        voxel_grid.add_voxel(o3d.geometry.Voxel(point, colour))

    UPDATE_CAMERA = False

def main(args):
    inp = Input()
    gui.Application.instance.initialize()
    vis = create_window("Target", width=1920//2, height=1080, x=0, y=0)
    index = 0

    dataset = shelve.open(args.occupancy_path, "r")
    metadata = dataset["metadata"]
    dataset_length = metadata["length"]

    voxel_grid = o3d.geometry.VoxelGrid().create_dense(
        [0, 0, 0], 
        [0, 0, 0], 
        metadata["voxel_size"], 
        metadata["grid_width"], 
        metadata["grid_depth"], 
        metadata["grid_height"]
    )
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    load_occupancy_grid(
        dataset[str(index)], 
        voxel_grid 
    )

    vis.add_geometry("voxels", voxel_grid, DEFAULT_MATERIAL)

    while(True):
        if(inp.get_key_down("q")):
            break

        if(inp.any_key_pressed()):
            if(inp.get_key_down("space")):
                index = min(index + 1, dataset_length - 1)
            elif(inp.get_key_down("a")):
                index = max(index - 1, 0)
            
            load_occupancy_grid(
                dataset[str(index)], 
                voxel_grid,
            )

            vis.clear_geometry()
            vis.add_geometry("voxels", voxel_grid, DEFAULT_MATERIAL)
            
        gui.Application.instance.run_one_tick()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)