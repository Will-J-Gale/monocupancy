import shelve
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from python_input import Input
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from monocupancy import Monocupancy

MODEL_INPUT_SIZE = (512, 256)
MODEL_INPUT_CHANNELS = 3
CAMERA_X = 18
VOXEL_COLOUR = (1.0, 1.0, 1.0)

parser = ArgumentParser()
parser.add_argument("dataset", default="./occupancy.dataset")
parser.add_argument("--weights", required=True)
parser.add_argument("--frame", default=0, type=int)
parser.add_argument("--image_scale", default=1.0, type=float)

DEFAULT_MATERIAL = rendering.MaterialRecord()
DEFAULT_MATERIAL.base_color = [1.0, 1.0, 1.0, 1]
DEFAULT_MATERIAL.shader = "defaultLit"

def generate_voxel_grid(voxel_size, grid_width, grid_height, grid_depth):
    voxel_grid = o3d.geometry.VoxelGrid().create_dense(
        [0, 0, 0], [0, 0, 0], 
        voxel_size, 
        grid_width, 
        grid_depth,
        grid_height
    )
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    return voxel_grid

def generate_model_input(data):
    images = []
    for path in data["image_paths"][-1:]:
        image = cv2.imread(path)
        image = cv2.resize(image, MODEL_INPUT_SIZE)
        image = image / 255
        image = np.moveaxis(image, 2, 0)
        images.append(image)

    images = np.vstack(images)
    occupancy_data = np.load(data["occupancy_path"])
    occupancy = occupancy_data["occupancy_grid"]

    train_X = np.array([images], dtype=np.float32)
    train_X = torch.from_numpy(train_X).cuda()
    train_Y = np.array([occupancy], dtype=np.float32)
    train_Y = torch.from_numpy(train_Y).cuda()

    return train_X, train_Y, cv2.imread(data["image_paths"][-1])

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

    scene.show_ground_plane(True, o3d.visualization.rendering.Scene.GroundPlane.XY)

    return scene, window

def tensor_to_voxel_grid(tensor, voxel_grid):
    [voxel_grid.remove_voxel(voxel.grid_index) for voxel in voxel_grid.get_voxels()]

    for index in torch.nonzero(tensor >= 0.95).cpu().numpy():
        x, z, y = index
        voxel_grid.add_voxel(o3d.geometry.Voxel([x,y,z], VOXEL_COLOUR))

def main(args):
    gui.Application.instance.initialize()

    inp = Input()
    target_vis, target_window = create_window("Target", width=1920//2, height=1080, x=0, y=0)
    prediction_vis, prediction_window = create_window("Prediction", width=1920//2, height=1080, x=1920//2, y=0)

    dataset = shelve.open(args.dataset, "r")
    metadata = dataset["metadata"]

    model = Monocupancy(MODEL_INPUT_CHANNELS).cuda()
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    model.eval()

    index = args.frame
    data = dataset[str(index)]
    train_X, train_Y, image = generate_model_input(data)
    prediction = model(train_X)
    
    target_voxel_grid = o3d.geometry.VoxelGrid().create_dense(
        [0, 0, 0], 
        [0, 0, 0], 
        metadata["voxel_size"], 
        metadata["grid_width"], 
        metadata["grid_depth"], 
        metadata["grid_height"]
    )
    [target_voxel_grid.remove_voxel(voxel.grid_index) for voxel in target_voxel_grid.get_voxels()]
    prediction_voxel_grid = o3d.geometry.VoxelGrid(target_voxel_grid)

    tensor_to_voxel_grid(train_Y[0], target_voxel_grid)
    tensor_to_voxel_grid(prediction[0], prediction_voxel_grid)

    target_vis.add_geometry("target", target_voxel_grid, DEFAULT_MATERIAL)
    prediction_vis.add_geometry("prediction", prediction_voxel_grid, DEFAULT_MATERIAL)

    while(True):
        if(inp.get_key_down("q")):
            break

        if(inp.any_key_pressed()):
            if(inp.get_key_down("a")):
                index = max(index - 1, 0)
            elif(inp.get_key_down("d")):
                index = min(index + 1, metadata["length"])
            
            data = dataset[str(index)]
            train_X, train_Y, image = generate_model_input(data)
            prediction = model(train_X)
            tensor_to_voxel_grid(train_Y[0], target_voxel_grid)
            tensor_to_voxel_grid(prediction[0], prediction_voxel_grid)

            target_vis.clear_geometry()
            prediction_vis.clear_geometry()
            target_vis.add_geometry("target", target_voxel_grid, DEFAULT_MATERIAL)
            prediction_vis.add_geometry("prediction", prediction_voxel_grid, DEFAULT_MATERIAL)
        
        target_window.post_redraw()
        prediction_window.post_redraw()
        gui.Application.instance.run_one_tick()

        cv2.imshow("image", cv2.resize(image, (0,0), fx=args.image_scale, fy=args.image_scale))
        cv2.waitKey(1)

    dataset.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)