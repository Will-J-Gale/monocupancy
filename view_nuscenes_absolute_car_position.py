import os
from math import sin, cos, radians

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from pyquaternion import Quaternion
from python_input import Input

CAM_IMAGE_WIDTH = 1600
CAM_IMAGE_HEIGHT = 900
CALCULATE_BBOX = True #Stupid hack to keep camera in same place, god o3d has a weird userspace...

class CarEgo:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0], dtype=np.float64)
        self.rotation = np.array([0, 0, 0, 0], dtype=np.float64)
        self.velocity = np.array([0, 0, 0], dtype=np.float64)
        self.prev_car_ego = None
    
    def update(self, car_ego):
        self.rotation = np.array(car_ego["rotation"])

        if(self.prev_car_ego is not None):
            car_position = np.array(car_ego["translation"])
            prev_car_position = np.array(self.prev_car_ego["translation"])

            car_quat = Quaternion(self.rotation)

            #Need to some how get car velociy vector based on the cars rotation
            # self.velocity = prev_car_position - car_position
            self.velocity[0], self.velocity[1] = rotate_2d_vector(self.velocity[0], self.velocity[1], car_quat.yaw_pitch_roll[0])

            self.position += -self.velocity
        else:
            self.position = np.array(car_ego["translation"])

        self.prev_car_ego = car_ego
    
def rotate_2d_vector(x, y, angle):
    new_x = x * cos(angle) - y * sin(angle)
    new_y = x * sin(angle) + y * cos(angle)
    return new_x, new_y

def create_line(start, end, colour):
    points = [start, end]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    line_set.colors = o3d.utility.Vector3dVector([colour])

    return line_set

def create_frustum(position, rotation, hfov_radians, vfov_radians, distance):
    hfov_half = hfov_radians / 2
    vfov_hald = vfov_radians / 2
    
    x, y = rotate_2d_vector(0, distance, hfov_half)
    z, _ = rotate_2d_vector(0, distance, vfov_hald)

    points = [
        [0, 0, 0],
        [-x, y, -z],
        [x, y, -z],
        [x, y, z],
        [-x, y, z],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]
    colors = [[1.0, 0.0, 0.0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.rotate(rotation, [0, 0, 0])
    line_set.translate(position)
    return line_set

def create_lidar_geometries(vis, pcd_path, label_path, colourmap, car_ego, boxes, lidar, cam_extrinsics, car_ego2:CarEgo):
    car_position = np.array(car_ego["translation"])
    car_rotation = np.array(car_ego["rotation"])
    lidar_position = np.array(lidar["translation"])
    lidar_rotation = np.array(lidar["rotation"])
    pcd_labels = np.fromfile(label_path, dtype=np.uint8)
    pcd_bin = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    pcd_bin = pcd_bin[..., 0:3]


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_bin)
    car_quat = Quaternion(car_rotation)

    # Absolute position
    pcd.translate(car_position, relative=False)
    pcd.rotate(car_quat.rotation_matrix, car_position)

    colours = [colourmap[label] for label in pcd_labels]
    pcd.colors = o3d.utility.Vector3dVector(colours)

    vis.add_geometry(pcd, CALCULATE_BBOX)

    # line = create_line(car_ego2.position, car_ego2.position + car_ego2.velocity, [1.0, 0.0, 0.0])
    zero = np.array([0, 0, 0], dtype=np.float64)
    line = create_line(zero, car_ego2.velocity, [1.0, 0.0, 0.0])
    vis.add_geometry(line, False)

    # for box in boxes:
    #     box.translate(-np.array(car_position))
    #     box.rotate(Quaternion(car_rotation))
    #     w, l, h = box.wlh
    #     x, y, z = box.center

    #     bbox = o3d.geometry.OrientedBoundingBox((x, y, z), box.rotation_matrix, (l, w, h))
    #     bbox.color = (0.9, 0.1, 0.1)
    #     vis.add_geometry(bbox, False)

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=car_position)
    vis.add_geometry(mesh_frame, False)

    fx = cam_extrinsics["camera_intrinsic"][0][0]
    fy = cam_extrinsics["camera_intrinsic"][1][1]

    cam_hfov = 2 * np.arctan2(CAM_IMAGE_WIDTH, 2 * fx)
    cam_vfov = 2 * np.arctan2(CAM_IMAGE_HEIGHT, 2 * fy)

    # frustum = create_frustum(
    #     cam_extrinsics["translation"], 
    #     # Quaternion(car_rotation).rotation_matrix,
    #     Quaternion(lidar_rotation).rotation_matrix,
    #     cam_hfov,
    #     cam_vfov,
    #     100
    # )
    # vis.add_geometry(frustum, False)

def draw_lidar_data(vis, nusc, sample, colourmap, car_ego:CarEgo):
    lidar_token = sample['data']['LIDAR_TOP']
    lidar = nusc.get('sample_data', lidar_token)
    cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_front_extrinsics = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
    lidar_extrinsics = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    boxes = nusc.get_boxes(lidar_token)
    pcd_path = nusc.get_sample_data_path(lidar_token)
    pcd_labels_path = os.path.join('/media/storage/datasets/nuscenes-v1.0-mini/lidarseg/v1.0-mini', lidar_token + '_lidarseg.bin')
    ego = nusc.get("ego_pose", lidar_token)

    # prev_ego = None
    # if(not sample["prev"] == str()):
    #     prev_sample = nusc.get("sample", sample["prev"])
    #     prev_lidar_token = prev_sample['data']['LIDAR_TOP']
    #     prev_ego = nusc.get("ego_pose", prev_lidar_token)
    
    car_ego.update(ego)

    create_lidar_geometries(
        vis,
        pcd_path, 
        pcd_labels_path, 
        colourmap, 
        ego, 
        boxes, 
        lidar_extrinsics,
        cam_front_extrinsics,
        car_ego
    )

def main():
    global CALCULATE_BBOX

    #Create window visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()

    opt.background_color = np.asarray([0, 0, 0])
    opt.show_coordinate_frame = True

    #Load nuscenes data
    nusc = NuScenes(version='v1.0-mini', dataroot='/media/storage/datasets/nuscenes-v1.0-mini', verbose=False)
    colourmap = {}

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    # nusc.render_sample_data(sample["data"]["LIDAR_TOP"])
    # nusc.render_sample_data(nusc.sample[30]["data"]["LIDAR_TOP"], use_flat_vehicle_coordinates=False, underlay_map=False)
    # exit()

    car_ego = CarEgo()
    draw_lidar_data(vis, nusc, sample, colourmap, car_ego) 
    CALCULATE_BBOX = False

    inp = Input()

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                if(sample["next"] == str()):
                    continue

                vis.clear_geometries()
                sample = nusc.get("sample", sample["next"]) 
                draw_lidar_data(vis, nusc, sample, colourmap, car_ego)
            
            if(inp.get_key_down("a")):
                if(sample["prev"] == str()):
                    continue

                vis.clear_geometries()
                sample = nusc.get("sample", sample["prev"]) 
                draw_lidar_data(vis, nusc, sample, colourmap, car_ego)
            
            vis.update_renderer()
        except KeyboardInterrupt:
            break
if __name__ == "__main__":
    main()