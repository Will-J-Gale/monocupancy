import os
from math import sin, cos, radians

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
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
    
    def update(self, pose):
        if(self.prev_car_ego is not None):
            current_pos = np.array(pose["pos"])
            prev_pos = np.array(self.prev_car_ego["pos"])

            self.velocity = prev_pos - current_pos
            self.position += self.velocity

        self.prev_car_ego = pose

class PointCloudTimeseries:
    def __init__(self):
        self.geometries = []
    
    def add_geometry(self, lidar_geometry):
        self.geometries.append(lidar_geometry)

class TimestmapData:
    def __init__(self, data:list, timestamps:list):
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = np.array(timestamps)

    def get_data_at_timestamp(self, timestamp):
        min_index = np.argmin(np.abs(self.timestamps - timestamp))
        return self.data[min_index]

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

def create_frustum_geometry(position, rotation, hfov_radians, vfov_radians, distance):
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

def create_lidar_geometries(pcd_path, label_path, colourmap, car_ego):
    car_rotation = Quaternion(np.array(car_ego["rotation"]))
    pcd_labels = np.fromfile(label_path, dtype=np.uint8)
    point_cloud_raw = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    point_cloud_raw = point_cloud_raw[..., 0:3]

    point_cloud_geometry = o3d.geometry.PointCloud()
    point_cloud_geometry.points = o3d.utility.Vector3dVector(point_cloud_raw)
    
    point_cloud_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
    colours = [colourmap[label] for label in pcd_labels]
    point_cloud_geometry.colors = o3d.utility.Vector3dVector(colours)

    return point_cloud_geometry

def generate_boxes(boxes, car_position, car_rotation):
    box_geometries = []
    for box in boxes:
        box.translate(-np.array(car_position))
        box.rotate(Quaternion(car_rotation))
        w, l, h = box.wlh
        x, y, z = box.center

        bbox = o3d.geometry.OrientedBoundingBox((x, y, z), box.rotation_matrix, (l, w, h))
        bbox.color = (1.0, 0.1, 0.0)
        box_geometries.append(bbox)

    return box_geometries

def generate_frustum_from_camera_extrinsics(cam_extrinsics, rotation):
    # Create frustum
    fx = cam_extrinsics["camera_intrinsic"][0][0]
    fy = cam_extrinsics["camera_intrinsic"][1][1]

    cam_hfov = 2 * np.arctan2(CAM_IMAGE_WIDTH, 2 * fx)
    cam_vfov = 2 * np.arctan2(CAM_IMAGE_HEIGHT, 2 * fy)

    frustum = create_frustum_geometry(
        cam_extrinsics["translation"], 
        # Quaternion(car_rotation).rotation_matrix,
        Quaternion(rotation).rotation_matrix,
        cam_hfov,
        cam_vfov,
        100
    )
    return frustum

def draw_lidar_data(vis, nusc, sample, colourmap, car_ego:CarEgo, pose_dataset:TimestmapData, point_cloud_timeseries:PointCloudTimeseries):
    global CALCULATE_BBOX
    lidar_token = sample['data']['LIDAR_TOP']
    lidar = nusc.get('sample_data', lidar_token)
    cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_front_extrinsics = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
    lidar_extrinsics = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    boxes = nusc.get_boxes(lidar_token)
    pcd_path = nusc.get_sample_data_path(lidar_token)
    pcd_labels_path = os.path.join('/media/storage/datasets/nuscenes-v1.0-mini/lidarseg/v1.0-mini', lidar_token + '_lidarseg.bin')
    ego = nusc.get("ego_pose", lidar_token)

    pose = pose_dataset.get_data_at_timestamp(lidar["timestamp"])
    car_ego.update(pose)

    lidar_geometry = create_lidar_geometries(
        pcd_path, 
        pcd_labels_path, 
        colourmap, 
        ego
    )

    pos = car_ego.position.copy()
    pos[0], pos[1] = rotate_2d_vector(pos[0], pos[1], radians(-90)) #No idea why it needs rotating -90 degrees, maybe because car forward is actuall X not Y?
    lidar_geometry.translate(pos, relative=False)
    
    point_cloud_timeseries.add_geometry(lidar_geometry)

    for geometry in point_cloud_timeseries.geometries:
        vis.add_geometry(geometry, CALCULATE_BBOX)

    return lidar_geometry

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
    nusc_can = NuScenesCanBus(dataroot='/media/storage/datasets/nuscenes-v1.0-mini')
    colourmap = {}

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    # nusc.render_sample_data(sample["data"]["LIDAR_TOP"])
    # nusc.render_sample_data(nusc.sample[30]["data"]["LIDAR_TOP"], use_flat_vehicle_coordinates=False, underlay_map=False)
    # exit()

    """
        "ms_imu",
        "pose",
        "steeranglefeedback",
        "vehicle_monitor",
        "zoe_veh_info",
        "zoesensors"
    """
    imu = nusc_can.get_messages(scene["name"], 'ms_imu')
    pose = nusc_can.get_messages(scene["name"], 'pose')

    pose_dataset = TimestmapData(pose, [p["utime"] for p in pose])

    car_ego = CarEgo()
    point_cloud_timeseries = PointCloudTimeseries()
    draw_lidar_data(vis, nusc, sample, colourmap, car_ego, pose_dataset, point_cloud_timeseries) 
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
                previous_lidar = draw_lidar_data(vis, nusc, sample, colourmap, car_ego, pose_dataset, point_cloud_timeseries)
            
            if(inp.get_key_down("a")):
                if(sample["prev"] == str()):
                    continue

                vis.clear_geometries()
                sample = nusc.get("sample", sample["prev"]) 
                draw_lidar_data(vis, nusc, sample, colourmap, car_ego, pose_dataset, point_cloud_timeseries)
            
            vis.update_renderer()
        except KeyboardInterrupt:
            break
if __name__ == "__main__":
    main()
