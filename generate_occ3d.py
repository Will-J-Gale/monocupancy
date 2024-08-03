import os
import time
from math import sin, cos, radians
from argparse import ArgumentParser

import cv2
import numpy as np
import open3d as o3d
from open3d.geometry import get_rotation_matrix_from_xyz
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pyquaternion import Quaternion
from python_input import Input

from visualisation import Visualizer

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes-v1.0-mini")
parser.add_argument("--voxel_size", type=float, default=0.30)
parser.add_argument("--show_image", action='store_true')
parser.add_argument("--dataset_version", default="v1.0-mini")

CAM_IMAGE_WIDTH = 1600
CAM_IMAGE_HEIGHT = 900
NUM_BOX_CLOUD_POINTS = 1000

class Transform:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class CarTrajectory:
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
        lines=o3d.utility.Vector2iVector(lines)
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

def create_lidar_geometries(pcd_path, label_path, colourmap, static_object_ids):
    pcd_labels = np.fromfile(label_path, dtype=np.uint8)
    point_cloud_raw = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    point_cloud_raw = point_cloud_raw[..., 0:3]

    static_points = []
    static_labels = []
    dynamic_points = []
    dynamic_labels = []

    for label, point in zip(pcd_labels, point_cloud_raw):
        if(label in static_object_ids):
            static_points.append(point)
            static_labels.append(label)
        else:
            dynamic_points.append(point)
            dynamic_labels.append(label)

    static_colours = [colourmap[label] for label in static_labels]
    dynamic_colours = [colourmap[label] for label in dynamic_labels]

    static_lidar_geometry = o3d.geometry.PointCloud()
    static_lidar_geometry.points = o3d.utility.Vector3dVector(static_points)
    static_lidar_geometry.colors = o3d.utility.Vector3dVector(static_colours)

    dynamic_lidar_geometry = o3d.geometry.PointCloud()
    dynamic_lidar_geometry.points = o3d.utility.Vector3dVector(dynamic_points)
    dynamic_lidar_geometry.colors = o3d.utility.Vector3dVector(dynamic_colours)

    return static_lidar_geometry, dynamic_lidar_geometry

def generate_box_pointclouds(box_detections, car_global_position, car_relative_position):
    box_clouds = []
    
    for box in box_detections:
        box.translate(-np.array(car_global_position))
        w, l, h = box.wlh
        x, y, z = box.center

        # #TriangleMesh - BOTTOM LEFT ANCHOR...
        bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
        bbox_mesh.compute_vertex_normals()
        bbox_mesh.paint_uniform_color([1.0, 0.0, 0.0])

        x -= l/2
        y -= w/2
        z -= h/2

        bbox_mesh.translate([x, y, z], relative=True)
        bbox_mesh.rotate(box.rotation_matrix)

        R = get_rotation_matrix_from_xyz([0, 0, radians(90)]) #Again, not sure why it needs rotating 90 degrees
        bbox_mesh.rotate(R, [0, 0, 0])
        bbox_mesh.translate(car_relative_position, relative=True)

        box_cloud = bbox_mesh.sample_points_uniformly(number_of_points=NUM_BOX_CLOUD_POINTS)
        box_clouds.append(box_cloud)

    return box_clouds

def generate_frustum_from_camera_extrinsics(cam_extrinsics, rotation):
    fx = cam_extrinsics["camera_intrinsic"][0][0]
    fy = cam_extrinsics["camera_intrinsic"][1][1]

    cam_hfov = 2 * np.arctan2(CAM_IMAGE_WIDTH, 2 * fx)
    cam_vfov = 2 * np.arctan2(CAM_IMAGE_HEIGHT, 2 * fy)

    frustum = create_frustum_geometry(
        cam_extrinsics["translation"], 
        Quaternion(rotation).rotation_matrix,
        cam_hfov,
        cam_vfov,
        100
    )
    return frustum

def generate_camera_view_occupancy(
        dense_pointcloud:o3d.geometry.PointCloud, 
        car_transform:Transform,
        x_scale:float, 
        y_scale:float, 
        z_scale:float,
        voxel_size:float):

    half_y = y_scale / 2
    half_z = z_scale / 2

    occupancy_box_pos = car_transform.position + np.array([0, half_y, half_z])
    occupancy_box = o3d.geometry.OrientedBoundingBox(occupancy_box_pos, np.eye(3), (x_scale, y_scale, z_scale))
    occupancy_box.rotate(car_transform.rotation.rotation_matrix, car_transform.position)
    occupancy_box.color = (1.0, 1.0, 0.0)

    occupancy_cloud = o3d.geometry.PointCloud()

    cropped_points = dense_pointcloud.crop(occupancy_box)
    occupancy_cloud.points.extend(o3d.utility.Vector3dVector(cropped_points.points))
    occupancy_cloud.colors.extend(o3d.utility.Vector3dVector(cropped_points.colors))
    occupancy_cloud.translate([0, 0, 0], False)
    occupancy_cloud.rotate(car_transform.rotation.inverse.rotation_matrix)
    
    return o3d.geometry.VoxelGrid.create_from_point_cloud(occupancy_cloud, voxel_size=voxel_size)



class DenseLidarGenerator:
    def __init__(
            self, 
            nusc, 
            nusc_can, 
            scene:dict, 
            num_adjacent_samples:int,
            colourmap:dict,
            static_object_ids:list):
        self.nusc = nusc
        self.nusc_can = nusc_can
        self.scene = scene
        self.num_adjacent_samples = num_adjacent_samples
        self.colourmap = colourmap
        self.static_object_ids = static_object_ids
        self.total_samples = scene["nbr_samples"]
        self.samples = []
        self.load_samples()

        pose = nusc_can.get_messages(scene["name"], 'pose')
        self.pose_dataset = TimestmapData(pose, [p["utime"] for p in pose])
    
    def load_samples(self):
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        self.samples.append(sample)

        while(sample["next"] != str()):
            sample = self.nusc.get("sample", sample["next"])
            self.samples.append(sample)

    def get(self, index):
        car_trajectory = CarTrajectory()
        dense_lidar = o3d.geometry.PointCloud()
        
        start = index - self.num_adjacent_samples
        end = index + self.num_adjacent_samples

        if(start < 0 or end >= len(self.samples)):
            raise Exception(f"Index {index} out of range")

        current_sample = self.samples[index]
        current_car_transform = None

        for i in range(start, end):
            #This part can be parallelised and takes the most time (0.17s per sample)
            sample = self.samples[i]
            lidar_token = sample['data']['LIDAR_TOP']
            lidar = self.nusc.get('sample_data', lidar_token)
            cam_front = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            cam_front_extrinsics = self.nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
            lidar_extrinsics = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            lidar_origin = np.array(lidar_extrinsics['translation'])
            box_detections = self.nusc.get_boxes(lidar_token)
            pcd_path = self.nusc.get_sample_data_path(lidar_token)
            pcd_labels_path = os.path.join(self.nusc.dataroot, f"lidarseg/{self.nusc.version}", lidar_token + '_lidarseg.bin')
            ego = self.nusc.get("ego_pose", lidar_token)
            car_world_position = ego["translation"]
            car_rotation = Quaternion(ego["rotation"])
            can_pose_at_timestamp = self.pose_dataset.get_data_at_timestamp(lidar["timestamp"])

            static_lidar_geometry, dynamic_lidar_geometry = create_lidar_geometries(
                pcd_path, 
                pcd_labels_path, 
                self.colourmap, 
                self.static_object_ids
            )

            #This part cannot be parallelised as it needs all above data but it's super fast (0.0008s per sample)
            car_trajectory.update(can_pose_at_timestamp)

            car_local_position = car_trajectory.position.copy()
            car_local_position[0], car_local_position[1] = rotate_2d_vector(car_local_position[0], car_local_position[1], radians(-90)) #No idea why it needs rotating -90 degrees, maybe because car forward is actually X not Y?
            static_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
            static_lidar_geometry.translate(lidar_origin, relative=True)
            static_lidar_geometry.translate(car_local_position, relative=True)
            
            if(sample == current_sample):
                current_car_transform = Transform(car_local_position, car_rotation)
                dynamic_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
                dynamic_lidar_geometry.translate(lidar_origin, relative=True)
                dynamic_lidar_geometry.translate(car_local_position, relative=True) 
                dense_lidar.points.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.points))
                dense_lidar.colors.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.colors))

                for box_cloud in  generate_box_pointclouds(box_detections, car_world_position, car_local_position):
                    dense_lidar.points.extend(box_cloud.points)
                    dense_lidar.colors.extend(box_cloud.colors)

            dense_lidar.points.extend(o3d.utility.Vector3dVector(static_lidar_geometry.points))
            dense_lidar.colors.extend(o3d.utility.Vector3dVector(static_lidar_geometry.colors))

        return dense_lidar, current_car_transform
    
def main(args):
    inp = Input()

    #Load nuscenes data
    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)
    colourmap = {}

    static_object_ids = [ 0, 13, 24, 25, 26, 27, 28, 29, 30 ]

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour
    
    num_adjacent_samples = 10

    lidar_generator = DenseLidarGenerator(
        nusc,
        nusc_can,
        nusc.scene[0],
        num_adjacent_samples,
        colourmap,
        static_object_ids
    )

    index = num_adjacent_samples
    dense_lidar, car_transform = lidar_generator.get(index)
    occupancy = generate_camera_view_occupancy(dense_lidar, car_transform, 35, 35, 10, args.voxel_size)
    #Create window visualizer
    vis = Visualizer()
    vis.add_lidar(dense_lidar, occupancy)

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                index += 1
                dense_lidar, car_transform = lidar_generator.get(index)
                occupancy = generate_camera_view_occupancy(dense_lidar, car_transform, 35, 35, 10, args.voxel_size)
                vis.reset()
                vis.add_lidar(dense_lidar, occupancy)

            if(inp.get_key_down("a")):
                pass
            
            vis.render()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
