import os
from math import radians
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from python_input import Input
from pyquaternion import Quaternion

from visualisation import Visualizer
from utils import (
    Transform, CarTrajectory, TimestmapData, Frustum, rotate_2d_vector, generate_frustum_from_camera_extrinsics, 
    create_lidar_geometries, generate_box_pointclouds
)

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes-v1.0-mini")
parser.add_argument("--scene_index", type=int, default=0)
parser.add_argument("--voxel_size", type=float, default=0.30)
parser.add_argument("--show_image", action='store_true')
parser.add_argument("--dataset_version", default="v1.0-mini")

CAM_IMAGE_WIDTH = 1600
CAM_IMAGE_HEIGHT = 900
NUM_BOX_CLOUD_POINTS = 2000

def generate_camera_view_occupancy(
        dense_pointcloud:o3d.geometry.PointCloud, 
        car_transform:Transform,
        x_scale:float, 
        y_scale:float, 
        z_scale:float,
        voxel_size:float,
        frustum:Frustum):

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

    visible_cloud = o3d.geometry.PointCloud()

    for point, color in zip(occupancy_cloud.points, occupancy_cloud.colors):
        if(frustum.contains_point(point)):
            visible_cloud.points.append(point)
            visible_cloud.colors.append(color)
    
    visible_cloud.translate([0, 0, 0], False)
    visible_cloud.rotate(car_transform.rotation.inverse.rotation_matrix)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(visible_cloud, voxel_size=voxel_size), occupancy_box

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
        camera_transform = None
        frustum = None

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
                camera_pos = car_local_position + cam_front_extrinsics["translation"]
                camera_pos[2] = car_local_position[2]
                camera_transform = Transform(camera_pos, car_rotation)
                dynamic_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
                dynamic_lidar_geometry.translate(lidar_origin, relative=True)
                dynamic_lidar_geometry.translate(car_local_position, relative=True) 
                dense_lidar.points.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.points))
                dense_lidar.colors.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.colors))

                for box_cloud in  generate_box_pointclouds(box_detections, car_world_position, car_local_position):
                    dense_lidar.points.extend(box_cloud.points)
                    dense_lidar.colors.extend(box_cloud.colors)

                frustum = generate_frustum_from_camera_extrinsics(cam_front_extrinsics, car_rotation, CAM_IMAGE_WIDTH, CAM_IMAGE_HEIGHT)
                frustum.translate(car_local_position)

            dense_lidar.points.extend(o3d.utility.Vector3dVector(static_lidar_geometry.points))
            dense_lidar.colors.extend(o3d.utility.Vector3dVector(static_lidar_geometry.colors))

        return dense_lidar, camera_transform, frustum
    
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
        nusc.scene[args.scene_index],
        num_adjacent_samples,
        colourmap,
        static_object_ids
    )

    index = num_adjacent_samples
    dense_lidar, car_transform, frustum_geometry = lidar_generator.get(index)
    frustum = Frustum(frustum_geometry.points)
    occupancy, occupancy_box = generate_camera_view_occupancy(dense_lidar, car_transform, 35, 35, 10, args.voxel_size, frustum)

    #Create window visualizer
    vis = Visualizer()
    vis.add_lidar(dense_lidar, occupancy)
    vis.add_pointcloud_geometry(occupancy_box)
    vis.add_pointcloud_geometry(frustum_geometry)

    while(True):
        try:
            if(not vis.poll_events() or inp.get_key_down("q")):
                break

            if(inp.get_key_down("space")):
                index += 1
                dense_lidar, car_transform, frustum_geometry = lidar_generator.get(index)
                occupancy, occupancy_box = generate_camera_view_occupancy(dense_lidar, car_transform, 35, 35, 10, args.voxel_size, frustum)
                frustum = Frustum(frustum_geometry.points)
                vis.reset()
                vis.add_lidar(dense_lidar, occupancy)
                vis.add_pointcloud_geometry(occupancy_box)
                vis.add_pointcloud_geometry(frustum_geometry)

            if(inp.get_key_down("a")):
                pass
            
            vis.render()

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
