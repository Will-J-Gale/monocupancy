import os
from math import radians
from typing import Tuple

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

from src.utils import (
    Transform, CarTrajectory, TimestmapData, rotate_2d_vector, generate_frustum_from_camera_extrinsics, 
    create_lidar_geometries, generate_box_pointclouds, Camera
)

class DenseLidarGenerator:
    def __init__(
            self, 
            nusc, 
            nusc_can, 
            scene:dict, 
            num_adjacent_samples:int,
            colourmap:dict,
            static_object_ids:list,
            num_box_cloud_points:int):
        self.nusc = nusc
        self.nusc_can = nusc_can
        self.scene = scene
        self.num_adjacent_samples = num_adjacent_samples
        self.colourmap = colourmap
        self.static_object_ids = static_object_ids
        self.num_box_cloud_points = num_box_cloud_points
        self.total_samples = scene["nbr_samples"]
        self.samples = []
        self.load_samples()

        pose = nusc_can.get_messages(scene["name"], 'pose')
        self.pose_dataset = TimestmapData(pose, [p["utime"] for p in pose])
    
    def load_samples(self) -> None:
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        self.samples.append(sample)

        while(sample["next"] != str()):
            sample = self.nusc.get("sample", sample["next"])
            self.samples.append(sample)

    def get(self, index:int) -> Tuple[o3d.geometry.PointCloud, Camera]:
        car_trajectory = CarTrajectory()
        dense_lidar = o3d.geometry.PointCloud()
        
        start = index - self.num_adjacent_samples
        end = index + self.num_adjacent_samples

        if(start < 0 or end >= len(self.samples)):
            raise Exception(f"Index {index} out of range")

        current_sample = self.samples[index]
        camera_transform = None
        camera_frustum = None

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

                for box_cloud in  generate_box_pointclouds(box_detections, car_world_position, car_local_position, self.num_box_cloud_points):
                    dense_lidar.points.extend(box_cloud.points)
                    dense_lidar.colors.extend(box_cloud.colors)

                camera_frustum = generate_frustum_from_camera_extrinsics(cam_front_extrinsics, car_rotation, cam_front["width"], cam_front["height"])
                camera_frustum.translate(car_local_position)

            dense_lidar.points.extend(o3d.utility.Vector3dVector(static_lidar_geometry.points))
            dense_lidar.colors.extend(o3d.utility.Vector3dVector(static_lidar_geometry.colors))

        return dense_lidar, Camera(camera_transform, camera_frustum)