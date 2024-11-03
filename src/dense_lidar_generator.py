import os
from math import radians
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

from src.utils import (
    Transform, CarTrajectory, rotate_2d_vector, generate_frustum_from_camera_extrinsics, 
    create_lidar_geometries, generate_box_pointclouds, Camera, dict_to_box
)

class DenseLidarGenerator:
    def __init__(
            self, 
            scene_samples:List[dict], 
            num_future_samples:int,
            num_video_frames:int,
            colourmap:dict,
            static_object_ids:list,
            num_box_cloud_points:int,
            frustum_distance:float):
        self.samples = scene_samples
        self.num_video_frames = num_video_frames
        self.num_future_samples = num_future_samples
        self.colourmap = colourmap
        self.static_object_ids = static_object_ids
        self.num_box_cloud_points = num_box_cloud_points
        self.frustum_distance = frustum_distance
        self.lidar_cache = {}
        self.sample_offset = num_video_frames - 1
        self.current_index = 0

        self.length = len(scene_samples) - self.num_future_samples - self.num_video_frames
    
    def __len__(self):
        return self.length

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if(self.current_index >= self.length):
            raise StopIteration

        return_value = self[self.current_index]
        self.current_index += 1
        return return_value

    def __getitem__(self, index:int) -> Tuple[o3d.geometry.PointCloud, Camera]:
        start = self.sample_offset + index
        end = start + self.num_future_samples

        if(start < 0 or end >= len(self.samples)):
            raise Exception(f"Index {index} out of range")

        car_trajectory = CarTrajectory()
        dense_lidar = o3d.geometry.PointCloud()
        current_sample = self.samples[start]
        camera_transform = None
        camera_frustum_geometry = None
        lidar_futures = []
        lidar_batch = []
        labels = []
        image_paths = self._get_previous_frame_paths(start)
        
        with ProcessPoolExecutor(max_workers=16) as executor:
            for i in range(start, end):
                #@TODO Lidar cache isn't actually used...
                sample = self.samples[i]
                pcd_path = sample["lidar_pcd_path"]
                pcd_labels_path = sample["lidar_pcd_labels_path"]

                if(pcd_path not in self.lidar_cache):
                    future = executor.submit(create_lidar_geometries, pcd_path, pcd_labels_path, self.colourmap.copy(), self.static_object_ids.copy())
                    lidar_futures.append((sample, future, pcd_path))
                else:
                    lidar_futures.append((sample, self.lidar_cache[pcd_path], pcd_path))

        for sample, data, pcd_path in lidar_futures:
            if(isinstance(data, Future)):
                result = data.result()
                lidar_batch.append((sample, result))
                self.lidar_cache[pcd_path] = result
            else:
                lidar_batch.append((sample, data))

        for sample, (dynamic_points, dynamic_colours, dynamic_labels, static_points, static_colours, static_labels) in lidar_batch:
            # Creating points using Vector3dVector takes ~400ms but cannot be cached because of the later translations
            # it requires a different translation each time
            static_lidar_geometry = o3d.geometry.PointCloud()
            static_lidar_geometry.points = o3d.utility.Vector3dVector(static_points)
            static_lidar_geometry.colors = o3d.utility.Vector3dVector(static_colours)
            dynamic_lidar_geometry = o3d.geometry.PointCloud()
            dynamic_lidar_geometry.points = o3d.utility.Vector3dVector(dynamic_points)
            dynamic_lidar_geometry.colors = o3d.utility.Vector3dVector(dynamic_colours)
            
            cam_front_extrinsics = sample["cam_front_extrinsics"]
            lidar_origin = np.array(sample["lidar_origin"])
            box_detections = [dict_to_box(box_dict) for box_dict in sample["box_detections"]]
            car_world_position = sample["car_world_position"]
            car_rotation = Quaternion(sample["car_rotation"])
            can_pose_at_timestamp = sample["can_pose_at_timestamp"]
            image_width = sample["image_width"]
            image_height = sample["image_height"]

            if(can_pose_at_timestamp is not None):
                car_trajectory.update(can_pose_at_timestamp["pos"])
            else:
                car_trajectory.update(car_world_position)

            car_local_position = car_trajectory.position.copy()
            car_local_position[0], car_local_position[1] = rotate_2d_vector(car_local_position[0], car_local_position[1], radians(-90)) #No idea why it needs rotating -90 degrees, maybe because car forward is actually X not Y?
            static_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
            static_lidar_geometry.translate(lidar_origin, relative=True)
            static_lidar_geometry.translate(car_local_position, relative=True)
            
            if(sample == current_sample):
                #Current sample always first sample! @TODO Move this out of the loop
                camera_pos = car_local_position
                camera_pos[2] = car_local_position[2]
                camera_transform = Transform(camera_pos, car_rotation)
                dynamic_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
                dynamic_lidar_geometry.translate(lidar_origin, relative=True)
                dynamic_lidar_geometry.translate(car_local_position, relative=True) 
                dense_lidar.points.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.points))
                dense_lidar.colors.extend(o3d.utility.Vector3dVector(dynamic_lidar_geometry.colors))
                labels.extend(dynamic_labels)

                for box_cloud in  generate_box_pointclouds(box_detections, car_world_position, car_local_position, self.num_box_cloud_points):
                    dense_lidar.points.extend(box_cloud.points)
                    dense_lidar.colors.extend(box_cloud.colors)

                camera_frustum_geometry = generate_frustum_from_camera_extrinsics(cam_front_extrinsics, car_local_position, car_rotation, image_width, image_height, self.frustum_distance)
                camera_frustum_geometry.translate(car_local_position)

            dense_lidar.points.extend(o3d.utility.Vector3dVector(static_lidar_geometry.points))
            dense_lidar.colors.extend(o3d.utility.Vector3dVector(static_lidar_geometry.colors))
            labels.extend(static_labels)
        
        return dense_lidar, labels, Camera(camera_transform, camera_frustum_geometry, image_paths)

    def _get_previous_frame_paths(self, current_sample_index):
        end = current_sample_index + 1
        start = current_sample_index - (self.num_video_frames - 1)
        image_paths = []

        for i in range(start, end):
            sample = self.samples[i]
            image_paths.append(sample["image_path"])
            
        return image_paths