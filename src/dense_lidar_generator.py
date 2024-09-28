import os
from math import radians
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

from src.utils import (
    Transform, CarTrajectory, TimestampData, rotate_2d_vector, generate_frustum_from_camera_extrinsics, 
    create_lidar_geometries, generate_box_pointclouds, Camera
)

class DenseLidarGenerator:
    def __init__(
            self, 
            nusc, 
            nusc_can, 
            scene:dict, 
            num_future_samples:int,
            num_video_frames:int,
            colourmap:dict,
            static_object_ids:list,
            num_box_cloud_points:int,
            frustum_distance:float):
        self.nusc = nusc
        self.nusc_can = nusc_can
        self.scene = scene
        self.num_video_frames = num_video_frames
        self.num_future_samples = num_future_samples
        self.colourmap = colourmap
        self.static_object_ids = static_object_ids
        self.num_box_cloud_points = num_box_cloud_points
        self.frustum_distance = frustum_distance
        self.total_samples = scene["nbr_samples"]
        self.samples = []
        self.lidar_cache = {}
        self.sample_offset = num_video_frames - 1
        self.current_index = 0
        self.load_samples()

        try:
            pose = nusc_can.get_messages(scene["name"], 'pose')
            self.pose_dataset = TimestampData(pose, [p["utime"] for p in pose])
        except:
            #CAN data does not exist for scene
            self.pose_dataset = None

        self.length = len(self.samples) - self.num_future_samples - self.num_video_frames
    
    def load_samples(self) -> None:
        sample = self.nusc.get("sample", self.scene["first_sample_token"])
        self.samples.append(sample)

        while(sample["next"] != str()):
            sample = self.nusc.get("sample", sample["next"])
            self.samples.append(sample)

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
        car_trajectory = CarTrajectory()
        dense_lidar = o3d.geometry.PointCloud()
        
        start = self.sample_offset + index
        end = start + self.num_future_samples

        if(start < 0 or end >= len(self.samples)):
            raise Exception(f"Index {index} out of range")

        current_sample = self.samples[start]
        camera_transform = None
        camera_frustum_geometry = None
        lidar_futures = []
        lidar_batch = []
        image_paths = self._get_previous_frame_paths(start)
        
        with ProcessPoolExecutor(max_workers=16) as executor:
            for i in range(start, end):
                sample = self.samples[i]

                if(sample["token"] not in self.lidar_cache):
                    lidar_token = sample['data']['LIDAR_TOP']
                    pcd_path = self.nusc.get_sample_data_path(lidar_token)
                    pcd_labels_path = os.path.join(self.nusc.dataroot, f"lidarseg/{self.nusc.version}", lidar_token + '_lidarseg.bin')
                    future = executor.submit(create_lidar_geometries, pcd_path, pcd_labels_path, self.colourmap.copy(), self.static_object_ids.copy())
                    lidar_futures.append((sample, future))
                else:
                    lidar_futures.append((sample, self.lidar_cache[sample["token"]]))

        for sample, data in lidar_futures:
            if(isinstance(data, Future)):
                lidar_batch.append((sample, data.result()))
            else:
                lidar_batch.append((sample, data))

        for sample, (dynamic_points, dynamic_colours, static_points, static_colours) in lidar_batch:
            # Creating points using Vector3dVector takes ~400ms but cannot be cached because of the later translations
            # it requires a different translation each time
            static_lidar_geometry = o3d.geometry.PointCloud()
            static_lidar_geometry.points = o3d.utility.Vector3dVector(static_points)
            static_lidar_geometry.colors = o3d.utility.Vector3dVector(static_colours)
            dynamic_lidar_geometry = o3d.geometry.PointCloud()
            dynamic_lidar_geometry.points = o3d.utility.Vector3dVector(dynamic_points)
            dynamic_lidar_geometry.colors = o3d.utility.Vector3dVector(dynamic_colours)
            
            lidar_token = sample['data']['LIDAR_TOP']
            lidar = self.nusc.get('sample_data', lidar_token)
            cam_front = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            cam_front_extrinsics = self.nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
            lidar_extrinsics = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            lidar_origin = np.array(lidar_extrinsics['translation'])
            box_detections = self.nusc.get_boxes(lidar_token)
            ego = self.nusc.get("ego_pose", lidar_token)
            car_world_position = ego["translation"]
            car_rotation = Quaternion(ego["rotation"])

            if(self.pose_dataset is not None):
                can_pose_at_timestamp = self.pose_dataset.get_data_at_timestamp(lidar["timestamp"])
                car_trajectory.update(can_pose_at_timestamp["pos"])
            else:
                car_trajectory.update(car_world_position)

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

                camera_frustum_geometry = generate_frustum_from_camera_extrinsics(cam_front_extrinsics, car_rotation, cam_front["width"], cam_front["height"], self.frustum_distance)
                camera_frustum_geometry.translate(car_local_position)

            dense_lidar.points.extend(o3d.utility.Vector3dVector(static_lidar_geometry.points))
            dense_lidar.colors.extend(o3d.utility.Vector3dVector(static_lidar_geometry.colors))

        
        return dense_lidar, Camera(camera_transform, camera_frustum_geometry, image_paths)

    def _get_previous_frame_paths(self, current_sample_index):
        end = current_sample_index + 1
        start = current_sample_index - (self.num_video_frames - 1)
        image_paths = []

        for i in range(start, end):
            sample = self.samples[i]
            cam_front = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            image_path = os.path.join(self.nusc.dataroot, cam_front["filename"])
            image_paths.append(image_path)
            
        return image_paths