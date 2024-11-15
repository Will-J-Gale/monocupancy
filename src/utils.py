import os
from math import sin, cos, radians

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from open3d.geometry import get_rotation_matrix_from_xyz

from src.constants import UNLABELLED_DATA_CLASS, GROUND_CLEARANCE

class Transform:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class CarTrajectory:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0], dtype=np.float64)
        self.rotation = np.array([0, 0, 0, 0], dtype=np.float64)
        self.velocity = np.array([0, 0, 0], dtype=np.float64)
        self.prev_car_position = None
    
    def update(self, position):
        if(self.prev_car_position is not None):
            current_pos = np.array(position)
            prev_pos = np.array(self.prev_car_position)

            self.velocity = prev_pos - current_pos
            self.position += self.velocity

        self.prev_car_position = np.array(position)

class TimestampData:
    def __init__(self, data:list, timestamps:list):
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = np.array(timestamps)

    def get_data_at_timestamp(self, timestamp):
        min_index = np.argmin(np.abs(self.timestamps - timestamp))
        return self.data[min_index]

class PointCloudTimeseries:
    def __init__(self):
        self.combined_geometry = None
        self.points = []
        self.colours = []
        self.combined_geometry = o3d.geometry.PointCloud()
    
    def add_geometry(self, lidar_geometry):
        self.combined_geometry.points.extend(o3d.utility.Vector3dVector(lidar_geometry.points))
        self.combined_geometry.colors.extend(o3d.utility.Vector3dVector(lidar_geometry.colors))

class Frustum:
    def __init__(self, points):
        self.planes = [
            [points[0], points[1], points[2]],
            [points[0], points[2], points[3]],
            [points[0], points[3], points[4]],
            [points[0], points[4], points[1]],
            [points[4], points[3], points[2], points[1]],
        ]

        self.normals = []
        for plane in self.planes:
            normal = np.cross((plane[1] - plane[0]), (plane[2] - plane[0]))
            self.normals.append(normal)
    
    def contains_point(self, point:np.ndarray):
        for plane, normal in zip(self.planes, self.normals):
            if(np.dot(plane[0] - point, normal) < 0):
                return False
        
        return True

class Camera:
    def __init__(self, transform, frustum_geometry, image_paths):
        self.transform = transform
        self.frustum_geometry = frustum_geometry
        self.image_paths = image_paths

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
    vfov_half = vfov_radians / 2
    
    x, y = rotate_2d_vector(0, distance, hfov_half)
    z, _ = rotate_2d_vector(0, distance, vfov_half)

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

def create_frustum_geometry_at_origin(hfov_radians, vfov_radians, distance):
    hfov_half = hfov_radians / 2
    vfov_half = vfov_radians / 2
    
    x, y = rotate_2d_vector(0, distance, hfov_half)
    z, _ = rotate_2d_vector(0, distance, vfov_half)

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
    return line_set

def create_lidar_geometries(pcd_path, label_path, colourmap, static_object_ids):
    if(os.path.exists(label_path)):
        pcd_labels = np.fromfile(label_path, dtype=np.uint8)
    else:
        pcd_labels = None

    point_cloud_raw = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
    point_cloud_raw = point_cloud_raw[..., 0:3]

    static_points = []
    static_labels = []
    dynamic_points = []
    dynamic_labels = []

    for i in range(len(point_cloud_raw)):
        point = point_cloud_raw[i]
        if(pcd_labels is not None):
            label = pcd_labels[i]

            if(label in static_object_ids):
                static_points.append(point)
                static_labels.append(label)
            else:
                dynamic_points.append(point)
                dynamic_labels.append(label)

        else:
            static_points.append(point)
            static_labels.append(UNLABELLED_DATA_CLASS)

    static_colours = [colourmap[label] for label in static_labels]
    dynamic_colours = [colourmap[label] for label in dynamic_labels]

    return dynamic_points, dynamic_colours, dynamic_labels, static_points, static_colours, static_labels

def generate_box_pointclouds(box_detections, car_global_position, car_relative_position, num_box_cloud_points=1000):
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

        box_cloud = bbox_mesh.sample_points_uniformly(number_of_points=num_box_cloud_points)
        box_clouds.append(box_cloud)

    return box_clouds

def generate_frustum_from_camera_extrinsics(cam_extrinsics:dict, position:np.array, rotation:Quaternion, image_width:int, image_height:int, distance=100):
    fx = cam_extrinsics["camera_intrinsic"][0][0]
    fy = cam_extrinsics["camera_intrinsic"][1][1]

    cam_hfov = 2 * np.arctan2(image_width, 2 * fx)
    cam_vfov = 2 * np.arctan2(image_height, 2 * fy)

    frustum = create_frustum_geometry(
        position,
        Quaternion(rotation).rotation_matrix,
        cam_hfov,
        cam_vfov,
        distance
    )
    return frustum

def generate_boxes_meshes(boxes, car_global_position, car_relative_position):
    box_meshes = []
    
    for box in boxes:
        box.translate(-np.array(car_global_position))
        w, l, h = box.wlh
        x, y, z = box.center

        # #TriangleMesh - BOTTOM LEFT ANCHOR...
        bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=l, height=w, depth=h)
        bbox_mesh.compute_vertex_normals()
        bbox_mesh.paint_uniform_color([1.0, 0.1, 0.0])

        x -= l/2
        y -= w/2
        z -= h/2

        bbox_mesh.translate([x, y, z], relative=True)
        bbox_mesh.rotate(box.rotation_matrix)

        R = get_rotation_matrix_from_xyz([0, 0, radians(90)]) #Again, not sure why it needs rotating 90 degrees
        bbox_mesh.rotate(R, [0, 0, 0])
        bbox_mesh.translate(car_relative_position, relative=True)

        box_meshes.append(bbox_mesh)

    return box_meshes

class OccupancyResult:
    def __init__(
            self, 
            occupancy_grid, 
            cropped_points,
            occupancy_box_origin,
            occupancy_box_car,
            frustum_origin):
        
        self.occupancy_grid = occupancy_grid
        self.cropped_points = cropped_points
        self.occupancy_box_origin = occupancy_box_origin
        self.occupancy_box_car = occupancy_box_car
        self.frustum_origin = frustum_origin

def generate_camera_view_occupancy(
        dense_pointcloud:o3d.geometry.PointCloud, 
        car_transform:Transform,
        x_scale:float, 
        y_scale:float, 
        z_scale:float,
        camera:Camera,
        voxel_grid_template:o3d.geometry.VoxelGrid):

    half_y = y_scale / 2
    half_z = z_scale / 2

    occupancy_box_pos = car_transform.position + np.array([0, half_y, half_z - GROUND_CLEARANCE])
    occupancy_box = o3d.geometry.OrientedBoundingBox(occupancy_box_pos, np.eye(3), (x_scale, y_scale, z_scale))
    occupancy_box.rotate(car_transform.rotation.rotation_matrix, car_transform.position)
    occupancy_box.color = (1.0, 1.0, 0.0) #TODO Magic tuple

    cropped_points = dense_pointcloud.crop(occupancy_box)
    cropped_points.rotate(car_transform.rotation.inverse.rotation_matrix, [0, 0, 0])

    frustum_geometry = o3d.geometry.LineSet(camera.frustum_geometry)
    frustum_geometry.rotate(car_transform.rotation.inverse.rotation_matrix, [0, 0, 0])
    frustum = Frustum(frustum_geometry.points)

    occupancy_grid = o3d.geometry.VoxelGrid(voxel_grid_template)

    for point, colour in zip(cropped_points.points, cropped_points.colors):
        if(frustum.contains_point(point)):
            voxel_index = occupancy_grid.get_voxel(point)
            occupancy_grid.add_voxel(o3d.geometry.Voxel(voxel_index, colour))

    occupancy_box_origin = o3d.geometry.OrientedBoundingBox(occupancy_box)
    occupancy_box_origin.rotate(car_transform.rotation.inverse.rotation_matrix, [0, 0, 0])

    return OccupancyResult(
        occupancy_grid, 
        cropped_points,
        occupancy_box_origin,
        occupancy_box,
        frustum_geometry
    )

def occupancy_grid_to_list(occupancy_grid:o3d.geometry.VoxelGrid):
    #Grid is X, Z, Y???
    grid_points = []
    colours = []
    for voxel in occupancy_grid.get_voxels():
        grid_points.append(list(voxel.grid_index))
        colours.append(list(voxel.color))

    return grid_points, colours

def box_to_dict(box):
    return dict(
        center = list(box.center),
        size = list(box.wlh),
        rotation_matrix = box.rotation_matrix,
        label = box.label,
        score = box.score,
        name = box.name,
        token = box.token,
    )

def dict_to_box(box_dict):
    return Box(
        box_dict["center"],
        box_dict["size"],
        Quaternion(matrix=box_dict["rotation_matrix"]),
        box_dict["label"],
        box_dict["score"]
    )

def occupancy_indicies_to_numpy(occupancy_indexes, grid_shape):
    occupancy = np.zeros(grid_shape, dtype=np.bool8)

    for index in occupancy_indexes:
        x, y, z = index
        occupancy[x, z, y] = 1 #Why/What?!

    return occupancy