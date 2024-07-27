import os
from math import sin, cos, radians
from argparse import ArgumentParser

import cv2
import open3d as o3d
import numpy as np
from open3d.geometry import get_rotation_matrix_from_xyz
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from pyquaternion import Quaternion
from python_input import Input

parser = ArgumentParser()
parser.add_argument("--dataset_root", default="/media/storage/datasets/nuscenes-v1.0-mini")
parser.add_argument("--voxel_size", type=float, default=0.5)
parser.add_argument("--scene_index", type=int, default=0)
parser.add_argument("--show_image", action='store_true')

CAM_IMAGE_WIDTH = 1600
CAM_IMAGE_HEIGHT = 900
CALCULATE_BBOX = True #Stupid hack to keep camera in same place, god o3d has a weird userspace...

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

class PointCloudTimeseries:
    def __init__(self):
        self.combined_geometry = None
        self.points = []
        self.colours = []
        self.combined_geometry = o3d.geometry.PointCloud()
    
    def add_geometry(self, lidar_geometry):
        self.combined_geometry.points.extend(o3d.utility.Vector3dVector(lidar_geometry.points))
        self.combined_geometry.colors.extend(o3d.utility.Vector3dVector(lidar_geometry.colors))

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

def generate_frustum_from_camera_extrinsics(cam_extrinsics, rotation):
    # Create frustum
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

def generate_camera_view_voxel_grid(
        point_cloud_list:list, 
        car_position:np.array, 
        car_rotation:np.array, 
        box_meshes:list,
        x_scale:float, 
        y_scale:float, 
        z_scale:float):
    half_y = y_scale / 2
    half_z = z_scale / 2

    occupancy_box_pos = car_position + np.array([0, half_y, half_z])
    occupancy_box = o3d.geometry.OrientedBoundingBox(occupancy_box_pos, np.eye(3), (x_scale, y_scale, z_scale))
    occupancy_box.rotate(car_rotation, car_position)
    occupancy_box.color = (1.0, 1.0, 0.0)

    occupancy_cloud = o3d.geometry.PointCloud()

    for cloud in point_cloud_list:
        cropped_points = cloud.crop(occupancy_box)
        occupancy_cloud.points.extend(o3d.utility.Vector3dVector(cropped_points.points))
        occupancy_cloud.colors.extend(o3d.utility.Vector3dVector(cropped_points.colors))
        # occupancy_cloud.colors.extend(np.ones_like(cropped_points.colors) * 255)
    
    for box in box_meshes:
        box_cloud = box.sample_points_uniformly(number_of_points=1000)
        box_cloud = box_cloud.crop(occupancy_box)
        occupancy_cloud.points.extend(o3d.utility.Vector3dVector(box_cloud.points))
        occupancy_cloud.colors.extend(o3d.utility.Vector3dVector(box_cloud.colors))
        # occupancy_cloud.colors.extend(np.ones_like(cropped_points.colors) * 100)

    return occupancy_cloud, occupancy_box

def draw_lidar_data(
        cloud_window:o3d.visualization.Visualizer, 
        occupancy_window:o3d.visualization.Visualizer, 
        nusc, 
        sample:dict, 
        colourmap:dict, 
        static_object_ids:list, 
        car_trajectory:CarTrajectory, 
        pose_dataset:TimestmapData, 
        point_cloud_timeseries:PointCloudTimeseries,
        voxel_size:float=0.5):
    global CALCULATE_BBOX
    lidar_token = sample['data']['LIDAR_TOP']
    lidar = nusc.get('sample_data', lidar_token)
    cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_front_extrinsics = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
    lidar_extrinsics = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    lidar_origin = np.array(lidar_extrinsics['translation'])
    box_detections = nusc.get_boxes(lidar_token)
    pcd_path = nusc.get_sample_data_path(lidar_token)
    pcd_labels_path = os.path.join('/media/storage/datasets/nuscenes-v1.0-mini/lidarseg/v1.0-mini', lidar_token + '_lidarseg.bin')
    ego = nusc.get("ego_pose", lidar_token)
    car_rotation = Quaternion(ego["rotation"])

    pose = pose_dataset.get_data_at_timestamp(lidar["timestamp"])
    car_trajectory.update(pose)

    static_lidar_geometry, dynamic_lidar_geometry = create_lidar_geometries(
        pcd_path, 
        pcd_labels_path, 
        colourmap, 
        static_object_ids
    )

    pos = car_trajectory.position.copy()
    pos[0], pos[1] = rotate_2d_vector(pos[0], pos[1], radians(-90)) #No idea why it needs rotating -90 degrees, maybe because car forward is actually X not Y?
    static_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
    static_lidar_geometry.translate(lidar_origin, relative=True)
    static_lidar_geometry.translate(pos, relative=True)
    dynamic_lidar_geometry.rotate(car_rotation.rotation_matrix, [0,0,0])
    dynamic_lidar_geometry.translate(lidar_origin, relative=True)
    dynamic_lidar_geometry.translate(pos, relative=True) 
    
    point_cloud_timeseries.add_geometry(static_lidar_geometry)    

    box_meshes = generate_boxes_meshes(box_detections, ego["translation"], pos)
    for box in box_meshes:
        cloud_window.add_geometry(box, False)

    cloud_window.add_geometry(point_cloud_timeseries.combined_geometry, CALCULATE_BBOX)
    cloud_window.add_geometry(dynamic_lidar_geometry, CALCULATE_BBOX)

    fx = cam_front_extrinsics["camera_intrinsic"][0][0]
    fy = cam_front_extrinsics["camera_intrinsic"][1][1]

    cam_hfov = 2 * np.arctan2(CAM_IMAGE_WIDTH, 2 * fx)
    cam_vfov = 2 * np.arctan2(CAM_IMAGE_HEIGHT, 2 * fy)

    camera_pos = np.array(cam_front_extrinsics["translation"]) + pos
    frustum = create_frustum_geometry(
        camera_pos,
        car_rotation.rotation_matrix,
        cam_hfov,
        cam_vfov,
        100
    )
    cloud_window.add_geometry(frustum, False)

    ##Generate occupancy
    camera_occupancy_grid_pos = camera_pos.copy()
    camera_occupancy_grid_pos[2] = 0
    occupancy_grid_pointcloud, occupancy_bounding_box = generate_camera_view_voxel_grid(
        [point_cloud_timeseries.combined_geometry, dynamic_lidar_geometry], 
        camera_occupancy_grid_pos, 
        car_rotation.rotation_matrix, 
        box_meshes,
        35, 
        35, 
        10
    )
    cloud_window.add_geometry(occupancy_bounding_box, CALCULATE_BBOX)

    occupancy_grid_pointcloud.translate([0, 0, 0], False)
    occupancy_grid_pointcloud.rotate(car_rotation.inverse.rotation_matrix)

    occupancy_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(occupancy_grid_pointcloud, voxel_size=voxel_size)
    occupancy_window.add_geometry(occupancy_grid, CALCULATE_BBOX)

    #DEBUG
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    mesh_frame.rotate(car_rotation.rotation_matrix, [0,0,0])
    mesh_frame.translate(pos)
    cloud_window.add_geometry(mesh_frame, False)
    
def main(args):
    global CALCULATE_BBOX

    inp = Input()

    #Create window visualizer
    point_cloud_window = o3d.visualization.Visualizer()
    occupancy_window = o3d.visualization.Visualizer()
    point_cloud_window.create_window("Point cloud", 1920, 1080, 0, 0)
    occupancy_window.create_window("Occupancy", 500, 500, 1920, 0)

    point_cloud_window.get_render_option().background_color = np.asarray([0, 0, 0])
    occupancy_window.get_render_option().background_color = np.asarray([0, 0, 0])

    #Load nuscenes data
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataset_root, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=args.dataset_root)
    colourmap = {}

    static_object_ids = [ 0, 13, 24, 25, 26, 27, 28, 29, 30 ]

    for index, name in nusc.lidarseg_idx2name_mapping.items():
        colour = nusc.colormap[name]
        colourmap[index] = colour
    
    scene = nusc.scene[args.scene_index]
    sample = nusc.get("sample", scene["first_sample_token"])
    pose = nusc_can.get_messages(scene["name"], 'pose')
    pose_dataset = TimestmapData(pose, [p["utime"] for p in pose])
    car_trajectory = CarTrajectory()
    point_cloud_timeseries = PointCloudTimeseries()

    draw_lidar_data(point_cloud_window, occupancy_window, nusc, sample, colourmap, static_object_ids, car_trajectory, pose_dataset, point_cloud_timeseries, args.voxel_size) 
    image = cv2.imread(os.path.join(args.dataset_root, nusc.get('sample_data', sample['data']['CAM_FRONT'])["filename"]))
    CALCULATE_BBOX = False

    while(True):
        try:
            if(not point_cloud_window.poll_events() or inp.get_key_down("q")):
                break


            occupancy_window.poll_events()

            if(inp.get_key_down("space")):
                if(sample["next"] == str()):
                    continue

                sample = nusc.get("sample", sample["next"]) 
                point_cloud_window.clear_geometries()
                occupancy_window.clear_geometries()
                draw_lidar_data(point_cloud_window, occupancy_window, nusc, sample, colourmap, static_object_ids, car_trajectory, pose_dataset, point_cloud_timeseries, args.voxel_size)
                image = cv2.imread(os.path.join(args.dataset_root, nusc.get('sample_data', sample['data']['CAM_FRONT'])["filename"]))
            
            if(inp.get_key_down("a")):
                if(sample["prev"] == str()):
                    continue

                sample = nusc.get("sample", sample["prev"]) 
                point_cloud_window.clear_geometries()
                occupancy_window.clear_geometries()
                draw_lidar_data(point_cloud_window, occupancy_window, nusc, sample, colourmap, static_object_ids, car_trajectory, pose_dataset, point_cloud_timeseries)
                image = cv2.imread(os.path.join(args.dataset_root, nusc.get('sample_data', sample['data']['CAM_FRONT'])["filename"]))
            
            point_cloud_window.update_renderer()
            occupancy_window.update_renderer()

            if(args.show_image):
                cv2.imshow("Image", image)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            break
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
