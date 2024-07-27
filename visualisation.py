import numpy as np
import open3d as o3d

class WindowOptions:
    def __init__(self, x:int, y:int, width:int, height:int, name:str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name

class Visualizer:
    def __init__(
            self,
            point_cloud_vis_options:WindowOptions=WindowOptions(0, 0, 1920, 1080, "Dense point cloud"),
            occupancy_vis_options:WindowOptions=WindowOptions(1920, 0, 500, 500, "Occupancy")):
        self.pointcloud_vis = o3d.visualization.Visualizer()
        self.occupancy_vis = o3d.visualization.Visualizer()

        self.pointcloud_vis.create_window(
            point_cloud_vis_options.name, 
            point_cloud_vis_options.width, 
            point_cloud_vis_options.height, 
            point_cloud_vis_options.x, 
            point_cloud_vis_options.y, 
        )
        self.pointcloud_vis.get_render_option().background_color = np.asarray([0, 0, 0])

        self.occupancy_vis.create_window(
            occupancy_vis_options.name, 
            occupancy_vis_options.width, 
            occupancy_vis_options.height, 
            occupancy_vis_options.x, 
            occupancy_vis_options.y, 
        )
        self.occupancy_vis.get_render_option().background_color = np.asarray([0, 0, 0])

        self.reset_bounding_box = True # Force camera to show whole lidar on first render
    
    def add_lidar(self, dense_lidar:o3d.geometry.PointCloud, occupancy:o3d.geometry.VoxelGrid):
        self.pointcloud_vis.add_geometry(dense_lidar, self.reset_bounding_box)
        self.occupancy_vis.add_geometry(occupancy, self.reset_bounding_box)
        self.reset_bounding_box = False
    
    def add_pointcloud_geometry(self, geometry:o3d.geometry.Geometry, reset_bounding_box:bool=False):
        self.pointcloud_vis.add_geometry(geometry, reset_bounding_box)

    def reset(self):
        self.pointcloud_vis.clear_geometries()
        self.occupancy_vis.clear_geometries()

    def render(self):
        self.pointcloud_vis.update_renderer()
        self.occupancy_vis.update_renderer()
    
    def poll_events(self):
        self.occupancy_vis.poll_events()
        return self.pointcloud_vis.poll_events()