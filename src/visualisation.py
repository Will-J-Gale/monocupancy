from typing import List

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
            point_cloud_vis_options:WindowOptions=WindowOptions(0, 0, 1920 // 2, 1080, "Dense point cloud"),
            occupancy_vis_options:WindowOptions=WindowOptions(1920 // 2, 0, 1920 // 2, 1080, "Occupancy")
        ):
        self.pointcloud_vis = o3d.visualization.Visualizer()
        self.occupancy_crop_vis = o3d.visualization.Visualizer()
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
        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0]) 
        self.reset()
    
    def add(self, point_cloud:List[o3d.geometry.PointCloud], occupancy:List[o3d.geometry.VoxelGrid]):
        self.add_pointcloud_geometry(point_cloud, self.reset_bounding_box)
        self.add_occupancy_geometry(occupancy, self.reset_bounding_box)
        self.reset_bounding_box = False
    
    def add_pointcloud_geometry(self, geometry:List[o3d.geometry.Geometry], reset_bounding_box:bool=False):
        if(not isinstance(geometry, list)):
            geometry = [geometry]

        for geo in geometry:
            self.pointcloud_vis.add_geometry(geo, reset_bounding_box)
        
    def add_occupancy_geometry(self, geometry:List[o3d.geometry.Geometry], reset_bounding_box:bool=False):
        if(not isinstance(geometry, list)):
            geometry = [geometry]

        for geo in geometry:
            self.occupancy_vis.add_geometry(geo, reset_bounding_box)
    
    def reset(self):
        self.pointcloud_vis.clear_geometries()
        self.occupancy_vis.clear_geometries()

        self.pointcloud_vis.add_geometry(self.origin, self.reset_bounding_box)
        self.occupancy_vis.add_geometry(self.origin, self.reset_bounding_box)
        
    def render(self):
        self.pointcloud_vis.update_renderer()
        self.occupancy_vis.update_renderer()
    
    def poll_events(self):
        self.occupancy_vis.poll_events()
        return self.pointcloud_vis.poll_events()