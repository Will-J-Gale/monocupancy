# Monocupancy
Monocular occupancy dataset generator

## Tools
* `tools/extract_nuscenes_data.py`
    * Extracts only the used data from nuscenes as loading nuscenes directly is huge in RAM
* `tools/interactive.py`
    * Runs the dataset generator in interactive mode
    * Displays the combined dense pointcloud from multiple frames
    * Displays resultant occupancy grid for the camera frustum
* `tools/view_occupancy_grid.py`
    * View the occupancy grid from the extracted dataset