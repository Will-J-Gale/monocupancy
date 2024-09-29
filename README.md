# Monocupancy
Monocular occupancy dataset generator

## Setup
* Download nuscenes dataset

## Running
* Extract nuscenes dataset with `tools/extract_nuscenes_data.py`
* Generate monocupancy dataset with tools `tools/generate_monocupancy.py`

## Tools
* `tools/extract_nuscenes_data.py`
    * Extracts only the used data from nuscenes as loading nuscenes directly is huge in RAM
* `tools/generate_monocupancy.py`
    * Uses the extracted data from `extract_nuscenes_data` to generate monocupancy data
* `tools/interactive.py`
    * Runs the dataset generator in interactive mode
    * Displays the combined dense pointcloud from multiple frames
    * Displays resultant occupancy grid for the camera frustum
* `tools/view_occupancy_grid.py`
    * View the occupancy grid from monocupancy dataset