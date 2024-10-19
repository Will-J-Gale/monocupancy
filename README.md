# Monocupancy
Monocular occupancy dataset generator

## Setup
* `pip install requirements.txt`

### Download
* https://www.nuscenes.org/download  
* Make sure to download:
    * Full dataset
        * Mini
        * Trainval
        * Test
    * nusences-lidarseg
    * CAN bus expansion

## Generate dataset
* `tools/extract_nuscenes_data.py --dataset_root <PATH_TO_NUSCESES_ROOT> --dataset_version <NUSCENES_VERSON>`
    * `<NUSCENES_ROOT>`: Path to downloaded nuscenes dataset
    * `<NUSCENES_VERSON>`: v1.0-trainval, v1.0-mini or v1.0-test
    * Do this for both v1.0-trainval and v1.0-test
    * This takes quite a few hours, but once it's done, this data is much quicker to view/access
* `tools/generate_monocupancy.py --nuscenes_dataset nuscenes_simplified_dataset --output_path occupancy.dataset --occupancy_data_output_dir .`
    * Do this for both v1.0-trainval.dataset and v1.0-test.dataset
* `python tools/create_training_val_split.py occupancy.dataset`

## Train monocupancy
* `train/train_monocupancy.py --train_dataset occupancy_train.dataset --validate_dataset occupancy_val.dataset`

## Tools
* `tools/extract_nuscenes_data.py`
    * Nuscenes data takes quite a while to load as uses alot of RAM
    * This script just takes the necessary data and saves it in an easy to access file
* `tools/generate_monocupancy.py`
    * Uses the extracted data from `tools/extract_nuscenes_data.py` to generate monocupancy data
* `tools/interactive.py`
    * Runs the dataset generator in interactive mode
    * Displays the combined dense pointcloud from multiple frames
    * Displays resultant occupancy grid for the camera frustum
* `tools/view_occupancy_grid.py --dataset nuscenes_simplified.dataset --scene <SCENE_NUM>`
    * View the monocupancy dataset generated from `test/generate_monocupancy.py`

## Occupancy dataset format:
* Dataset is a shelve file: https://docs.python.org/3/library/shelve.html
```python
data = {
    "metadata": {
        "length": 1000,
        "num_occupied": 1000,
        "num_not_occupied": 9999,
        "voxel_size":0.3, #Units in car space
        "grid_width": 26, #Units in car space
        "grid_height": 12, #Units in car space
        "grid_depth": 36, #Units in car space
        "num_width_voxels": 120,
        "num_height_voxels": 40,
        "num_depth_voxels": 120
    },
    "0": {
        "image_paths": ["frame_1_path", "frame_2_path", "frame_3_path", "frame_4_path"],
        "occupancy_path": "path_to_occupancy.npz" #npz file contains numpy arrays for occupancy, colours and labels
    }
    "1": {...}
    "2": {...}
    ...
}
```