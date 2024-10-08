import shelve
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("dataset_path")

def generate_occupancy_grid(data, grid_shape):
    occupancy = np.zeros(grid_shape)
    occupancy_indexes = data["occupancy"]

    # print(len(occupancy_indexes))
    for index in occupancy_indexes:
        x, y, z = index
        occupancy[x, z, y] = 1 #Why/What?!

    train_Y = np.array(occupancy, dtype=np.float32)

    return train_Y

def main(args):
    # frames = [2350, 11847, 17145] #3 frames selected at random
    epochs = 1000
    dataset = shelve.open(args.dataset_path, "r")
    metadata = dataset["metadata"]
    dataset_length = metadata["length"]
    num_width_voxels = metadata["num_width_voxels"]
    num_height_voxels = metadata["num_height_voxels"]
    num_depth_voxels = metadata["num_depth_voxels"]

    not_occupied = 0
    occupied = 0

    for i in tqdm(range(dataset_length)):
        data = dataset[str(i)]
        grid = generate_occupancy_grid(data, (num_width_voxels, num_height_voxels, num_depth_voxels))
        occupied += np.count_nonzero(grid == 1.0)
        not_occupied += np.count_nonzero(grid == 0.0)
    
    print(not_occupied, occupied)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)