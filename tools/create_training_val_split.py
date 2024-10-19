import shelve
from random import shuffle
from argparse import ArgumentParser

import numpy as np

parser = ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument("--split", type=float, default=0.9)

def main(args):
    dataset = shelve.open(args.dataset_path, "r")
    metadata = dataset["metadata"]
    dataset_length = metadata["length"]

    split_index = round(dataset_length * args.split)
    indexes = np.arange(dataset_length)
    shuffle(indexes)

    training_indexes = indexes[:split_index]
    val_indexes = indexes[split_index:]

    training_data = shelve.open("occupancy_train.dataset", "c")
    val_data = shelve.open("occupancy_val.dataset", "c")

    training_metadata = metadata.copy()
    training_metadata["length"] = len(training_indexes)
    training_data["metadata"] = training_metadata

    for i, index in enumerate(training_indexes):
        training_data[str(i)] = dataset[str(index)]

    val_metadata = metadata.copy()
    val_metadata["length"] = len(val_indexes)
    val_data["metadata"] = val_metadata

    for i, index in enumerate(val_indexes):
        val_data[str(i)] = dataset[str(index)]
    
    training_data.close()
    val_data.close()
    dataset.close()

    print("Train dataset length: ", len(training_indexes))
    print("Val dataset length: ", len(val_indexes))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)