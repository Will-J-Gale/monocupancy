import os
import math
import shelve
from random import shuffle
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from monocupancy import Monocupancy

MODEL_INPUT_SIZE = (512, 256)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--training_data_path", required=True)
parser.add_argument("--validation_data_path", required=True)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--weights", default=None, type=str)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr_decay", default=0.97, type=float)
parser.add_argument("--batch_size", default=16, type=int)

class DataGenerator:
    def __init__(self, dataset, batchSize=32, device=0, imageSize=MODEL_INPUT_SIZE, add_flipped=True):
        self.dataset = dataset
        self.originalDatasetLength = dataset["metadata"]["length"]
        self.datasetLength = self.originalDatasetLength * 2 if add_flipped else self.originalDatasetLength
        self.grid_size = (
            dataset["metadata"]["num_width_voxels"], 
            dataset["metadata"]["num_height_voxels"],
            dataset["metadata"]["num_depth_voxels"]
        )
        self.datasetIndexes = np.arange(self.datasetLength)
        self.batchSize = batchSize
        self.device = device
        self.imageSize = imageSize

        self.inputBatch = []
        self.outputBatch = []

        self.shuffle()

    def loadData(self, data, flipped=False):
        images = []
        for path in data["image_paths"][-1:]:
            image = cv2.imread(path)
            image = cv2.resize(image, MODEL_INPUT_SIZE)
            if(flipped):
                image = np.fliplr(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
            image = np.moveaxis(image, 2, 0)
            images.append(image)

        images = np.vstack(images)
        occupancy_data = np.load(data["occupancy_path"])
        occupancy = occupancy_data["occupancy_grid"]

        if(flipped):
            occupancy = np.flip(occupancy, 0)

        return images, occupancy

    def onDataLoaded(self, future):
        inputImages, occupancyGrids = future.result()

        self.inputBatch.append(inputImages)
        self.outputBatch.append(occupancyGrids)

    def __len__(self):
        return math.ceil(self.datasetLength / self.batchSize)
    
    def __getitem__(self, index):
        start = index * self.batchSize
        end = start + self.batchSize
        
        self.inputBatch = []
        self.outputBatch = []
    
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in self.datasetIndexes[start:end]:
                dataset_index = i 
                flipped = False
                if(i >= self.originalDatasetLength):
                    dataset_index -= self.originalDatasetLength
                    flipped = True

                data = self.dataset[str(dataset_index)]
                job = executor.submit(self.loadData, data, flipped=flipped)
                job.add_done_callback(self.onDataLoaded)

        trainX = np.array(self.inputBatch, dtype=np.float32)
        trainY = np.array(self.outputBatch, dtype=np.float32)

        trainX = torch.from_numpy(trainX).to(self.device)
        trainY = torch.from_numpy(trainY).to(self.device)

        return trainX, trainY
    
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
    
    def shuffle(self):
        shuffle(self.datasetIndexes)

def main(args):
    model_name = "monocupancy"
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda")
    training_dataset_file = shelve.open(args.training_data_path, "r")
    val_dataset_file = shelve.open(args.validation_data_path)
    training_data = DataGenerator(training_dataset_file, args.batch_size, device, add_flipped=False)
    val_dataset = DataGenerator(val_dataset_file, args.batch_size, device, add_flipped=False)

    model = Monocupancy(3).cuda()
    if(args.weights):
        model.load_state_dict(torch.load(args.weights, weights_only=True))
    model.train()
    loss_function = BCEWithLogitsLoss(pos_weight=torch.tensor([50]).cuda())
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = ExponentialLR(optimizer, args.lr_decay)
    writer = SummaryWriter(log_dir="logs")

    writer.add_text("hyper params", str(dict(
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        weights=args.weights,
        learning_rate=args.lr,
        learning_rate_decay=args.lr_decay,
        batch_size=args.batch_size
    )))

    step = len(training_data) * args.start_epoch
    try:
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epoch"):
            writer.add_scalar("learning_rate", lr_decay.get_last_lr()[0], epoch)
            training_data.shuffle()
            epoch_losses = []
            epoch_progress = tqdm(
                total=len(training_data),
                bar_format="{n_fmt}/{total_fmt} {percentage:3.0f}%|{bar:20} | ETA: {elapsed}<{remaining} - Loss: {postfix}",
                postfix=0,
                leave=False
            )

            model.train()
            for train_X, train_Y in training_data:
                predictions = model(train_X)
                loss = loss_function(predictions, train_Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.cpu().detach().numpy()
                epoch_losses.append(loss_value)
                writer.add_scalar("training_loss", loss_value, step)
                step += 1
                epoch_progress.postfix = np.mean(epoch_losses)
                epoch_progress.update()

            lr_decay.step()
            
            avg_epoch_loss = np.mean(epoch_losses)
            writer.add_scalar("epoch_training_loss", avg_epoch_loss, epoch)

            model_filename = f"{model_name}_{epoch}_{avg_epoch_loss:.5f}.weights"
            torch.save(model.state_dict(), os.path.join(weights_dir, model_filename))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_X, val_Y in tqdm(val_dataset, desc="validation", leave=False):
                    predictions = model(val_X)
                    val_loss = loss_function(predictions, val_Y)

                    val_loss_value = val_loss.cpu().detach().numpy()
                    val_losses.append(val_loss_value)

            writer.add_scalar("validation_loss", np.mean(val_losses), epoch)

    except KeyboardInterrupt:
        print("Interrupted, closing")

    training_dataset_file.close()
    val_dataset_file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)